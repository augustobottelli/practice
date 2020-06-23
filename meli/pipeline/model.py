import logging
import math
from itertools import chain
from collections import Iterable, ChainMap
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib
from lightgbm import LGBMRegressor


BASE_URL = "https://api.mercadolibre.com/"
MAX_OFFSET = 10000
OFFSET = 50
MAX_REQUESTS = math.ceil(MAX_OFFSET / OFFSET)
APP_INFO = dict(app_id=7214923947282925, secret_key="")
AUTH = {"refresh_token": ""}
THREADS = cpu_count()  # Get computer avaiable threads
THREAD_POOL = ThreadPool(THREADS)


class MeliClient:
    def __init__(self, app_id, secret_key, refresh_token):
        self.session = self.make_session()
        self.token = self.get_token(app_id, secret_key, refresh_token)

    def make_session(self):
        adapter = HTTPAdapter(max_retries=Retry(total=3))
        session = requests.Session()
        session.mount("https://", adapter)
        return session

    def get_token(self, app_id, secret_key, refresh_token):
        params = dict(endpoint="oauth", attr="token")
        params["query_params"] = {
            "grant_type": "refresh_token",
            "client_id": app_id,
            "client_secret": secret_key,
            "refresh_token": refresh_token,
        }
        query = self.parse_query(**params)
        req = self.session.post(query)
        resp = req.json()
        token = resp.get("access_token")
        return token

    def parse_query(self, endpoint, filter_id=None, attr=None, query_params={}):
        args = list(filter(lambda x: x is not None, [endpoint, filter_id, attr]))
        query = BASE_URL + "/".join(args) + ("?" if query_params else "")
        query += "&".join(
            f"{key}={','.join(value) if isinstance(value, list) else value}"
            for key, value in query_params.items()
            if query_params and (key and value) is not None
        )
        return query

    def query_api(self, endpoint, filter_id=None, attr=None, query_params={}):
        """General function to query the API

        Parameters
        ----------
        endpoint : str
            Main API endpoint
        filter_id : str, optional
            Filter id (e.g $SITE_ID) required for some endpoints, by default None
        attr : str, optional
            Attribute to extract from base response, by default None
        query_params : dict, optional
            Query values, parameters, and filters, by default {}

        Returns
        -------
        dict
            Json with API response
        """

        query = self.parse_query(endpoint, filter_id, attr, query_params)
        logging.debug(f"Querying {query}")
        resp = self.session.get(query, timeout=20)
        if resp.ok:
            response = resp.json()
            return response
        else:
            raise Exception(resp.json())


class Dataset:
    """Requests all the information necessary to train the model"""

    def __init__(self, client):
        self.client = client
        self.token = client.token

    def get_all_request_data(
        self, endpoint, filter_id=None, attr=None, query_params={}
    ):
        new_params = []
        for offset in range(0, MAX_OFFSET + 1, OFFSET):
            query_params["offset"] = offset
            new_params.append(query_params.copy())
        try:
            results = THREAD_POOL.map(
                lambda x: self.client.query_api(endpoint, filter_id, attr, x),
                new_params,
            )
            results = [res.get("results") for res in results]
            results = list(chain(*results))
            return results
        except Exception:
            logging.exception("A request failed")

    def chunks(self, ids, n=20):
        return [ids[i : i + n] for i in range(0, len(ids), n)]

    def get_item_info(self, ids):
        ids = list(self.chunks(ids))
        params = [{"access_token": self.token, "ids": chunk} for chunk in ids]
        try:
            results = THREAD_POOL.map(
                lambda x: self.client.query_api(endpoint="items", query_params=x),
                params,
            )
            results = list(chain(*results))
            results = [res.get("body") for res in results]
            return results
        except Exception:
            logging.exception("A request failed")

    def filter_ids(self, results):
        correct_domains = results["domain_id"].value_counts()[:5].index
        results = results[results["domain_id"].isin(correct_domains)]
        ids = results["id"].values.tolist()
        return ids

    def get_dataset(self, products):
        results = []
        for item in products:
            logging.info(f"Getting data for {item}")
            params = dict(
                endpoint="sites",
                filter_id="MLA",
                attr="search",
                query_params={"q": item, "access_token": self.token},
            )
            results += self.get_all_request_data(**params)
        search_results = pd.json_normalize(results)
        logging.info("Filtering ids")
        ids = self.filter_ids(search_results)
        logging.info("Requesting ids")
        items_results = self.get_item_info(ids)
        final_df = pd.json_normalize(items_results)
        return final_df


class DatasetModeler:
    """Filter transform and clean Dataset before supplying it to the model"""

    def __init__(self, dataset, numeric_features, categorical_features):
        self.dataset = dataset
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def clean(self):
        not_geolocation = [
            "seller_address.latitude",
            "seller_address.longitude",
            "geolocation.latitude",
            "geolocation.longitude",
        ]
        not_price = ["base_price"]
        not_base = [
            "id",
            "permalink",
            "thumbnail",
            "secure_thumbnail",
            "descriptions",
            "pictures",
        ]
        not_time = ["start_time", "historical_start_time"]
        not_seller = [
            "seller_address.city.id",
            "seller_address.state.name",
            "seller_address.search_location.neighborhood.name",
            "seller_address.search_location.city.name",
            "seller_address.search_location.state.name",
        ]
        uniques = {}
        categorical_features = list(
            self.dataset.drop(self.numeric_features, axis=1).columns
        )
        for col in categorical_features:
            uniques[col] = self.dataset[col].apply(str).nunique() / len(self.dataset)
        not_keys = [
            key for key, value in sorted(uniques.items(), key=lambda x: x[1])[:17]
        ]
        undesired_cols = (
            not_geolocation + not_price + not_seller + not_time + not_keys + not_base
        )
        self.dataset.drop(undesired_cols, axis=1, inplace=True)

    def log_transform(self, columns):
        for col in columns:
            self.dataset.loc[:, f"log_{col}"] = self.dataset[col].apply(
                lambda x: math.log(x + 1)
            )
        return self.dataset

    def time_parser(self):
        def get_days(timedelta):
            return timedelta.dt.days

        now = pd.Timestamp(datetime.utcnow(), tz="UTC")
        date_cols = ["date_created", "stop_time", "last_updated"]
        for date_col in date_cols:
            self.dataset[date_col] = pd.to_datetime(self.dataset["date_created"])
        self.dataset["days_old"] = get_days(now - self.dataset["date_created"])
        self.dataset["days_remaining"] = get_days(self.dataset["stop_time"] - now)
        self.dataset["days_from_update"] = get_days(now - self.dataset["last_updated"])
        self.dataset.drop(date_cols, axis=1, inplace=True)

    def text_parser(self):
        def string_parser(x):
            return x.lower().strip()

        def transform_title(x):
            words = string_parser(x).split(" ")[:2]
            parsed_words = " ".join(words)
            return parsed_words

        self.dataset.loc[:, "title"] = self.dataset["title"].apply(transform_title)
        self.dataset.loc[:, "seller_address.city.name"] = self.dataset[
            "seller_address.city.name"
        ].apply(string_parser)

        unwanted_seller = [
            "seller_address.city.id",
            "seller_address.state.name",
            "seller_address.search_location.neighborhood.name",
            "seller_address.search_location.city.name",
            "seller_address.search_location.state.name",
        ]
        self.dataset.drop(unwanted_seller, axis=1, inplace=True)

    def unpack_tags(self):
        self.dataset["tags"] = (
            self.dataset["tags"]
            .apply(lambda x: " ".join(x))
            .apply(lambda x: x.replace("-", "_"))
        )
        cvect = CountVectorizer()
        tags = cvect.fit_transform(self.dataset["tags"]).toarray()
        for i, col in enumerate(cvect.get_feature_names()):
            self.dataset[col] = tags[:, i]
        self.dataset.drop("tags", axis=1, inplace=True)

        attrs_df = pd.json_normalize(
            self.dataset["attributes"].apply(
                lambda lista: {d["id"]: d["value_name"] for d in lista}
            )
        )
        self.dataset = pd.concat([self.dataset, attrs_df], axis=1)
        self.dataset.drop("attributes", axis=1, inplace=True)
        self.dataset = self.dataset[~self.dataset["sold_quantity"].isna()]
        return self.dataset


class SalesEstimator:
    def __init__(self, dataset_modeler):
        self.dataset = dataset_modeler.dataset

    def estimator_pipeline(self):
        y_log = self.dataset["sold_quantity"].apply(lambda x: math.log(x + 1))
        X = self.dataset.drop("sold_quantity", axis=1)

        numeric_features = [
            "price",
            "discount",
            "available_quantity",
            "initial_quantity",
            "original_price",
            "days_old",
        ]

        X[numeric_features] = X[numeric_features].applymap(np.float)
        categorical_features = list(X.drop(numeric_features, axis=1).columns)
        X[categorical_features] = X[categorical_features].applymap(str)

        X_train, X_test, y_train, y_test = train_test_split(X, y_log)

        numeric_transformer = Pipeline(
            [("imputer", SimpleImputer(strategy="constant", fill_value=0),)]
        )
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        estimator = LGBMRegressor(n_jobs=-1)
        pipe = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
        self.pipeline = pipe.fit(X_train, y_train)
        cv = KFold(n_splits=5)
        self.scores = cross_val_score(pipe, X_test, y_test, cv = cv, scoring='r2')
        return self


def train_and_export(classes, items):
    """
    Train a model to predict `sold_quanty`. The function pickles the model pipeline
    only if the model has an equal or better cross validation score than the previous
    one. 
    """
    pass


def predict_item(model_path, item):
    """Predict `sold_quantity` for a given item"""
    pass
