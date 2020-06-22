import requests
import logging
from math import ceil
from itertools import chain
from collections import Iterable, ChainMap
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import pandas as pd
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

BASE_URL = "https://api.mercadolibre.com/"
BASE_URL = "https://api.mercadolibre.com/"
MAX_OFFSET = 10000
OFFSET = 50
MAX_REQUESTS = ceil(MAX_OFFSET / OFFSET)
APP_INFO = dict(app_id=7214923947282925, secret_key="")
AUTH = {"refresh_token": ""}
THREADS = cpu_count()  # Get computer avaiable threads
THREAD_POOL = ThreadPool(THREADS)


def query_api(endpoint, filter_id=None, attr=None, query_params={}):
    """General function to query the API

    Parameters
    ----------
    endpoint : str
        Main API endpoint
    filter_id : str, optional
        Filter id (e.g $SITE_ID, $ITEM_ID) required for some endpoints, by default None
    attr : str, optional
        Attribute to extract from base response, by default None
    query_params : dict, optional
        Query values, parameters, and filters, by default {}

    Returns
    -------
    dict
        Json with API response
    """
    adapter = HTTPAdapter(max_retries=Retry(total=3))
    session = requests.Session()
    session.mount("https://", adapter)
    query = parse_query(endpoint, filter_id, attr, query_params)
    logging.debug(f"Querying {query}")
    resp = session.get(query, timeout=20)
    if resp.ok:
        response = resp.json()
        return response
    else:
        raise Exception(resp.json())


def parse_query(endpoint, filter_id=None, attr=None, query_params={}):
    args = list(filter(lambda x: x is not None, [endpoint, filter_id, attr]))
    query = BASE_URL + "/".join(args) + ("?" if query_params else "")
    query += "&".join(
        f"{key}={','.join(value) if isinstance(value, list) else value}"
        for key, value in query_params.items()
        if query_params and (key and value) is not None
    )
    return query


def get_token(app_id, secret_key, refresh_token):
    params = dict(endpoint="oauth", attr="token")
    params["query_params"] = {
        "grant_type": "refresh_token",
        "client_id": app_id,
        "client_secret": secret_key,
        "refresh_token": refresh_token,
    }
    query = parse_query(**params)
    req = requests.post(query)
    resp = req.json()
    token = resp.get("access_token")
    return token


def chunks(ids, n=20):
    return [ids[i : i + n] for i in range(0, len(ids), n)]


def get_item_info(ids, token):
    ids = list(chunks(ids))
    params = [{"access_token": token, "ids": chunk} for chunk in ids]
    try:
        results = THREAD_POOL.map(
            lambda x: query_api(endpoint="items", query_params=x), params
        )
        results = list(chain(*results))
        results = [res.get("body") for res in results]
        return results
    except Exception:
        logging.exception("A request failed")


def get_all_request_data(endpoint, filter_id=None, attr=None, query_params={}):
    new_params = []
    for offset in range(0, MAX_OFFSET + 1, OFFSET):
        query_params["offset"] = offset
        new_params.append(query_params.copy())
    try:
        results = THREAD_POOL.map(
            lambda x: query_api(endpoint, filter_id, attr, x), new_params
        )
        results = [res.get("results") for res in results]
        results = list(chain(*results))
        return results
    except Exception:
        logging.exception("A request failed")


def get_dataset(products, token):
    results = []
    for item in products:
        logging.info(f"Getting data for {item}")
        params = dict(
            endpoint="sites",
            filter_id="MLA",
            attr="search",
            query_params={"q": item, "access_token": token},
        )
        results += get_all_request_data(**params)
    search_results = pd.json_normalize(results)
    logging.info("Filtering ids")
    correct_domains = search_results["domain_id"].value_counts()[:5].index
    search_results = search_results[search_results["domain_id"].isin(correct_domains)]
    ids = search_results["id"].values.tolist()
    logging.info("Requesting ids")
    items_results = get_item_info(ids, token)
    final_df = pd.json_normalize(items_results)
    return final_df


# Functions for requesting subcategories (not used)
def add_main_category(response, query_params):
    if query_params["category"] is None:
        raise Exception("A category must me supplied")
    for result in response["results"]:
        result["main_category"] = query_params["category"]
    return response


def get_categories_subcategories(categories):
    full_cats = {}
    for i, category in enumerate(categories):
        logging.debug(f"Category {i} of {len(categories)}")
        category_id = category["id"]
        resp = query_api(endpoint="categories", filter_id=category_id)
        subcats = [subcat.get("id") for subcat in resp["children_categories"]]
        full_cats[category_id] = subcats
    return full_cats
