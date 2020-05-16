import requests
from .Config import API_URL, API_KEY
import json
import logging
from .Config import test


def get_summarization(q, date):
    if test:
            with open('context.json') as f:
                resp=json.load(f)
            return resp
    else: 
        return call_api(q,date)

def call_api(q,date):
    params={"q":q,"since_date":date}
    headers = {"Authorization": f"Token {API_KEY}"}
    r = requests.get(API_URL, headers=headers, params=params)
    if r.status_code== requests.codes.ok :
        response=r.content
        return json.loads(response)
    
    elif r.status_code==400:
        response=r.content
        logging.info(f"# Bad Request: ",json.loads(response))
        return 400
    elif r.status_code==429:
        response=r.content
        logging.info(f"# Tweep problem: ",json.loads(response))
        return 429
    else:
        response=r.content
        logging.info(f"# Server problem: ",response)
        return 500