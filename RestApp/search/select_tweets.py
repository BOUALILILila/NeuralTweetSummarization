from . import search_twitter as st,prepare_tweets as pt, preprocessing_utils as pu
from .Config import N_TOP
from . import relevance_estimator as rel_estimator
from . import novelty_estimator as nov_estimator
import json
import pandas as pd
import logging

def get_summary(query,since,embed):
    tweets=st.get_tweets(query,since)
    logging.info('\n# Retrieved from Twitter API [OK]\n')
    processed_tweets,retrieved=pt.preporcess_searched_tweets(tweets)
    candidates=len(processed_tweets)
    summary=pd.DataFrame()
    if len(processed_tweets)>0:
        q=pu.preprocess_query(query)
        print(f'\n>> query: {q}\n')
        predictions=rel_estimator.predict_similarity_scores(processed_tweets,q,N_TOP)
        del processed_tweets
        summary=nov_estimator.delete_redudants(predictions,embed)
    return json.loads(summary.to_json(orient='records')),retrieved,candidates
