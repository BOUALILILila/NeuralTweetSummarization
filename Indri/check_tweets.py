import json


with open('/home/lila/CORPUS/right/right_2017.json') as f:
    tweets=json.load(f)
tweet_ids=[tweet['tweetid'] for tweet in tweets]
tweet_ids=set(tweet_ids)

with open('/home/lila/CORPUS/relation/relation_2017.json') as rel:
    relations=json.load(rel)

with open('qrels_2017.txt','w') as qrels:
    for rel in relations:
        if rel['tweetid'] in tweet_ids:
            qrels.write(f"{rel['topid']} Q0 {rel['tweetid']} {rel['label']}\n")
            



