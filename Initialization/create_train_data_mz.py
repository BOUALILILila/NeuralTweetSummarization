import json
import pandas as pd

with open('/projets/iris/PROJETS/lboualil/CORPUS/left/left_2016.json') as f:
    left = json.load(f)
with open('/projets/iris/PROJETS/lboualil/CORPUS/right/right_2015.json') as f:
    right = json.load(f)
with open('/projets/iris/PROJETS/lboualil/CORPUS/relation/relation_2015.json') as f:
    relation = json.load(f)

left=pd.DataFrame.from_dict(left)
left.drop_duplicates('topid','first',True)
left.set_index('topid',inplace=True)

right=pd.DataFrame.from_dict(right)
right.drop_duplicates('tweetid','first',True)
right.set_index('tweetid',inplace=True)

## constrcut the csv file
rows=[]
for rel in relation:
    if rel['topid'] in left.index and rel['tweetid'] in right.index:
        topic_text=left.loc[rel['topid'],'title']
        tweet_text=right.loc[rel['tweetid'],'text']
        rows+=[{'topic_id':rel['topid'],
                'topic_text':topic_text,
                'tweet_id':rel['tweetid'],
                'tweet_text':tweet_text,
                'label':rel['label']
               }]
dataset=pd.DataFrame.from_dict(rows)
dataset.head()
dataset.to_csv("/projets/iris/PROJETS/lboualil/CORPUS/tweets/train_data_2015.csv")