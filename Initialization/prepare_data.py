import os
import errno
import json
import pandas as pd
import glob
from . import preprocessing_utils as pre
from nltk.tokenize import RegexpTokenizer
import logging
import html

#Initializations
tokenizer = RegexpTokenizer(r'\w+')


#---------------------------------
#Checks if the tweet is a retweets of a seen tweet
#>>Returns: True is so
def is_retweet(original_tweet,seen_ids):
    if original_tweet['id_str'] in seen_ids:
        return True
    return original_tweet


#---------------------------------
#Checks if the tweet is not trash:
#   1-has profane vocab
#   2-contains > 3 hashtags
#   3-contains > 2 usernames
#Preprocess the text of the tweet:
#   1-Remove URLs
#   2-Remove usernames
#   3-Remove non letters
#   4-Lower case
#   5-Remove Stop words
#Tokenize the text
#Remove isolated letters (len(token)==1)
#Check if it has at least 5 tokens
#Stemming
#
#>>Returns: the processed text if it passed the verfications
#                               else returns None
def trash_filter(text):
    text=html.unescape(text)
    text=pre.remove_duplicated_letters(text)
    if pre.isTrash(text)is False:
                            ## Preprocess, tokenize and stem
                            text=pre.preProcessText(text)
                            tokens=tokenizer.tokenize(text)
                            tokens=[token for token in tokens if len(token)>1]
                            # Starts with Rt for retweet
                            try:
                                if tokens[0] == 'rt' :
                                    tokens.pop(0)
                            except IndexError:
                                return 0
                            # more than 5 unique tokens
                            if(len(set(tokens))>=5):
                                stems=pre.stemTokens(tokens)
                                text=" ".join([stem for stem in stems])
                                return text
    return 0
                                
#--------------------------------
# Do trash filtering:
#   1-check if the record is a tweet
#   2-Not a response to another tweet
#   3-Not a retweet of a seen tweet  
#   4-trash filtering and preprocessing usinf the function trash_filter  
#              
#>>Returns: the processed tweets(list) 
#
def prepare_statuses(self,tweets):
    logging.info('\n# Preprocessing...\n')
    if len(tweets)==0:
        return [],0
    data=pd.DataFrame(tweets)
    data['text']=data['full_text']
    #Remvoe duplicated tweets
    c_id=(pd.isnull(data.id_str))
    data=data[~c_id]
    data.drop_duplicates('id_str',inplace=True)
    data.drop_duplicates('text',inplace=True)
    #Remove replies
    c_reply=(pd.isnull(data.in_reply_to_status_id))
    data=data[c_reply]
    #Get the tweets: not retweet
    c_retweet=(pd.isnull(data.retweeted_status))
    d1=data.copy()
    d1=d1[c_retweet]
    #Get the retweets
    data=data[~c_retweet]
    #trash_filter + preprocess d1
    d1['processed_text']=d1['text'].map(trash_filter)
    d1=d1[d1.processed_text!=0]
    #process data: retweets
    ids=ids=set(d1.loc[:,'id_str']) #seen tweets
    data['retweeted_status']=data['retweeted_status'].apply((lambda x: is_retweet(x,ids)))
    data=data[data.retweeted_status!=True]
    #trash_filter + preprocess data
    data['processed_text']=data['text'].map(trash_filter)
    data=data[data.processed_text!=0]
    final=pd.concat([data,d1],sort=True)
    del d1,data
    final.reset_index(inplace=True,drop=True)
    #final=final.loc[:,['id_str','processed_text']]
    final.drop_duplicates('processed_text',inplace=True)
    return final, len(tweets)


def prepare_query(self,query):
    print('q')
    text=query.strip()
    text=pre.preProcessText(text)
    tokens=tokenizer.tokenize(text)
    stems=pre.stemTokens(tokens)
    text=" ".join([stem for stem in stems])
    return text