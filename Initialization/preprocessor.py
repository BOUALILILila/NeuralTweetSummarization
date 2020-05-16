import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
import nltk.data
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from profanityfilter import ProfanityFilter
import pandas as pd
import logging  # Setting up the loggings 

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

URL_REGEX=r"http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+"

def remove_duplicated_letters(text):
    return re.sub(r'(.)\1+', r'\1\1', text)
        
## Remove tweet if True:
def isTrash(text):
    # has profane vocab
    pf=ProfanityFilter()
    if pf.is_profane(text):
        return True
    # trash filtering
    urls=re.findall(URL_REGEX, text)
    if len(urls)> 1:
        return True
    # > 3 hashtags
    hashtags=re.findall(r"#\S+",text)
    if len(hashtags)> 3:
        return True
    # > 2 usernames
    users=re.findall(r"@\S+",text)
    if len(users)> 2:
        return True
    return False

def strip_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text) # no emoji

def preProcessText(text):
    text=strip_emojis(text)
    ## 1- Remove URLs
    text=re.sub(URL_REGEX, " ",text)
    ## 2- Remove usernames
    text=re.sub(r"@\S+"," ",text)
    ## 3- Remove non letters
    text=re.sub("[^a-zA-Z]", " ",text)
    ## 4- Lower case
    text=text.lower()
    ## 5- Remove Stop words
    stop_words=set(stopwords.words('english')) 
    return ' '.join([word for word in text.split() if word not in stop_words])

def stemTokens(tokens):
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def lemmatize_verbs(word):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word, pos='v')
    return lemma