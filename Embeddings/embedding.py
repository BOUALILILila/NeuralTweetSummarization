from abc import ABC, abstractmethod
import json
import multiprocessing
import PATHS
import glob
import os

class Embedding(ABC):
 
    def __init__(self):
        self.processed_tweets=[]
        path=os.path.join(PATHS.TRAIN_TWEETS,'train_data_*.json')
        files = glob.glob(path)
        for fil in files:
            with open(fil, 'r') as f:
                self.processed_tweets+=json.load(f)
        ## Get stats about data
        #self.processed_tweets=[['hi','london'],['so','happy'],['hey','summer']]
        self.all_words = [word for tokens in self.processed_tweets for word in tokens]
        self.sentence_lengths = [len(tokens) for tokens in self.processed_tweets]
        self.VOCAB = sorted(list(set(self.all_words)))
        ## Count the number of cores in a computer
        cores = multiprocessing.cpu_count() 
        self.cores= (cores,4)[cores==1]
        super().__init__()
    
    @abstractmethod
    def train(self):
        pass
