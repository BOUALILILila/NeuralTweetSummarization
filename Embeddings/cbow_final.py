
import gensim
from embedding import Embedding
import PATHS
import glob
import json
import multiprocessing
import os

## Count the number of cores in a computer
cores = multiprocessing.cpu_count() 
cores= (cores,4)[cores==1]

sent_len=0
word_count=0
class MyIter(object):
     def __iter__(self):
        global sent_len, word_count
        dir_path=os.path.join(PATHS.TRAIN_TWEETS,'train_data_20*.json')
        #dir_path='/home/lila/pfe/test_dir/*.json'
        files=glob.glob(dir_path)
        for file_path in files:
            with open(file_path) as fin:
                sentences=[]
                sentences=json.load(fin)
                for s in sentences:
                    l=len(s)
                    word_count+=l
                    if(len(s)>sent_len):
                        sent_len=l
                    yield s
  
w2v_model = gensim.models.Word2Vec(sg=0, #sg=0 cbow
                                    min_count=0,
                                    window=5,
                                    size=300,
                                    sample=0, # downsampling most-common words (already removed stop-words)
                                    alpha=0.03,
                                    min_alpha=0.0007,
                                    negative=2,
                                    workers=cores)

w2v_model.build_vocab(sentences=MyIter())
total_examples = w2v_model.corpus_count
print('corpus count: ',total_examples)
print('\nvocab', len(w2v_model.wv.vocab))
print('\nall owrds:', word_count)
print('\nmax sent:', sent_len)

with open(PATHS.CBOW_STATS_FULL,'w') as f:
            f.write(f'Corpus count (tweet count)= {total_examples}')
            f.write(f'\nTotal words= {word_count}')
            f.write(f'\nVocabularay size= {len(w2v_model.wv.vocab)}')
            f.write(f"\nMax tweet length= {sent_len}")

w2v_model.train(sentences=MyIter(),
                total_examples=total_examples, 
                epochs=30, #30
                report_delay=1)
print('model: ', w2v_model)
## L2 nrom to save memory
w2v_model.init_sims(replace=True)

## Save Model
w2v_model.wv.save_word2vec_format(PATHS.CBOW_VECTORS_FULL)