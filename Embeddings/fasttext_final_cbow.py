
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
  
fst_model = gensim.models.FastText( 
                            size=300,  #300
                            window=5, 
                            alpha=0.03,
                            min_alpha=0.0007,
                            min_count=0, 
                            workers=cores,
                            word_ngrams=3,
                            sample=0,
                            negative=2,
                            sg=0)
                           
fst_model.build_vocab(sentences=MyIter())
total_examples = fst_model.corpus_count
print('corpus count: ',total_examples)
print('vocab', len(fst_model.wv.vocab))
print('all owrds:', word_count)
print('max sent:', sent_len)

with open(PATHS.FASTTEXT_STATS_FULL_CBOW,'w') as f:
            f.write(f'Corpus count (tweet count)= {total_examples}')
            f.write(f'Total words= {word_count}')
            f.write(f'Vocabularay size= {len(fst_model.wv.vocab)}')
            f.write(f"Max tweet length= {sent_len}")

fst_model.train(sentences=MyIter(),
                total_examples=total_examples, 
                epochs=30, #30
                report_delay=1)
print('model: ', fst_model)
## L2 nrom to save memory
fst_model.init_sims(replace=True)

## Save Model
fst_model.wv.save_word2vec_format(PATHS.FASTTEXT_VECTORS_FULL_CBOW)