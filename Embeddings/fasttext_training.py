
import gensim
from embedding import Embedding
import PATHS

class FastText(Embedding):
    def __init__(self):
        Embedding.__init__(self)

    def train(self):
        with open(PATHS.FASTTEXT_STATS_FULL,'w') as f:
            f.write("%s words total, with a vocabulary size of %s" % (len(self.all_words), len(self.VOCAB)))
            f.write("Max tweet length is %s" % max(self.sentence_lengths))

        fst_model = gensim.models.FastText(self.processed_tweets, # replace with actual data
                            size=300, 
                            window=5, 
                            alpha=0.03,
                            min_alpha=0.0007,
                            min_count=0, 
                            workers=self.cores,
                            sg=1)

        fst_model.train(self.processed_tweets,
                        total_examples=fst_model.corpus_count, 
                        epochs=30, 
                        report_delay=1)
        
        ## L2 nrom to save memory
        fst_model.init_sims(replace=True)

        ## Save Model
        fst_model.wv.save_word2vec_format(PATHS.FASTTEXT_VECTORS_FULL)