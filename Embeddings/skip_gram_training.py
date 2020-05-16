
import gensim
from embedding import Embedding
import PATHS

class Skip_gram(Embedding):
    def __init__(self):
        Embedding.__init__(self)

    def train(self):
        with open(PATHS.STATS_SKIP_GRAM,'w') as f:
            f.write("%s words total, with a vocabulary size of %s" % (len(self.all_words), len(self.VOCAB)))
            f.write("Max tweet length is %s" % max(self.sentence_lengths))

         ## Model
        w2v_model = gensim.models.Word2Vec(sg=1, #sg=0 cbow
                                        min_count=0,
                                        window=5,
                                        size=300,
                                        sample=0, # downsampling most-common words (already removed stop-words)
                                        alpha=0.03,
                                        min_alpha=0.0007,
                                        negative=2,
                                        workers=self.cores)

        ## Build the model with the vocab
        w2v_model.build_vocab(self.processed_tweets,
                            progress_per=1000000)

        ## Train the model
        w2v_model.train(self.processed_tweets, 
                        total_examples=w2v_model.corpus_count, 
                        epochs=30, # change epochs 30
                        report_delay=1)
        
        ## L2 nrom to save memory
        w2v_model.init_sims(replace=True)

        ## Save Model
        w2v_model.wv.save_word2vec_format(PATHS.SKIP_GRAM_VECTORS)
