from django.db import models
from .PATHS import PREPROCESSOR, MODEL, EMBED
import matchzoo as mz
import gensim
import logging 

# Create your models here.
logging.info('\n# Loading Embeddings, Model...\n')
#pre=mz.engine.base_preprocessor.load_preprocessor(PREPROCESSOR)
#predict_model=mz.engine.base_model.load_model(MODEL)
embed_model=gensim.models.KeyedVectors.load_word2vec_format(EMBED, binary=False)