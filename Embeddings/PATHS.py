import os
config=1
base_dir='/projets/iris/PROJETS/lboualil'

TRAIN_DATA_FOLDER_PATH = '/CORPUS/collection_2018/*.txt'
TOPICS_FOLDER_PATH='/CORPUS/topics/'
TRAIN_TWEETS_2018='/CORPUS/training_data_embeddings/train_data_2018.json'
STATS_SKIP_GRAM='/CORPUS/embeddings/word2vec/skip-gram/stats.txt'
SKIP_GRAM_VECTORS='/CORPUS/embeddings/word2vec/skip-gram/vectors.txt'
STATS_CBOW='/CORPUS/embeddings/word2vec/cbow/stats.txt'
CBOW_VECTORS='/CORPUS/embeddings/word2vec/cbow/vectors.txt'
STATS_FASTTEXT='/CORPUS/embeddings/fasttext/stats.txt'
FASTTEXT_VECTORS='/CORPUS/embeddings/fasttext/vectors.txt'
FASTTEXT_VECTORS_FULL_SG='/CORPUS/embeddings/full/fasttext/SKIP-GRAM/vectors.txt'
FASTTEXT_STATS_FULL_SG='/CORPUS/embeddings/full/fasttext/SKIP-GRAM/stats.txt'
FASTTEXT_VECTORS_FULL_CBOW='/CORPUS/embeddings/full/fasttext/CBOW/vectors.txt'
FASTTEXT_STATS_FULL_CBOW='/CORPUS/embeddings/full/fasttext/CBOW/stats.txt'
CBOW_VECTORS_FULL='/CORPUS/embeddings/full/CBOW/vectors.txt'
CBOW_STATS_FULL='/CORPUS/embeddings/full/CBOW/stats.txt'
TRAIN_TWEETS='/CORPUS/training_data_embeddings'
if(config):
    TRAIN_DATA_FOLDER_PATH = base_dir+ TRAIN_DATA_FOLDER_PATH
    TOPICS_FOLDER_PATH=base_dir+TOPICS_FOLDER_PATH
    TRAIN_TWEETS_2018=base_dir+TRAIN_TWEETS_2018
    STATS_SKIP_GRAM=base_dir+STATS_SKIP_GRAM
    SKIP_GRAM_VECTORS=base_dir+SKIP_GRAM_VECTORS
    STATS_CBOW=base_dir+STATS_CBOW
    CBOW_VECTORS= base_dir+CBOW_VECTORS
    STATS_FASTTEXT=base_dir+STATS_FASTTEXT
    FASTTEXT_VECTORS=base_dir+FASTTEXT_VECTORS
    FASTTEXT_VECTORS_FULL_SG=base_dir+FASTTEXT_VECTORS_FULL_SG
    FASTTEXT_STATS_FULL_SG=base_dir+FASTTEXT_STATS_FULL_SG
    FASTTEXT_VECTORS_FULL_CBOW=base_dir+FASTTEXT_VECTORS_FULL_CBOW
    FASTTEXT_STATS_FULL_CBOW=base_dir+FASTTEXT_STATS_FULL_CBOW
    CBOW_VECTORS_FULL=base_dir+CBOW_VECTORS_FULL
    CBOW_STATS_FULL=base_dir+CBOW_STATS_FULL
    TRAIN_TWEETS=base_dir+TRAIN_TWEETS