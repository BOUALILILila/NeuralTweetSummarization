import matchzoo as mz
import typing
import csv
import os
import keras
import pandas as pd
import nltk
import data_loader as loader
import sys
import argparse


parser = argparse.ArgumentParser(description='Match Pyramid neural model')
parser.add_argument('-d', required=True, metavar='data', help='train data')
parser.add_argument('-e', required=True, metavar='embeddings', help='embeddings')
#parser.add_argument('-m', required=True, metavar='model', help='path to save model')
#parser.add_argument('-p', required=True, metavar='preprocessor', help='path to save preprocessor')

args = parser.parse_args()
data_path = vars(args)['d']
embed_path = vars(args)['e']
#model_path = vars(args)['m']
#pre_path = vars(args)['p']

dataset =loader.load_data(stage="train",path=data_path)

from sklearn.model_selection import StratifiedKFold
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
mrrscores = []
mapscores=[]
ndcgscores=[]
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.engine.parse_metric('mrr'),
    mz.metrics.MeanAveragePrecision(),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=1)

]
X= dataset.frame()
Y=dataset.frame()['label']
#w2v_path=os.path.join("/projets/iris/PROJETS/lboualil/CORPUS/embeddings","full","fasttext/CBOW/vectors.txt")
embedding = mz.embedding.load_from_file(embed_path)
for train, test in kfold.split(X, Y):
    # create model
    #create preprocessor
    train_split=dataset[train]
    test_split=dataset[test]
    preprocessor=mz.preprocessors.BasicPreprocessor(fixed_length_left=10,
                                                fixed_length_right=128,
                                                filter_low_freq=0,
                                                filter_high_freq=1000,
                                                remove_stop_words=False)

    # fit on train data
    train_pack_processed=preprocessor.fit_transform(train_split)
    print(preprocessor.context)
    predict_pack_processed = preprocessor.transform(test_split)

    model = mz.models.ArcII()
    model.params['input_shapes'] = preprocessor.context['input_shapes']
    model.params['task'] = ranking_task
    model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
    model.params['embedding_output_dim'] = 300
    model.params['embedding_trainable'] = True
    model.params['num_blocks'] = 2
    model.params['kernel_1d_count'] = 32
    model.params['kernel_1d_size'] = 3
    model.params['kernel_2d_count'] = [64, 64]
    model.params['kernel_2d_size'] = [3, 3]
    model.params['pool_2d_size'] = [[3, 3], [3, 3]]
    model.params['optimizer'] = 'adam'
    model.guess_and_fill_missing_params()
    model.build()
    model.compile()
    model.backend.summary()

    matrix = embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
    model.load_embedding_matrix(matrix)
    
    pred_x, pred_y = predict_pack_processed[:].unpack()
    evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_y))
    len(pred_y)

    train_generator = mz.PairDataGenerator(train_pack_processed, num_dup=2, num_neg=1, batch_size=20)
    len(train_generator)

    history = model.fit_generator(train_generator, epochs=30, callbacks=[evaluate], workers=30, use_multiprocessing=True)
    # evaluate the model
    scores = model.evaluate(pred_x, pred_y,batch_size=len(pred_y))
    mrrscores.append(scores[mz.engine.parse_metric('mrr')] * 100)
    mapscores.append(scores[mz.metrics.MeanAveragePrecision()]*100)
    ndcgscores.append(scores[mz.metrics.NormalizedDiscountedCumulativeGain(k=1)]*100)
print("\n>>> Resultat mrr:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mrrscores), numpy.std(mrrscores)))
print("\n>>> Resultat map:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mapscores), numpy.std(mapscores)))
print("\n>>> Resultat ndcg@1:  %.2f%% (+/- %.2f%%)" % (numpy.mean(ndcgscores), numpy.std(ndcgscores)))