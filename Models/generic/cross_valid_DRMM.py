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
p_10scores=[]
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.engine.parse_metric('mrr'),
    mz.metrics.MeanAveragePrecision(),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=1),
        mz.metrics.Precision(k=10)
]
X= dataset.frame()
Y=dataset.frame()['label']

#w2v_path=os.path.join("/projets/iris/PROJETS/lboualil/CORPUS/embeddings","GoogleNews-vectors-negative300.txt")
#embedding = mz.datasets.embeddings.load_glove_embedding(dimension=50)
#w2v_path=os.path.join("/projets/iris/PROJETS/lboualil/CORPUS/embeddings","full","fasttext/SKIP-GRAM/vectors.txt")
embedding=mz.embedding.load_from_file(embed_path)
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

    bin_size = 20 # discretize the interval of cosine similarity between term vectors into a set of bins
    model = mz.models.DRMM()
    model.params['input_shapes'] = [[10,], [10, bin_size,]]
    model.params['task'] = ranking_task
    model.params['mask_value'] = 0
    model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
    model.params['embedding_output_dim'] = 300
    model.params['embedding_trainable']= True
    model.params['mlp_num_layers'] = 2
    model.params['mlp_num_units'] = 10
    model.params['mlp_num_fan_out'] = 1
    model.params['mlp_activation_func'] = 'tanh'
    model.params['optimizer'] = 'adadelta'
    model.guess_and_fill_missing_params()
    model.build()
    model.compile()
    model.backend.summary()

    matrix = embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
    model.load_embedding_matrix(matrix)
    
    pred_generator = mz.HistogramDataGenerator(data_pack=predict_pack_processed,
                                           embedding_matrix=matrix,
                                           bin_size=bin_size, 
                                           hist_mode='LCH')
    pred_x, pred_y = pred_generator[:]
    evaluate = mz.callbacks.EvaluateAllMetrics(model, 
                                            x=pred_x, 
                                            y=pred_y, 
                                            once_every=1, 
                                            batch_size=len(pred_y),
                                            #model_save_path='./drmm_pretrained_model/'
                                            )

    train_generator = mz.HistogramPairDataGenerator(train_pack_processed, matrix, bin_size, 'LCH',
                                                num_dup=2, num_neg=4, batch_size=20)

    
    # Fit the model
    history = model.fit_generator(train_generator, epochs=30,callbacks=[evaluate], workers=30, use_multiprocessing=True)

    # evaluate the model
    scores = model.evaluate(pred_x, pred_y,batch_size=len(pred_y))
    mrrscores.append(scores[mz.engine.parse_metric('mrr')] * 100)
    mapscores.append(scores[mz.metrics.MeanAveragePrecision()] * 100)
    ndcgscores.append(scores[mz.metrics.NormalizedDiscountedCumulativeGain(k=1)] * 100)
    p_10scores.append(scores[mz.metrics.Precision(k=10)]*100)
print("\n>>> Resultat mrr:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mrrscores), numpy.std(mrrscores)))
print("\n>>> Resultat map:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mapscores), numpy.std(mapscores)))
print("\n>>> Resultat ndcg@1:  %.2f%% (+/- %.2f%%)" % (numpy.mean(ndcgscores), numpy.std(ndcgscores)))
print("\n>>> Resultat p@10:  %.2f%% (+/- %.2f%%)" % (numpy.mean(p_10scores), numpy.std(p_10scores)))