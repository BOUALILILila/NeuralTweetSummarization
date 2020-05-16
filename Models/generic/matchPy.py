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
parser.add_argument('-t', required=True, metavar='test', help='test data')
parser.add_argument('-m', required=True, metavar='model', help='path to save model')
parser.add_argument('-p', required=True, metavar='preprocessor', help='path to save preprocessor')

args = parser.parse_args()
data_path = vars(args)['d']
embed_path = vars(args)['e']
test_path = vars(args)['t']
model_path = vars(args)['m']
pre_path = vars(args)['p']

dataset =loader.load_data(stage="train",path=data_path)

ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.engine.parse_metric('mrr'),
    mz.metrics.MeanAveragePrecision(),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=1),
    mz.metrics.Precision(k=10)
]

embedding = mz.embedding.load_from_file(embed_path)

#create preprocessor
train_split=dataset
test_split=loader.load_data(stage="train",path=test_path)
preprocessor=mz.preprocessors.BasicPreprocessor(fixed_length_left=10,
                                                fixed_length_right=128,
                                                filter_low_freq=0,
                                                filter_high_freq=1000,
                                                remove_stop_words=False)

    # fit on train data
train_pack_processed=preprocessor.fit_transform(train_split)
predict_pack_processed = preprocessor.transform(test_split)

model = mz.models.MatchPyramid()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = ranking_task
model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
model.params['embedding_output_dim'] = 300
model.params['embedding_trainable'] = True
model.params['num_blocks'] = 2
model.params['kernel_count'] = [16, 32]
model.params['kernel_size'] = [[3, 3], [3, 3]]
model.params['dpool_size'] = [3, 10]
model.params['optimizer'] = 'adagrad'
model.params['dropout_rate'] = 0.4
model.guess_and_fill_missing_params()
model.build()
model.compile()
model.backend.summary()
matrix = embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
model.load_embedding_matrix(matrix)
train_generator = mz.DPoolPairDataGenerator(train_pack_processed,
                                            fixed_length_left=10,
                                            fixed_length_right=128,
                                            num_dup=2,
                                            num_neg=0,
                                            batch_size=20)
predict_generator = mz.DPoolDataGenerator(predict_pack_processed,
                                          fixed_length_left=10,
                                          fixed_length_right=128,
                                          batch_size=20)
pred_x, pred_y = predict_generator[:]
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_y)) 
history = model.fit_generator(train_generator, epochs=30, workers=30, use_multiprocessing=True)

scores = model.evaluate(pred_x, pred_y,batch_size=len(pred_y))

print("\n>>> Resultat mrr:  %.2f%% " % (scores[mz.engine.parse_metric('mrr')] * 100))
print("\n>>> Resultat map:  %.2f%% " % (scores[mz.metrics.MeanAveragePrecision()] * 100))
print("\n>>> Resultat ndcg@1:  %.2f%% " % (scores[mz.metrics.NormalizedDiscountedCumulativeGain(k=1)] * 100))
print("\n>>> Resultat precision@10:  %.2f%% " % (scores[mz.metrics.Precision(k=10)] * 100))
# save the model
model.save(model_path)
preprocessor.save(pre_path)
    
