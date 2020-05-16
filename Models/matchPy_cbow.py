import matchzoo as mz
import typing
import csv
import os
import keras
import pandas as pd
import nltk
import data_loader as loader

dataset =loader.load_data(stage="train",path='/projets/iris/PROJETS/lboualil/CORPUS/tweets/worse_bm25/train_data_2015_2016_2017.csv')
#dataset =loader.load_data(stage="train",path='/projets/iris/PROJETS/lboualil/CORPUS/tweets/train_data_2016+2015.csv')

ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.engine.parse_metric('mrr'),
    mz.metrics.MeanAveragePrecision(),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=1),
    mz.metrics.Precision(k=10)
]

w2v_path=os.path.join("/projets/iris/PROJETS/lboualil/CORPUS/embeddings","full","CBOW/vectors.txt")
embedding = mz.embedding.load_from_file(w2v_path)

#create preprocessor
train_split=dataset
test_split=loader.load_data(stage="train",path='/projets/iris/PROJETS/lboualil/CORPUS/tweets/worse_bm25/test_data_2017_nist.csv')
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
                                            num_neg=1,
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
model.save('/projets/iris/PROJETS/lboualil/Models/MatchPy/MatchPy_full_2015-2016_fasttext_x')
preprocessor.save('/projets/iris/PROJETS/lboualil/Preprocessors/MatchPy/MatchPy_full_2016_fasttext_x')
    
