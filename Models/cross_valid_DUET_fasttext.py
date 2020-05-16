import matchzoo as mz
import typing
import csv
import os
import keras
import pandas as pd
import nltk
import data_loader as loader

dataset =loader.load_data(stage="train",path="/projets/iris/PROJETS/lboualil/CORPUS/tweets/train_data_2016_new.csv")


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

w2v_path=os.path.join("/projets/iris/PROJETS/lboualil/CORPUS/embeddings","full","fasttext/CBOW/vectors.txt")
#embedding = mz.datasets.embeddings.load_glove_embedding(dimension=50)
#w2v_path=os.path.join("/projets/iris/PROJETS/lboualil/CORPUS/embeddings/word2vec","vectors.txt")
embedding=mz.embedding.load_from_file(w2v_path)
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

    model = mz.models.DUET()
    model.params['input_shapes'] = preprocessor.context['input_shapes']
    model.params['task'] = ranking_task
    model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
    model.params['embedding_output_dim'] = 300
    model.params['lm_filters'] = 32
    model.params['lm_hidden_sizes'] = [32]
    model.params['dm_filters'] = 32
    model.params['dm_kernel_size'] = 3
    model.params['dm_d_mpool'] = 4
    model.params['dm_hidden_sizes'] = [32]
    model.params['dropout_rate'] = 0.4
    model.params['optimizer'] = 'adagrad'
    model.guess_and_fill_missing_params()
    model.build()
    model.compile()
    model.backend.summary()

    matrix = embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
    model.load_embedding_matrix(matrix)
    
    pred_x, pred_y = predict_pack_processed[:].unpack()
    evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_y))
    train_generator = mz.PairDataGenerator(train_pack_processed, num_dup=2, num_neg=1, batch_size=20)
    history = model.fit_generator(train_generator, epochs=30, callbacks=[evaluate], workers=30, use_multiprocessing=True)
    # evaluate the model
    scores = model.evaluate(pred_x, pred_y,batch_size=len(pred_y))
    mrrscores.append(scores[mz.engine.parse_metric('mrr')] * 100)
    mapscores.append(scores[mz.metrics.MeanAveragePrecision()] * 100)
    ndcgscores.append(scores[mz.metrics.NormalizedDiscountedCumulativeGain(k=1)] * 100)
print("\n>>> Resultat mrr:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mrrscores), numpy.std(mrrscores)))
print("\n>>> Resultat map:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mapscores), numpy.std(mapscores)))
print("\n>>> Resultat ndcg@1:  %.2f%% (+/- %.2f%%)" % (numpy.mean(ndcgscores), numpy.std(ndcgscores)))