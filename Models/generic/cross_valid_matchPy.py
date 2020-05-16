import matchzoo as mz
import typing
import csv
import os
import keras
import pandas as pd
import nltk
import data_loader as loader
import json
from sklearn.model_selection import StratifiedKFold
import numpy
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Match Pyramid neural model')
    parser.add_argument('-d', required=True, metavar='data', help='train data')
    parser.add_argument('-e', required=True, metavar='embeddings', help='embeddings')
    parser.add_argument('-o', required=True, metavar='output', help='output file path')
    #parser.add_argument('-m', required=True, metavar='model', help='path to save model')
    #parser.add_argument('-p', required=True, metavar='preprocessor', help='path to save preprocessor')

    args = parser.parse_args()
    data_path = vars(args)['d']
    embed_path = vars(args)['e']
    out_path = vars(args)['o']
    #model_path = vars(args)['m']
    #pre_path = vars(args)['p']


    dataset =loader.load_data(stage="train",path=data_path)

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # define 5-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    mrrscores = []
    mapscores=[]
    ndcgscores=[]
    p_10scores=[]
    p_50scores=[]
    p_100scores=[]
    p_200scores=[]
    ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
    ranking_task.metrics = [
        mz.engine.parse_metric('mrr'),
        mz.metrics.MeanAveragePrecision(),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=1),
        mz.metrics.Precision(k=10),
        mz.metrics.Precision(k=50),
        mz.metrics.Precision(k=100),
        mz.metrics.Precision(k=200)
    ]
    X= dataset.frame()
    Y=dataset.frame()['label']
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

        model = mz.models.MatchPyramid()
        model.params['input_shapes'] = preprocessor.context['input_shapes']
        model.params['task'] = ranking_task
        model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
        model.params['embedding_output_dim'] = 300
        model.params['embedding_trainable'] = True 
        model.params['num_blocks'] = 2
        model.params['kernel_count'] = [16,32]
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
        # Fit the model
        evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_y))
        history = model.fit_generator(train_generator, epochs=30,callbacks=[evaluate], workers=30, use_multiprocessing=True)

        # evaluate the model
        scores = model.evaluate(pred_x, pred_y,batch_size=len(pred_y))
        '''
        predictions=model.predict(pred_x)
        i=0
        x=pd.DataFrame(pred_x,columns=['id_left','id_right'])
        for index,row in x.iterrows():
            trec_rows+=[{"id_left":row['id_left'], "id_right": row['id_right'], "score":predictions[i][0]}]
            i+=1
        '''
        mrrscores.append(scores[mz.engine.parse_metric('mrr')] * 100)
        mapscores.append(scores[mz.metrics.MeanAveragePrecision()] * 100)
        ndcgscores.append(scores[mz.metrics.NormalizedDiscountedCumulativeGain(k=1)] * 100)
        p_10scores.append(scores[mz.metrics.Precision(k=10)]*100)
        p_50scores.append(scores[mz.metrics.Precision(k=50)]*100)
        p_100scores.append(scores[mz.metrics.Precision(k=100)]*100)
        p_200scores.append(scores[mz.metrics.Precision(k=200)]*100)
    '''
    pred_table=pd.DataFrame.from_dict(trec_rows)
    pred_table=pred_table.sort_values(['id_left','score'],ascending=[True,False])
    for index,row in pred_table.iterrows():
        trec_file.write(f"{row['id_left']} Q0 {row['id_right']} index {row['score']} matchPyCBOW\n")
    trec_file.close()
    '''
    #model.save(model_path)
    #preprocessor.save(pre_path)
    with open(out_path,'w') as f:
        f.write("\n>>> Resultat mrr:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mrrscores), numpy.std(mrrscores)))
        f.write("\n>>> Resultat map:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mapscores), numpy.std(mapscores)))
        f.write("\n>>> Resultat ndcg@1:  %.2f%% (+/- %.2f%%)" % (numpy.mean(ndcgscores), numpy.std(ndcgscores)))
        f.write("\n>>> Resultat p@10:  %.2f%% (+/- %.2f%%)" % (numpy.mean(p_10scores), numpy.std(p_10scores)))
        f.write("\n>>> Resultat p@50:  %.2f%% (+/- %.2f%%)" % (numpy.mean(p_50scores), numpy.std(p_50scores)))
        f.write("\n>>> Resultat p@100:  %.2f%% (+/- %.2f%%)" % (numpy.mean(p_100scores), numpy.std(p_100scores)))
        f.write("\n>>> Resultat p@200:  %.2f%% (+/- %.2f%%)" % (numpy.mean(p_200scores), numpy.std(p_200scores)))
    print("\n>>> Resultat mrr:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mrrscores), numpy.std(mrrscores)))
    print("\n>>> Resultat map:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mapscores), numpy.std(mapscores)))
    print("\n>>> Resultat ndcg@1:  %.2f%% (+/- %.2f%%)" % (numpy.mean(ndcgscores), numpy.std(ndcgscores)))
    print("\n>>> Resultat p@10:  %.2f%% (+/- %.2f%%)" % (numpy.mean(p_10scores), numpy.std(p_10scores)))
        