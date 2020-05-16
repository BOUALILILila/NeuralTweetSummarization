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

if __name__ == "__main__":
    dataset =loader.load_data(stage="train",path="/projets/iris/PROJETS/lboualil/CORPUS/tweets/train_data_2017_nist.csv")
    #dataset =loader.load_data(stage="train",path='/projets/iris/PROJETS/lboualil/CORPUS/tweets/train_data_2015-2016.csv')


    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # define 5-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    mrrscores = []
    mapscores=[]
    ndcgscores=[]
    p_10scores=[]
    p_100scores=[]
    ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
    ranking_task.metrics = [
        mz.engine.parse_metric('mrr'),
        mz.metrics.MeanAveragePrecision(),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=1),
        mz.metrics.Precision(k=10),
        mz.metrics.Precision(k=100)
    ]
    X= dataset.frame()
    Y=dataset.frame()['label']
    w2v_path=os.path.join("/projets/iris/PROJETS/lboualil/CORPUS/embeddings","full","fasttext/CBOW/vectors.txt")
    embedding = mz.embedding.load_from_file(w2v_path)
    trec_rows=[]
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
        model.params['kernel_size'] = [[3,3],[3,3]]
        model.params['dpool_size'] = [3,10]
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
        
        predictions=model.predict(pred_x)
        i=0
        x=pd.DataFrame(pred_x,columns=['id_left','id_right'])
        for index,row in x.iterrows():
            trec_rows+=[{"id_left":row['id_left'], "id_right": row['id_right'], "score":predictions[i][0]}]
            i+=1
        
        mrrscores.append(scores[mz.engine.parse_metric('mrr')] * 100)
        mapscores.append(scores[mz.metrics.MeanAveragePrecision()] * 100)
        ndcgscores.append(scores[mz.metrics.NormalizedDiscountedCumulativeGain(k=1)] * 100)
        p_10scores.append(scores[mz.metrics.Precision(k=10)]*100)
        p_100scores.append(scores[mz.metrics.Precision(k=100)]*100)
    
    pred_table=pd.DataFrame.from_dict(trec_rows)
    pred_table=pred_table.sort_values(['id_left','score'],ascending=[True,False])
    #pred_table.to_csv('/projets/iris/PROJETS/lboualil/CORPUS/predictions_2017/MatchPy/MatchPy_cross_fasttext_cbow_2017/predictions_2017.csv')
        
    #model.save('/projets/iris/PROJETS/lboualil/Models/MatchPy/MatchPy_cbow')
    #preprocessor.save('/projets/iris/PROJETS/lboualil/Preprocessors/MatchPy/preprocessor_matchPy_cbow')
    print("\n>>> Resultat mrr:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mrrscores), numpy.std(mrrscores)))
    print("\n>>> Resultat map:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mapscores), numpy.std(mapscores)))
    print("\n>>> Resultat ndcg@1:  %.2f%% (+/- %.2f%%)" % (numpy.mean(ndcgscores), numpy.std(ndcgscores)))
    print("\n>>> Resultat p@10:  %.2f%% (+/- %.2f%%)" % (numpy.mean(p_10scores), numpy.std(p_10scores)))
    print("\n>>> Resultat p@100:  %.2f%% (+/- %.2f%%)" % (numpy.mean(p_100scores), numpy.std(p_100scores)))
    with open(f'/users/iris/lboualil/cross_valid_fst_cbow_2017_p@100.txt','w') as f:
        f.write("\n>>> Resultat mrr:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mrrscores), numpy.std(mrrscores)))
        f.write("\n>>> Resultat map:  %.2f%% (+/- %.2f%%)" % (numpy.mean(mapscores), numpy.std(mapscores)))
        f.write("\n>>> Resultat ndcg@1:  %.2f%% (+/- %.2f%%)" % (numpy.mean(ndcgscores), numpy.std(ndcgscores)))
        f.write("\n>>> Resultat p@10:  %.2f%% (+/- %.2f%%)" % (numpy.mean(p_10scores), numpy.std(p_10scores)))
        f.write("\n>>> Resultat p@100:  %.2f%% (+/- %.2f%%)" % (numpy.mean(p_100scores), numpy.std(p_100scores)))