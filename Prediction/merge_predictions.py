import pandas as pd
import glob
import os
import json

if __name__ == "__main__":
    trec=open("/projets/iris/PROJETS/lboualil/CORPUS/predictions_2017/MatchPy/MatchPy_fasttext_full_2016/predictions_TREC_format.txt",'w')

    with open("/projets/iris/PROJETS/lboualil/CORPUS/left/left_2017_nist_evaluated_qrels_real.json") as f:
        topics=json.load(f)
    topic_ids=[topic['topid'] for topic in topics]
    topic_ids=set(topic_ids)
    path=os.path.join('/projets/iris/PROJETS/lboualil/CORPUS/predictions_2017/MatchPy/MatchPy_fasttext_full_2016','predictions_2017-*.csv')
    files = glob.glob(path)
    predictions=pd.DataFrame()
    for f in files:
        day_pred = pd.read_csv(f)
        predictions=pd.concat([predictions,day_pred])
    predictions.reset_index(inplace=True,drop=True)
    table=pd.DataFrame()
    for topic_id in topic_ids:
        topic_rank=predictions.loc[predictions['id_left']==topic_id]
        topic_rank=topic_rank.sort_values(['score'],ascending=[False])
        topic_rank.reset_index(inplace=True,drop=True)
        for rank,row in topic_rank.head(100).iterrows():
            trec.write(f"{row['id_left']} Q0 {row['id_right']} {rank+1} {row['score']} matchPyFasttext_2017\n")
        table=pd.concat([table,topic_rank.head(100)])
    del predictions
    table.sort_values(['id_left','score'],ascending=[True,False])
    table.reset_index(inplace=True,drop=True)
    table.to_csv('/projets/iris/PROJETS/lboualil/CORPUS/predictions_2017/MatchPy/MatchPy_fasttext_full_2016/predictions.csv')
    
    #TREC format
    trec.close()

