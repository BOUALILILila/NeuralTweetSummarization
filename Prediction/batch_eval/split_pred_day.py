import glob
import errno
import json
import os
import pandas as pd
import sys

if __name__ == "__main__":
    day_ids_path=sys.argv[1]
    with open(day_ids_path, 'r') as f:
            day_ids=json.load(f)
    pred=pd.read_csv('/projets/iris/PROJETS/lboualil/CORPUS/predictions_2016/MatchPy/MatchPy_cross_fasttext_cbow_2016/predictions_2016_norm.csv', index_col=0)
    pred=pred.reset_index(drop=True)
    pred_day=pred[pred['id_right'].isin(day_ids)]
    path_parts=day_ids_path.split('/')
    filename=path_parts[8]
    filename_parts=filename.split('_')
    ds=filename_parts[1].split('.')
    d=ds[0]
    path=f'/projets/iris/PROJETS/lboualil/CORPUS/predictions_2016/MatchPy/MatchPy_cross_fasttext_cbow_2016/pred_{d}.csv'
    pred_day=pred_day.reset_index(drop=True)
    pred_day.to_csv(path)
        
'''
for i in range(1,4):
    path = f"/home/lila/CORPUS/hour_processed_tweets/statuses_2017-08-{i:=0{2}}"

    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)
'''