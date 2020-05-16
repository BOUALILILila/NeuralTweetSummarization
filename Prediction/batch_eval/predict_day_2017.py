import sys

import os
import json
import novelty_2017 as nov_estimator
import pandas as pd 


if __name__ == "__main__":
        pred_day=sys.argv[1]
        seuil=float(sys.argv[2])
        predictions=pd.read_csv(pred_day, index_col=0,dtype={'id_right': object})
        predictions=predictions.set_index('id_left')
        print('++++++file: ',pred_day)

        print('head\n',predictions.head())
        if predictions is not None:
            summaries=nov_estimator.delete_redudants(predictions,seuil)
            
            path_parts=pred_day.split('/')
            filename=path_parts[9]
            filename_parts=filename.split('_')
            ds=filename_parts[1].split('.')
            d=ds[0]
            date=d.split('-')
            month=date[1]
            day=date[2]
            path=f"/projets/iris/PROJETS/lboualil/CORPUS/predictions_2017/MatchPy/MatchPy_cross_fasttext_cbow_2017_norm_{seuil}/summaries_2017-{month}-{day}.csv"
            summaries.to_csv(path)
            trec=open(f"/projets/iris/PROJETS/lboualil/CORPUS/predictions_2017/MatchPy/MatchPy_cross_fasttext_cbow_2017_norm_{seuil}/run_{d}.run",'w')
            for index,row in summaries.iterrows():
                trec.write(f"2017{month}{day} {index} Q0 {row['id_right']} {row['rank']} {row['score']} matchPyFasttextCBOW_cross_2017\n")
            trec.close()
            

        
        