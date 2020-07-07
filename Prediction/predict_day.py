import sys
import glob
import os
import json
import relevance_estimator as rel_estimator
import novelty_estimator as nov_estimator

N_TOP=100

if __name__ == "__main__":
    day_dir=sys.argv[1]
    print('dir:',day_dir)
    path=os.path.join(day_dir,'statuses.log.2017-*.json')
    files = glob.glob(path)
    processed_tweets=[]
    for hour_file in files:
        print('file:',hour_file)
        with open(hour_file) as f:
            processed_tweets+=json.load(f)
    '''
    processed_tweets=[{'tweetid':'1','text':'thai medalist gran die celebr win year old grandmoth thai olymp bronz medal sport'},
        {'tweetid':'2','text':'rio olymp thai medalist grandmoth die celebr win via'},
        {'tweetid':'3','text':'rio olymp thailand weightlift sinphet kruaithong grandmoth die celebr win indian express'},
        {'tweetid':'4','text':'rt bambam gonna bring gold medal back thailand dabolymp'},
        {'tweetid':'5','text':'rt phelp alreadi mani medal thailand mani gold ethiopia argentina'},
        {'tweetid':'6','text':'silver medal smo suvarnabhumi intern airport bangkok thailand'},
        {'tweetid':'7','text':'break thailand sopita tanasan gold weightlift olymp'},
        {'tweetid':'8','text':'rt first olymp gold medal thailand'}
]
'''
    if len(processed_tweets)>0:
        predictions=rel_estimator.predict_similarity_scores(processed_tweets,N_TOP)
        del processed_tweets
        w=day_dir.split('-')
        month=w[1]
        day=w[2]
        path=f"/data/CORPUS/predictions_2017/MatchPy/MatchPy_fasttext_cbow_2015_2016_rel/predictions_2017-{month}-{day}.csv"
        predictions.to_csv(path)
        if predictions is not None:
            summaries=nov_estimator.delete_redudants(predictions)
            path=f"/data/CORPUS/predictions_2017/MatchPy/MatchPy_fasttext_cbow_2015_2016_rel/summaries_2017-{month}-{day}.csv"
            summaries.to_csv(path)
            trec=open(f"/data/CORPUS/predictions_2017/MatchPy/MatchPy_fasttext_cbow_2015_2016_rel/run_2017-{month}-{day}.run",'w')
            for index,row in summaries.iterrows():
                trec.write(f"2017{month}{day} {index} Q0 {row['id_right']} {row['rank']} {row['score']} matchPyFasttextCBOW_2017\n")
            trec.close()

        
        