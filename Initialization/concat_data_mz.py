import os
import pandas as pd

path = os.path.join("/projets/iris/PROJETS/lboualil/CORPUS/tweets",'train_data_2016_new.csv')
table=pd.read_csv(path, index_col=0)
path2 = os.path.join("/projets/iris/PROJETS/lboualil/CORPUS/tweets",'train_data_2015.csv')
table2=pd.read_csv(path2, index_col=0)
df=pd.concat([table,table2])
df.head()
df=df.reset_index(drop=True)
df.to_csv('/projets/iris/PROJETS/lboualil/CORPUS/tweets/train_data_2015-2016.csv')