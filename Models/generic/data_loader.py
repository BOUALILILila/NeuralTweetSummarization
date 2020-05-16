import matchzoo as mz
import typing
import csv
import os
import keras
import pandas as pd

def load_data(stage: str = 'train', task: str = 'ranking',path='/projets/iris/PROJETS/lboualil/CORPUS/tweets/train_data_2016_new.csv'
              ) -> typing.Union[mz.DataPack, tuple]:
    """
    Load data.
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`mz.engine.BaseTask` instance.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    if task == 'ranking':
        task = mz.tasks.Ranking()
    if task == 'classification':
        task = mz.tasks.Classification()
        
    table=pd.read_csv(path, index_col=0)
    # change column names
    df=pd.DataFrame({
        "text_left": table['topic_text'],
        'text_right': table['tweet_text'],
        'id_left': table['topic_id'],
        'id_right':table['tweet_id'],
        'label': table['label']
    })
    df=df.reset_index()
    data_pack = mz.pack(df)

    if isinstance(task, mz.tasks.Ranking):
        data_pack.relation['label'] = \
            data_pack.relation['label'].astype('float32')
        return data_pack
    elif isinstance(task, mz.tasks.Classification):
        data_pack.relation['label'] = data_pack.relation['label'].astype(int)
        return data_pack.one_hot_encode_label(num_classes=2), [False, True]
    else:
        raise ValueError(f"{task} is not a valid task.")