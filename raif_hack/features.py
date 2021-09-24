import pandas as pd
from raif_hack.utils import UNKNOWN_VALUE

def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные категориальные переменные
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    fillna_cols = ['region','city','street','realty_type']
    df_new[fillna_cols] = df_new[fillna_cols].fillna(UNKNOWN_VALUE)
    return df_new

def get_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует time-признаки
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = train.copy()
    
    df_new['date'] = pd.to_datetime(df_new['date'], format='%Y-%m-%d')

    df_new['year'] = df_new['date'].dt.year 
    df_new['month'] = df_new['date'].dt.month 
    df_new['day'] = df_new['date'].dt.day

    df_new['dayofweek_num'] = df_new['date'].dt.dayofweek  
    df_new['quarter'] = df_new['date'].dt.quarter

    df_new['dayofyear'] = df_new['date'].dt.dayofyear  
    df_new['weekofyear'] = df_new['date'].dt.weekofyear
    
    return df_new