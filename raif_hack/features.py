import pandas as pd
from raif_hack.utils import UNKNOWN_VALUE
from raif_hack.settings import TRAIN_INFLATION, TRAIN_INFLATION_M, TARGET

import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import re



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
    df_new = df.copy()
    
    df_new['date'] = pd.to_datetime(df_new['date'], format='%Y-%m-%d')

    df_new['year'] = df_new['date'].dt.year 
    df_new['month'] = df_new['date'].dt.month 
    df_new['day'] = df_new['date'].dt.day

    df_new['dayofweek_num'] = df_new['date'].dt.dayofweek  
    df_new['quarter'] = df_new['date'].dt.quarter

    df_new['dayofyear'] = df_new['date'].dt.dayofyear  
    df_new['weekofyear'] = df_new['date'].dt.weekofyear
    
    return df_new

def change_target_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Изменяет таргет в зависимости от инфляции 
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    
    for mnth in TRAIN_INFLATION_M:
        df_new[TARGET] = df_new.apply(lambda x:  x[TARGET] + x[TARGET]*TRAIN_INFLATION[mnth-2]/100 if x['month'] >= mnth else x[TARGET],axis=1)
    
    return df_new

def get_territory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует территориалььные признаки
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    
    other_001 =  ['osm_amenity_points_in_0.001', 'osm_catering_points_in_0.001',  'osm_shops_points_in_0.001',                                                                                                       'osm_culture_points_in_0.001']
    other_005 =  ['osm_amenity_points_in_0.005', 'osm_catering_points_in_0.005',  'osm_shops_points_in_0.005',  'osm_healthcare_points_in_0.005',  'osm_leisure_points_in_0.005',   'osm_historic_points_in_0.005',  'osm_culture_points_in_0.005']
    other_0075 = ['osm_amenity_points_in_0.0075','osm_catering_points_in_0.0075', 'osm_shops_points_in_0.0075', 'osm_healthcare_points_in_0.0075', 'osm_leisure_points_in_0.0075',  'osm_historic_points_in_0.0075', 'osm_culture_points_in_0.0075']
    other_01 =   ['osm_amenity_points_in_0.01',  'osm_catering_points_in_0.01',   'osm_shops_points_in_0.01',   'osm_healthcare_points_in_0.01',   'osm_leisure_points_in_0.01',    'osm_historic_points_in_0.01',   'osm_culture_points_in_0.01']
     
    df_new['sum_other_001'] = df_new[other_001].sum(axis=1)
    df_new['sum_other_005'] = df_new[other_005].sum(axis=1)
    df_new['sum_other_0075'] = df_new[other_0075].sum(axis=1)
    df_new['sum_other_01'] = df_new[other_01].sum(axis=1)
    
    all_mean_001 = df_new['sum_other_001'].mean()
    all_mean_005 = df_new['sum_other_005'].mean()
    all_mean_0075 = df_new['sum_other_0075'].mean()
    all_mean_01 = df_new['sum_other_01'].mean()
    
    df_new['sum_other_001_diff'] = df_new['sum_other_001'] - all_mean_001
    df_new['sum_other_005_diff'] = df_new['sum_other_005'] - all_mean_005
    df_new['sum_other_0075_diff'] = df_new['sum_other_0075'] - all_mean_0075
    df_new['sum_other_01_diff'] = df_new['sum_other_01'] - all_mean_01
    
    df_new['sum_other_001_share'] = df_new['sum_other_001'] / all_mean_001
    df_new['sum_other_005_share'] = df_new['sum_other_005'] / all_mean_005
    df_new['sum_other_0075_share'] = df_new['sum_other_0075'] / all_mean_0075
    df_new['sum_other_01_share'] = df_new['sum_other_01'] / all_mean_01
    
    # regional
    
    all_mean_001 = df_new.groupby('region')['sum_other_001'].mean().apply(lambda x: max(1, x)).reset_index().rename(columns={'sum_other_001':'sum_other_001_region'})
    all_mean_005 = df_new.groupby('region')['sum_other_005'].mean().apply(lambda x: max(1, x)).reset_index().rename(columns={'sum_other_005':'sum_other_005_region'})
    all_mean_0075 = df_new.groupby('region')['sum_other_0075'].mean().apply(lambda x: max(1, x)).reset_index().rename(columns={'sum_other_0075':'sum_other_0075_region'})
    all_mean_01 = df_new.groupby('region')['sum_other_01'].mean().apply(lambda x: max(1, x)).reset_index().rename(columns={'sum_other_01':'sum_other_01_region'})
    
    df_new = df_new.merge(all_mean_001, how='inner', on='region')
    df_new = df_new.merge(all_mean_005, how='inner', on='region')
    df_new = df_new.merge(all_mean_0075, how='inner', on='region')
    df_new = df_new.merge(all_mean_01, how='inner', on='region')
    
    
    df_new['sum_other_001_diff_regional'] = df_new['sum_other_001'] - df_new['sum_other_001_region']
    df_new['sum_other_005_diff_regional'] = df_new['sum_other_005'] - df_new['sum_other_005_region']
    df_new['sum_other_0075_diff_regional'] = df_new['sum_other_0075'] - df_new['sum_other_0075_region']
    df_new['sum_other_01_diff_regional'] = df_new['sum_other_01'] - df_new['sum_other_01_region']
    
    df_new['sum_other_001_share_regional'] = df_new['sum_other_001'] / df_new['sum_other_001_region']
    df_new['sum_other_005_share_regional'] = df_new['sum_other_005'] / df_new['sum_other_005_region']
    df_new['sum_other_0075_share_regional'] = df_new['sum_other_0075'] / df_new['sum_other_0075_region']
    df_new['sum_other_01_share_regional'] = df_new['sum_other_01'] / df_new['sum_other_01_region']
    
    ###
    
    
    df_new['sum_other_001_diff'] = df_new['sum_other_001'] - df_new['sum_other_001'].mean()
    df_new['sum_other_005_diff'] = df_new['sum_other_005'] - df_new['sum_other_005'].mean()
    df_new['sum_other_0075_diff'] = df_new['sum_other_0075'] - df_new['sum_other_0075'].mean()
    df_new['sum_other_01_diff'] = df_new['sum_other_01'] - df_new['sum_other_01'].mean()
    
    df_new['sum_other_001_share'] = df_new['sum_other_001'] / df_new['sum_other_001'].mean()
    df_new['sum_other_005_share'] = df_new['sum_other_005'] / df_new['sum_other_005'].mean()
    df_new['sum_other_0075_share'] = df_new['sum_other_0075'] / df_new['sum_other_0075'].mean()
    df_new['sum_other_01_share'] = df_new['sum_other_01'] / df_new['sum_other_01'].mean()
     
    
    build_001 =  ['osm_building_points_in_0.001',  'osm_finance_points_in_0.001',                                 'osm_offices_points_in_0.001']
    build_005 =  ['osm_building_points_in_0.005',  'osm_finance_points_in_0.005',  'osm_hotels_points_in_0.005',  'osm_offices_points_in_0.005']
    build_0075 = ['osm_building_points_in_0.0075', 'osm_finance_points_in_0.0075', 'osm_hotels_points_in_0.0075', 'osm_offices_points_in_0.0075']
    build_01 =   ['osm_building_points_in_0.01',   'osm_finance_points_in_0.01',   'osm_hotels_points_in_0.01',   'osm_offices_points_in_0.01']
     
    df_new['sum_build_001'] = df_new[build_001].sum(axis=1)
    df_new['sum_build_005'] = df_new[build_005].sum(axis=1)
    df_new['sum_build_0075'] = df_new[build_0075].sum(axis=1)
    df_new['sum_build_01'] = df_new[build_01].sum(axis=1)
    
    df_new['sum_build_001_diff'] = df_new['sum_build_001'] - df_new['sum_build_001'].mean()
    df_new['sum_build_005_diff'] = df_new['sum_build_005'] - df_new['sum_build_005'].mean()
    df_new['sum_build_0075_diff'] = df_new['sum_build_0075'] - df_new['sum_build_0075'].mean()
    df_new['sum_build_01_diff'] = df_new['sum_build_01'] - df_new['sum_build_01'].mean()
    
    df_new['sum_build_001_share'] = df_new['sum_build_001'] / df_new['sum_build_001'].mean()
    df_new['sum_build_005_share'] = df_new['sum_build_005'] / df_new['sum_build_005'].mean()
    df_new['sum_build_0075_share'] = df_new['sum_build_0075'] / df_new['sum_build_0075'].mean()
    df_new['sum_build_01_share'] = df_new['sum_build_01'] / df_new['sum_build_01'].mean()
    
    # regional
    
    all_mean_001 = df_new.groupby('region')['sum_build_001'].mean().apply(lambda x: max(1, x)).reset_index().rename(columns={'sum_build_001':'sum_build_001_region'})
    all_mean_005 = df_new.groupby('region')['sum_build_005'].mean().apply(lambda x: max(1, x)).reset_index().rename(columns={'sum_build_005':'sum_build_005_region'})
    all_mean_0075 = df_new.groupby('region')['sum_build_0075'].mean().apply(lambda x: max(1, x)).reset_index().rename(columns={'sum_build_0075':'sum_build_0075_region'})
    all_mean_01 = df_new.groupby('region')['sum_build_01'].mean().apply(lambda x: max(1, x)).reset_index().rename(columns={'sum_build_01':'sum_build_01_region'})
    
    df_new = df_new.merge(all_mean_001, how='inner', on='region')
    df_new = df_new.merge(all_mean_005, how='inner', on='region')
    df_new = df_new.merge(all_mean_0075, how='inner', on='region')
    df_new = df_new.merge(all_mean_01, how='inner', on='region')
    
    
    df_new['sum_build_001_diff_regional'] = df_new['sum_build_001'] - df_new['sum_build_001_region']
    df_new['sum_build_005_diff_regional'] = df_new['sum_build_005'] - df_new['sum_build_005_region']
    df_new['sum_build_0075_diff_regional'] = df_new['sum_build_0075'] - df_new['sum_build_0075_region']
    df_new['sum_build_01_diff_regional'] = df_new['sum_build_01'] - df_new['sum_build_01_region']
    
    df_new['sum_build_001_share_regional'] = df_new['sum_build_001'] / df_new['sum_build_001_region']
    df_new['sum_build_005_share_regional'] = df_new['sum_build_005'] / df_new['sum_build_005_region']
    df_new['sum_build_0075_share_regional'] = df_new['sum_build_0075'] / df_new['sum_build_0075_region']
    df_new['sum_build_01_share_regional'] = df_new['sum_build_01'] / df_new['sum_build_01_region']
    
  
    
    return df_new


def get_random_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует территориалььные признаки
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    
    df_new['randNumCol'] = np.random.randint(1, 6, df_new.shape[0])
    
    return df_new


def preproc_floors(old_df):
    '''
    Функция энкодинга признака floor (этажи)
    В результате создаёт перезаписывает признак floor (-1 если много этажей или этаж не цифрой),
        создаёт признаки basement (наличие подвала или цоколя), mezzanine (наличие антресоли и мансарды)
        tech (наличие тех этажа)
    '''
    df = old_df.copy()
    
    floor = df['floor'].str.lower().to_numpy()
    num_floors = []
    res_floors = []
    basement = [] #наличие подвала или цоколя
    mezzanine = [] #наличие антресоли и мансарды
    tech = [] # тех этаж
    for item in floor:
        count_floors = 0
        
        if item != item:
            basement.append(0)
            mezzanine.append(0)
            tech.append(0)
            num_floors.append(1)
            res_floors.append(np.nan)
            continue
            
        if 'подв' or 'цок' in item:
            basement.append(1)
            count_floors += 1 # или 2
        else:
            basement.append(0)
            
        if 'манса' or 'антре' or 'мезо' in item:
            mezzanine.append(1)
            count_floors += 1 # или 2
            
        else:
            mezzanine.append(0)
            
        if 'тех' in item:
            tech.append(1)
            count_floors += 1 
        else:
            tech.append(0)
            
        item = re.sub('[^\d\. - :]',' ', item)
        item = item.replace('-', '.').replace(':', '.')
        new_item = item.split()
        if len(new_item) == 0:
            num_floors.append(count_floors)
            res_floors.append(-1)
            continue
        elif len(new_item) == 1:
            if '.' in new_item[0]:
                last_item = new_item[0].split('.')
                if last_item[-1] == '0':
                    count_floors += 1 
                    num_floors.append(count_floors)
                    res_floors.append(int(''.join(last_item[:-1])))
                    continue
                else:
                    count_floors += len(range(int(last_item[0]), int(last_item[-1]) + 1))
                    num_floors.append(count_floors)
                    res_floors.append(-1)
                    continue
            else:
                count_floors += 1 
                num_floors.append(count_floors)
                res_floors.append(int(float(new_item[0])))
                continue
        else:
            count_floors += len(new_item)
            num_floors.append(count_floors)
            res_floors.append(-1)
            
    df = df.drop(['floor'], axis = 1)
    df['floor'] = res_floors
    df['num_floors'] = num_floors
    df['basement'] = basement
    df['mezzanine'] = mezzanine
    df['tech'] = tech
            
    return df


def fill_na(old_df):
    '''
    Функция заполянет null значения в датафрейме
    '''
    df = old_df.copy()
    
    #floor
    # заполнение null = -1, дабавляется в num_floors площадь / медина площади на этаж
    # по факту заполянет всё 1 :(((
    square = df[df['floor'].isna() == False]['total_square'].to_numpy() 
    n_floor = df[df['floor'].isna() == False]['num_floors'].to_numpy() 
    square_per_floor = square / n_floor
    mean_square_per_floor = np.mean(square_per_floor)
    df[df['floor'].isna()]['num_floor'] = np.around(df[df['floor'].isna()]['total_square'].to_numpy() / mean_square_per_floor)
    df['floor'] = df['floor'].fillna(-1)
    
    # reform_house_population_1000 reform_house_population_500 
    # reform_mean_floor_count_1000 reform_mean_floor_count_500
    # reform_mean_year_building_1000 reform_mean_year_building_500
    # заполненяет null значения на среднее по региону
    group1 =  df[~df['reform_house_population_1000'].isna()].groupby(['region'])['reform_house_population_1000']
    group2 =  df[~df['reform_house_population_500'].isna()].groupby(['region'])['reform_house_population_500']
    group3 =  df[~df['reform_mean_floor_count_1000'].isna()].groupby(['region'])['reform_mean_floor_count_1000']
    group4 =  df[~df['reform_mean_floor_count_500'].isna()].groupby(['region'])['reform_mean_floor_count_500']
    group5 =  df[~df['reform_mean_year_building_1000'].isna()].groupby(['region'])['reform_mean_year_building_1000']
    group6 =  df[~df['reform_mean_year_building_500'].isna()].groupby(['region'])['reform_mean_year_building_500']
    
    df['reform_house_population_1000'] = df.apply(lambda x: np.mean(group1.groups[x['region']]) \
                                                  if x['reform_house_population_1000'] != x['reform_house_population_1000'] else \
                                                  x['reform_house_population_1000'], axis = 1)
    df['reform_house_population_500'] = df.apply(lambda x: np.mean(group1.groups[x['region']]) \
                                                  if x['reform_house_population_500'] != x['reform_house_population_500'] else \
                                                  x['reform_house_population_500'], axis = 1)
    df['reform_mean_floor_count_1000'] = df.apply(lambda x: np.mean(group1.groups[x['region']]) \
                                                  if x['reform_mean_floor_count_1000'] != x['reform_mean_floor_count_1000'] else \
                                                  x['reform_mean_floor_count_1000'], axis = 1)
    df['reform_mean_floor_count_500'] = df.apply(lambda x:  np.mean(group1.groups[x['region']]) \
                                                  if x['reform_mean_floor_count_500'] != x['reform_mean_floor_count_500'] else \
                                                  x['reform_mean_floor_count_500'], axis = 1)
    df['reform_mean_year_building_1000'] = df.apply(lambda x: np.mean(group1.groups[x['region']]) \
                                                  if x['reform_mean_year_building_1000'] != x['reform_mean_year_building_1000'] else \
                                                  x['reform_mean_year_building_1000'], axis = 1)
    df['reform_mean_year_building_500'] = df.apply(lambda x: np.mean(group1.groups[x['region']]) \
                                                  if x['reform_mean_year_building_500'] != x['reform_mean_year_building_500'] else \
                                                  x['reform_mean_year_building_500'], axis = 1)
    
    # дропаем street
    #df = df.dropna(subset=['street'])
    
    return df


def number_encode_features(df):
    result = df.copy() 
    result['street'] += result['city']
    encoders = {}
    for column in result.columns.drop(['id']):
        if result.dtypes[column] == 'object':
            encoders[column] = preprocessing.LabelEncoder() 
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders