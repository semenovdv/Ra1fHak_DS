import argparse
import logging.config
import pandas as pd
from traceback import format_exc

from raif_hack.model import BenchmarkModel
from raif_hack.settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET
from raif_hack.settings import MY_FEATURES
from raif_hack.utils import PriceTypeEnum
from raif_hack.metrics import metrics_stat
from raif_hack.features import prepare_categorical, get_time_features, change_target_inflation, get_territory_features
from raif_hack.features import get_random_feature, preproc_floors, fill_na, number_encode_features

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для обучения модели
     
     Примеры:
        1) с poetry - poetry run python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        2) без poetry - python3 train.py --train_data /path/to/train/data --model_path /path/to/model
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--train_data", "-tr", type=str, dest="tr", required=True, help="Путь до обучающего датасета")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True, help="Куда сохранить обученную ML модель")
    
    parser.add_argument("--test_data", "-te", type=str, dest="te", required=True, help="Путь до отложенной выборки")
    parser.add_argument("--output", "-o", type=str, dest="o", required=True, help="Путь до выходного файла")

    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START train.py')
        args = vars(parse_args())
        logger.info('Load train df')
        train_df = pd.read_csv(args['tr'])
        logger.info(f'Input shape: {train_df.shape}')
        
        logger.info('START predict.py')
        args = vars(parse_args())
        logger.info('Load test df')
        test_df = pd.read_csv(args['te'])
        logger.info(f'Input shape: {test_df.shape}')
        
        train_df['is_train'] = 1
        test_df['is_train'] = 0

        full_df = pd.concat([train_df, test_df])
        
        
        full_df = get_time_features(full_df)
        full_df = prepare_categorical(full_df)
        
        full_df = get_territory_features(full_df)
        
        # попробовать на честность
        full_df = change_target_inflation(full_df)
        
        full_df = preproc_floors(full_df)
        full_df = fill_na(full_df)
        
        full_df = get_random_feature(full_df)
        
        full_df, _ = number_encode_features(full_df)
        
        train_df = full_df[full_df['is_train'] == 1]
        test_df = full_df[full_df['is_train'] == 0]
        

        X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES + MY_FEATURES]
        y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        X_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES + MY_FEATURES]
        y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        logger.info(f'X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}')
        model = BenchmarkModel(numerical_features=NUM_FEATURES+MY_FEATURES, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                  ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=MODEL_PARAMS)
        logger.info('Fit model')
        model.fit(X_offer, y_offer, X_manual, y_manual)
        logger.info('Save model')
        model.save(args['mp'])

        predictions_offer = model.predict(X_offer)
        metrics = metrics_stat(y_offer.values, predictions_offer/(1+model.corr_coef)) # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
        logger.info(f'Metrics stat for training data with offers prices: {metrics}')

        predictions_manual = model.predict(X_manual)
        metrics = metrics_stat(y_manual.values, predictions_manual)
        logger.info(f'Metrics stat for training data with manual prices: {metrics}')
        
        
        logger.info('Predict')
        test_df['per_square_meter_price'] = model.predict(test_df[NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES + MY_FEATURES])
        logger.info('Save results')
        test_df[['id','per_square_meter_price']].to_csv(args['o'], index=False)


    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')
    