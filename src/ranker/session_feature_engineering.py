import sys
import logging
import argparse
import glob
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    feature_engineering_directory = pathlib.Path(settings.DATA / 'feature_engineering')
    feature_engineering_directory.mkdir(parents=True, exist_ok=True)

    if args.mode == 'validation':

        logging.info('Running session feature engineering in validation mode')

        df = pd.concat((
            pd.read_parquet(settings.DATA / 'splits' / 'train.parquet'),
            pd.read_parquet(settings.DATA / 'splits' / 'val.parquet')
        ), axis=0, ignore_index=True).reset_index(drop=True)
        df.sort_values(by=['session', 'ts'], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['datetime'] = pd.to_datetime(df['ts'] + (2 * 60 * 60), unit='s')
        df['hour'] = df['datetime'].dt.hour.astype(np.uint8)
        df['day_of_week'] = df['datetime'].dt.dayofweek.astype(np.uint8)
        df['is_weekend'] = (df['day_of_week'] > 4).astype(np.uint8)
        df['day_of_year'] = df['datetime'].dt.dayofyear.astype(np.uint16)
        logging.info('Created datetime features')

        df_session_features = df.groupby('session').agg({
            'session': 'count',
            'aid': ['nunique', 'last'],
            'type': ['mean', 'last'],
            'ts': ['max', 'min'],
            'hour': ['mean', 'std', 'last'],
            'day_of_week': ['mean', 'std', 'last'],
            'is_weekend': ['mean', 'last'],
            'day_of_year': 'nunique',
        }).reset_index()
        logging.info('Created session aggregation features')

        df_session_features.columns = 'session_' + df_session_features.columns.map('_'.join).str.strip('_')
        df_session_features = df_session_features.rename(columns={'session_session': 'session', 'session_session_count': 'session_count'})

        df_session_features['session_ts_difference'] = (df_session_features['session_ts_max'] - df_session_features['session_ts_min']).astype(np.uint32)
        df_session_features['session_ts_ratio'] = (df_session_features['session_ts_max'] / df_session_features['session_ts_min']).astype(np.float32)
        df_session_features.drop(columns=['session_ts_min', 'session_ts_max'], inplace=True)
        logging.info('Created additional features')

        df_session_features['session'] = df_session_features['session'].astype(np.int64)
        df_session_features['session_count'] = df_session_features['session_count'].astype(np.uint32)
        df_session_features['session_aid_nunique'] = df_session_features['session_aid_nunique'].astype(np.uint8)
        df_session_features['session_aid_last'] = df_session_features['session_aid_last'].astype(np.uint32)
        df_session_features['session_type_mean'] = df_session_features['session_type_mean'].astype(np.float32)
        df_session_features['session_type_last'] = df_session_features['session_type_last'].astype(np.uint8)
        df_session_features['session_hour_mean'] = df_session_features['session_hour_mean'].astype(np.float32)
        df_session_features['session_hour_std'] = df_session_features['session_hour_std'].astype(np.float32)
        df_session_features['session_hour_last'] = df_session_features['session_hour_last'].astype(np.uint8)
        df_session_features['session_day_of_week_mean'] = df_session_features['session_day_of_week_mean'].astype(np.float32)
        df_session_features['session_day_of_week_std'] = df_session_features['session_day_of_week_std'].astype(np.float32)
        df_session_features['session_day_of_week_last'] = df_session_features['session_day_of_week_last'].astype(np.uint8)
        df_session_features['session_is_weekend_mean'] = df_session_features['session_is_weekend_mean'].astype(np.float32)
        df_session_features['session_is_weekend_last'] = df_session_features['session_is_weekend_last'].astype(np.uint8)
        df_session_features['session_day_of_year_nunique'] = df_session_features['session_day_of_year_nunique'].astype(np.uint8)
        logging.info('Down-casted features')

        for event_type_value, event_type in enumerate(['click', 'cart', 'order']):

            df_session_type_features = df.loc[df['type'] == event_type_value].groupby('session').agg({
                'session': 'count',
                'aid': ['nunique', 'last'],
                'ts': ['max', 'min'],
                'hour': ['mean', 'std', 'last'],
                'day_of_week': ['mean', 'std', 'last'],
                'is_weekend': ['mean', 'last'],
                'day_of_year': 'nunique'
            })
            logging.info(f'Created aid {event_type} aggregation features')

            df_session_type_features.columns = f'session_{event_type}_' + df_session_type_features.columns.map('_'.join).str.strip('_')
            df_session_type_features = df_session_type_features.rename(columns={f'session_{event_type}_session_count': f'session_{event_type}_count'})
            df_session_type_features[f'session_{event_type}_ts_difference'] = (df_session_type_features[f'session_{event_type}_ts_max'] - df_session_type_features[f'session_{event_type}_ts_min']).astype(np.uint32)
            df_session_type_features[f'session_{event_type}_ts_ratio'] = (df_session_type_features[f'session_{event_type}_ts_max'] / df_session_type_features[f'session_{event_type}_ts_min']).astype(np.float32)
            logging.info(f'Created session {event_type} additional features')
            df_session_type_features.drop(columns=[f'session_{event_type}_ts_min', f'session_{event_type}_ts_max'], inplace=True)

            df_session_type_features[f'session_{event_type}_count'] = df_session_type_features[f'session_{event_type}_count'].fillna(0).astype(np.uint32)
            logging.info(f'Down-casted {event_type} features')

            for column in df_session_type_features.columns:
                df_session_features[column] = df_session_features['session'].map(df_session_type_features[column]).astype(np.float32)

        df_session_features['session_click_ratio'] = (df_session_features['session_click_count'] / df_session_features['session_count']).astype(np.float32)
        df_session_features['session_cart_ratio'] = (df_session_features['session_cart_count'] / df_session_features['session_count']).astype(np.float32)
        df_session_features['session_order_ratio'] = (df_session_features['session_order_count'] / df_session_features['session_count']).astype(np.float32)
        logging.info(f'Created additional event features')

        df_session_features.to_pickle(feature_engineering_directory / 'train_session_features.pkl')
        logging.info(f'Saved train_session_features.pkl to {feature_engineering_directory}')

    elif args.mode == 'submission':

        logging.info('Running session feature engineering on training and test set')

        df = pd.concat((
            pd.read_pickle(settings.DATA / 'train.pkl'),
            pd.read_pickle(settings.DATA / 'test.pkl')
        ))
        df.sort_values(by=['session', 'ts'], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['datetime'] = pd.to_datetime((df['ts'] / 1000) + (2 * 60 * 60), unit='s')
        df['ts_diff'] = df.groupby('session')['ts'].diff().fillna(0).astype(np.uint8)
        df['hour'] = df['datetime'].dt.hour.astype(np.uint8)
        df['day_of_week'] = df['datetime'].dt.dayofweek.astype(np.uint8)
        df['is_weekend'] = (df['day_of_week'] > 4).astype(np.uint8)
        logging.info('Created datetime features')

        df_session_features = df.groupby('session').agg({
            'session': 'count',
            'aid': 'nunique',
            'type': 'mean',
            'ts_diff': 'mean',
            'hour': ['mean', 'std'],
            'day_of_week': ['mean', 'std'],
            'is_weekend': 'mean',
        }).reset_index()
        df_session_features.columns = 'session_' + df_session_features.columns.map('_'.join).str.strip('_')
        df_session_features = df_session_features.rename(columns={'session_session': 'session', 'session_session_count': 'session_count'})
        df_session_features['session_count'] = df_session_features['session_count'].astype(np.uint32)
        df_session_features['session_aid_nunique'] = df_session_features['session_aid_nunique'].astype(np.uint32)
        df_session_features['session_type_mean'] = df_session_features['session_type_mean'].astype(np.float32)
        df_session_features['session_ts_diff_mean'] = df_session_features['session_ts_diff_mean'].astype(np.float32)
        df_session_features['session_hour_mean'] = df_session_features['session_hour_mean'].astype(np.float32)
        df_session_features['session_hour_std'] = df_session_features['session_hour_std'].astype(np.float32)
        df_session_features['session_day_of_week_mean'] = df_session_features['session_day_of_week_mean'].astype(np.float32)
        df_session_features['session_day_of_week_std'] = df_session_features['session_day_of_week_std'].astype(np.float32)
        df_session_features['session_is_weekend_mean'] = df_session_features['session_is_weekend_mean'].astype(np.float32)
        logging.info('Created session aggregation features')

        for event_type_value, event_type in enumerate(['click', 'cart', 'order']):

            df_session_type_features = df.loc[df['type'] == event_type_value].groupby('session').agg({
                'session': 'count',
                'aid': 'nunique',
                'ts_diff': 'mean',
                'hour': ['mean', 'std'],
                'day_of_week': ['mean', 'std'],
                'is_weekend': 'mean',
            })
            df_session_type_features.columns = f'session_{event_type}_' + df_session_type_features.columns.map('_'.join).str.strip('_')
            df_session_type_features = df_session_type_features.rename(columns={f'session_{event_type}_session_count': f'session_{event_type}_count'})
            for column in df_session_type_features.columns:
                df_session_features[column] = df_session_features['session'].map(df_session_type_features[column]).astype(np.float32)

            logging.info(f'Created session {event_type} aggregation features')

        del df
        df_session_features.to_pickle(feature_engineering_directory / 'train_and_test_session_features.pkl')
        logging.info(f'Saved train_and_test_session_features.pkl to {feature_engineering_directory}')

    else:
        raise ValueError('Invalid mode')
