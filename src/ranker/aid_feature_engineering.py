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

        logging.info('Running aid feature engineering on training set')
        truncated_train_parquet_directory = pathlib.Path(settings.DATA / 'parquet_files' / 'train_truncated_parquet')
        truncated_train_parquet_file_paths = glob.glob(str(truncated_train_parquet_directory / '*'))

        df = []
        for file_path in tqdm(truncated_train_parquet_file_paths):
            df.append(pd.read_parquet(file_path))

        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        df.sort_values(by=['session', 'ts'], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['datetime'] = pd.to_datetime((df['ts'] / 1000) + (2 * 60 * 60), unit='s')
        df['ts_diff'] = df.groupby('aid')['ts'].diff().fillna(0).astype(np.uint8)
        df['hour'] = df['datetime'].dt.hour.astype(np.uint8)
        df['day_of_week'] = df['datetime'].dt.dayofweek.astype(np.uint8)
        df['is_weekend'] = (df['day_of_week'] > 4).astype(np.uint8)
        logging.info('Created datetime features')

        df_aid_features = df.groupby('aid').agg({
            'aid': 'count',
            'session': 'nunique',
            'type': 'mean',
            'ts_diff': 'mean',
            'hour': ['mean', 'std'],
            'day_of_week': ['mean', 'std'],
            'is_weekend': ['mean'],
        }).reset_index()
        df_aid_features.columns = 'aid_' + df_aid_features.columns.map('_'.join).str.strip('_')
        df_aid_features = df_aid_features.rename(columns={'aid_aid': 'aid', 'aid_aid_count': 'aid_count'})
        df_aid_features['aid_count'] = df_aid_features['aid_count'].astype(np.uint32)
        df_aid_features['aid_session_nunique'] = df_aid_features['aid_session_nunique'].astype(np.uint32)
        df_aid_features['aid_ts_diff_mean'] = df_aid_features['aid_ts_diff_mean'].astype(np.float32)
        df_aid_features['aid_type_mean'] = df_aid_features['aid_type_mean'].astype(np.float32)
        df_aid_features['aid_hour_mean'] = df_aid_features['aid_hour_mean'].astype(np.float32)
        df_aid_features['aid_hour_std'] = df_aid_features['aid_hour_std'].astype(np.float32)
        df_aid_features['aid_day_of_week_mean'] = df_aid_features['aid_day_of_week_mean'].astype(np.float32)
        df_aid_features['aid_day_of_week_std'] = df_aid_features['aid_day_of_week_std'].astype(np.float32)
        df_aid_features['aid_is_weekend_mean'] = df_aid_features['aid_is_weekend_mean'].astype(np.float32)
        logging.info('Created aid aggregation features')

        for event_type_value, event_type in enumerate(['click', 'cart', 'order']):

            df_aid_type_features = df.loc[df['type'] == event_type_value].groupby('aid').agg({
                'aid': 'count',
                'session': 'nunique',
                'ts_diff': 'mean',
                'hour': ['mean', 'std'],
                'day_of_week': ['mean', 'std'],
                'is_weekend': ['mean'],
            })
            df_aid_type_features.columns = f'aid_{event_type}_' + df_aid_type_features.columns.map('_'.join).str.strip('_')
            df_aid_type_features = df_aid_type_features.rename(columns={f'aid_{event_type}_aid_count': f'aid_{event_type}_count'})
            for column in df_aid_type_features.columns:
                df_aid_features[column] = df_aid_features['aid'].map(df_aid_type_features[column]).astype(np.float32)

            logging.info(f'Created aid {event_type} aggregation features')

        del df
        df_aid_features.to_pickle(feature_engineering_directory / 'train_aid_features.pkl')
        logging.info(f'Saved train_aid_features.pkl to {feature_engineering_directory}')

    elif args.mode == 'submission':

        logging.info('Running aid feature engineering on training and test set')
        df = pd.concat((
            pd.read_pickle(settings.DATA / 'train.pkl'),
            pd.read_pickle(settings.DATA / 'test.pkl')
        ))

        df.sort_values(by=['session', 'ts'], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['datetime'] = pd.to_datetime((df['ts'] / 1000) + (2 * 60 * 60), unit='s')
        df['ts_diff'] = df.groupby('aid')['ts'].diff().fillna(0).astype(np.uint8)
        df['hour'] = df['datetime'].dt.hour.astype(np.uint8)
        df['day_of_week'] = df['datetime'].dt.dayofweek.astype(np.uint8)
        df['is_weekend'] = (df['day_of_week'] > 4).astype(np.uint8)
        logging.info('Created datetime features')

        df_aid_features = df.groupby('aid').agg({
            'aid': 'count',
            'session': 'nunique',
            'type': 'mean',
            'ts_diff': 'mean',
            'hour': ['mean', 'std'],
            'day_of_week': ['mean', 'std'],
            'is_weekend': ['mean'],
        }).reset_index()
        df_aid_features.columns = 'aid_' + df_aid_features.columns.map('_'.join).str.strip('_')
        df_aid_features = df_aid_features.rename(columns={'aid_aid': 'aid', 'aid_aid_count': 'aid_count'})
        df_aid_features['aid_count'] = df_aid_features['aid_count'].astype(np.uint32)
        df_aid_features['aid_session_nunique'] = df_aid_features['aid_session_nunique'].astype(np.uint32)
        df_aid_features['aid_ts_diff_mean'] = df_aid_features['aid_ts_diff_mean'].astype(np.float32)
        df_aid_features['aid_type_mean'] = df_aid_features['aid_type_mean'].astype(np.float32)
        df_aid_features['aid_hour_mean'] = df_aid_features['aid_hour_mean'].astype(np.float32)
        df_aid_features['aid_hour_std'] = df_aid_features['aid_hour_std'].astype(np.float32)
        df_aid_features['aid_day_of_week_mean'] = df_aid_features['aid_day_of_week_mean'].astype(np.float32)
        df_aid_features['aid_day_of_week_std'] = df_aid_features['aid_day_of_week_std'].astype(np.float32)
        df_aid_features['aid_is_weekend_mean'] = df_aid_features['aid_is_weekend_mean'].astype(np.float32)
        logging.info('Created aid aggregation features')

        for event_type_value, event_type in enumerate(['click', 'cart', 'order']):

            df_aid_type_features = df.loc[df['type'] == event_type_value].groupby('aid').agg({
                'aid': 'count',
                'session': 'nunique',
                'ts_diff': 'mean',
                'hour': ['mean', 'std'],
                'day_of_week': ['mean', 'std'],
                'is_weekend': ['mean'],
            })
            df_aid_type_features.columns = f'aid_{event_type}_' + df_aid_type_features.columns.map('_'.join).str.strip('_')
            df_aid_type_features = df_aid_type_features.rename(columns={f'aid_{event_type}_aid_count': f'aid_{event_type}_count'})
            for column in df_aid_type_features.columns:
                df_aid_features[column] = df_aid_features['aid'].map(df_aid_type_features[column]).astype(np.float32)

            logging.info(f'Created aid {event_type} aggregation features')

        del df
        df_aid_features.to_pickle(feature_engineering_directory / 'train_and_test_aid_features.pkl')
        logging.info(f'Saved train_and_test_aid_features.pkl to {feature_engineering_directory}')
