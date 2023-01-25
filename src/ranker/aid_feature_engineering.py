import sys
import logging
import argparse
import pathlib
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

        logging.info('Running aid feature engineering in validation mode')
        df = pd.concat((
            pd.read_parquet(settings.DATA / 'splits' / 'train.parquet'),
            pd.read_parquet(settings.DATA / 'splits' / 'val.parquet')
        ), axis=0, ignore_index=True).reset_index(drop=True)

    elif args.mode == 'submission':

        logging.info('Running aid feature engineering in submission mode')
        df = pd.concat((
            pd.read_pickle(settings.DATA / 'train.pkl'),
            pd.read_pickle(settings.DATA / 'test.pkl')
        ), axis=0, ignore_index=True)
        df['ts'] /= 1000

    else:
        raise ValueError('Invalid mode')

    df.sort_values(by=['session', 'ts'], ascending=[True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['datetime'] = pd.to_datetime(df['ts'] + (2 * 60 * 60), unit='s')
    df['hour'] = df['datetime'].dt.hour.astype(np.uint8)
    df['day_of_week'] = df['datetime'].dt.dayofweek.astype(np.uint8)
    df['day_of_year'] = df['datetime'].dt.dayofyear.astype(np.uint16)
    df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(np.uint8)
    df['session_cumcount'] = (df.groupby('session')['aid'].cumcount() + 1).astype(np.uint16)
    df['session_cumcount_normalized'] = df['session_cumcount'] / df.groupby('session')['session'].transform('count')
    df['is_session_start'] = (df['session_cumcount'] == 1).astype(np.uint8)
    df['is_session_end'] = (df['session_cumcount_normalized'] == 1).astype(np.uint8)
    df['type+1'] = (df['type'] + 1).astype(np.uint8)
    df['session_type+1_cumsum'] = df.groupby('session')['type+1'].cumsum()
    logging.info('Created datetime features')

    df_aid_features = df.groupby('aid').agg({
        'aid': 'count',
        'session': 'nunique',
        'type': 'mean',
        'ts': ['max', 'min'],
        'hour': ['mean', 'std'],
        'day_of_week': ['mean', 'std'],
        'day_of_year': 'nunique',
        'session_cumcount_normalized': 'mean',
        'is_session_start': ['mean', 'count'],
        'is_session_end': ['mean', 'count'],
        'session_type+1_cumsum': 'mean'
    }).reset_index()
    logging.info('Created aid aggregation features')

    df_aid_features.columns = 'aid_' + df_aid_features.columns.map('_'.join).str.strip('_')
    df_aid_features = df_aid_features.rename(columns={'aid_aid': 'aid', 'aid_aid_count': 'aid_count'})

    df_aid_features['aid_count_rank_pct'] = df_aid_features['aid_count'].rank(pct=True).astype(np.float32)
    df_aid_features['aid_session_nunique_rank_pct'] = df_aid_features['aid_session_nunique'].rank(pct=True).astype(np.float32)
    df_aid_features['aid_day_of_year_nunique_rank_pct'] = df_aid_features['aid_day_of_year_nunique'].rank(pct=True).astype(np.float32)
    df_aid_features['aid_is_session_start_count_rank_pct'] = df_aid_features['aid_is_session_start_count'].rank(pct=True).astype(np.float32)
    df_aid_features['aid_is_session_end_count_rank_pct'] = df_aid_features['aid_is_session_end_count'].rank(pct=True).astype(np.float32)
    df_aid_features['aid_ts_ratio'] = (df_aid_features['aid_ts_max'] / df_aid_features['aid_ts_min']).astype(np.float32)
    df_aid_features.drop(columns=[
        'aid_session_nunique', 'aid_day_of_year_nunique',
        'aid_is_session_start_count', 'aid_is_session_end_count',
        'aid_ts_min', 'aid_ts_max',
    ], inplace=True)
    logging.info('Created additional features')

    df_aid_features['aid'] = df_aid_features['aid'].astype(np.int32)
    df_aid_features['aid_type_mean'] = df_aid_features['aid_type_mean'].astype(np.float32)
    df_aid_features['aid_hour_mean'] = df_aid_features['aid_hour_mean'].astype(np.float32)
    df_aid_features['aid_hour_std'] = df_aid_features['aid_hour_std'].astype(np.float32)
    df_aid_features['aid_day_of_week_mean'] = df_aid_features['aid_day_of_week_mean'].astype(np.float32)
    df_aid_features['aid_day_of_week_std'] = df_aid_features['aid_day_of_week_std'].astype(np.float32)
    df_aid_features['aid_session_cumcount_normalized_mean'] = df_aid_features['aid_session_cumcount_normalized_mean'].astype(np.float32)
    df_aid_features['aid_is_session_start_mean'] = df_aid_features['aid_is_session_start_mean'].astype(np.float32)
    df_aid_features['aid_is_session_end_mean'] = df_aid_features['aid_is_session_end_mean'].astype(np.float32)
    df_aid_features['aid_session_type+1_cumsum_mean'] = df_aid_features['aid_session_type+1_cumsum_mean'].astype(np.float32)
    logging.info('Down-casted features')

    for event_type_value, event_type in enumerate(['click', 'cart', 'order']):

        df_aid_type_features = df.loc[df['type'] == event_type_value].groupby('aid').agg({
            'aid': 'count',
            'session': 'nunique',
            'ts': ['max', 'min'],
            'hour': ['mean', 'std'],
            'day_of_week': ['mean', 'std'],
            'day_of_year': 'nunique',
            'session_cumcount_normalized': 'mean',
            'is_session_start': ['mean', 'count'],
            'is_session_end': ['mean', 'count'],
        })
        logging.info(f'Created aid {event_type} aggregation features')

        df_aid_type_features.columns = f'aid_{event_type}_' + df_aid_type_features.columns.map('_'.join).str.strip('_')
        df_aid_type_features = df_aid_type_features.rename(columns={f'aid_{event_type}_aid_count': f'aid_{event_type}_count'})
        df_aid_type_features = df_aid_type_features.reset_index()

        df_aid_type_features[f'aid_{event_type}_count_rank_pct'] = df_aid_type_features[f'aid_{event_type}_count'].rank(pct=True).astype(np.float32)
        df_aid_type_features[f'aid_{event_type}_session_nunique_rank_pct'] = df_aid_type_features[f'aid_{event_type}_session_nunique'].rank(pct=True).astype(np.float32)
        df_aid_type_features[f'aid_{event_type}_day_of_year_nunique_rank_pct'] = df_aid_type_features[f'aid_{event_type}_day_of_year_nunique'].rank(pct=True).astype(np.float32)
        df_aid_type_features[f'aid_{event_type}_is_session_start_count_rank_pct'] = df_aid_type_features[f'aid_{event_type}_is_session_start_count'].rank(pct=True).astype(np.float32)
        df_aid_type_features[f'aid_{event_type}_is_session_end_count_rank_pct'] = df_aid_type_features[f'aid_{event_type}_is_session_end_count'].rank(pct=True).astype(np.float32)
        df_aid_type_features[f'aid_{event_type}_ts_ratio'] = (df_aid_type_features[f'aid_{event_type}_ts_max'] / df_aid_type_features[f'aid_{event_type}_ts_min']).astype(np.float32)
        logging.info(f'Created aid {event_type} additional features')
        df_aid_type_features.drop(columns=[
            f'aid_{event_type}_session_nunique', f'aid_{event_type}_day_of_year_nunique',
            f'aid_{event_type}_is_session_start_count', f'aid_{event_type}_is_session_end_count',
            f'aid_{event_type}_ts_min', f'aid_{event_type}_ts_max'
        ], inplace=True)
        logging.info(f'Down-casted {event_type} features')

        df_aid_features = df_aid_features.merge(df_aid_type_features.astype(np.float32), on='aid', how='left')
        del df_aid_type_features

    df_aid_features['aid_click_ratio'] = (df_aid_features['aid_click_count'] / df_aid_features['aid_count']).astype(np.float32)
    df_aid_features['aid_cart_ratio'] = (df_aid_features['aid_cart_count'] / df_aid_features['aid_count']).astype(np.float32)
    df_aid_features['aid_order_ratio'] = (df_aid_features['aid_order_count'] / df_aid_features['aid_count']).astype(np.float32)
    logging.info(f'Created additional event features')

    df_last_week = df.loc[df['week_of_year'] == df['week_of_year'].max()].reset_index(drop=True)
    df_aid_last_week_features = df_last_week.groupby('aid').agg({
        'aid': 'count',
        'session': 'nunique',
        'type': 'mean',
        'ts': ['max', 'min'],
        'hour': ['mean', 'std'],
        'day_of_week': ['mean', 'std'],
        'day_of_year': 'nunique',
        'session_cumcount_normalized': 'mean',
        'is_session_start': ['mean', 'count'],
        'is_session_end': ['mean', 'count'],
        'session_type+1_cumsum': 'mean'
    })
    logging.info(f'Created aid last week aggregation features')
    df_aid_last_week_features.columns = f'aid_last_week_' + df_aid_last_week_features.columns.map('_'.join).str.strip('_')
    df_aid_last_week_features = df_aid_last_week_features.rename(columns={'aid_last_week_aid_count': 'aid_last_week_count'})
    df_aid_last_week_features = df_aid_last_week_features.reset_index()

    df_aid_last_week_features['aid_last_week_count_rank_pct'] = df_aid_last_week_features['aid_last_week_count'].rank(pct=True).astype(np.float32)
    df_aid_last_week_features['aid_last_week_session_nunique_rank_pct'] = df_aid_last_week_features['aid_last_week_session_nunique'].rank(pct=True).astype(np.float32)
    df_aid_last_week_features['aid_last_week_day_of_year_nunique_rank_pct'] = df_aid_last_week_features['aid_last_week_day_of_year_nunique'].rank(pct=True).astype(np.float32)
    df_aid_last_week_features['aid_last_week_is_session_start_count_rank_pct'] = df_aid_last_week_features['aid_last_week_is_session_start_count'].rank(pct=True).astype(np.float32)
    df_aid_last_week_features['aid_last_week_is_session_end_count_rank_pct'] = df_aid_last_week_features['aid_last_week_is_session_end_count'].rank(pct=True).astype(np.float32)
    df_aid_last_week_features['aid_last_week_ts_ratio'] = (df_aid_last_week_features['aid_last_week_ts_max'] / df_aid_last_week_features['aid_last_week_ts_min']).astype(np.float32)
    logging.info('Created additional last week features')

    df_aid_features = df_aid_features.merge(df_aid_last_week_features.astype(np.float32), on='aid', how='left')
    del df_aid_last_week_features
    logging.info('Merged aid last week features')

    last_seven_days = sorted(df['day_of_year'].unique())[-7:]
    for nth, last_nth_day in enumerate(last_seven_days):

        nth = 7 - nth

        df_last_nth_day = df.loc[df['day_of_year'] == last_nth_day].reset_index(drop=True)
        df_aid_last_nth_day_features = df_last_nth_day.groupby('aid').agg({
            'aid': 'count',
            'session': 'nunique',
            'type': 'mean',
            'ts': ['max', 'min'],
            'hour': ['mean', 'std'],
            'day_of_week': ['mean', 'std'],
            'day_of_year': 'nunique',
            'session_cumcount_normalized': 'mean',
            'is_session_start': ['mean', 'count'],
            'is_session_end': ['mean', 'count'],
            'session_type+1_cumsum': 'mean'
        })
        logging.info(f'Created aid {nth} day aggregation features')
        df_aid_last_nth_day_features.columns = f'aid_last_{nth}_day_' + df_aid_last_nth_day_features.columns.map('_'.join).str.strip('_')
        df_aid_last_nth_day_features = df_aid_last_nth_day_features.rename(columns={f'aid_last_{nth}_day_aid_count': f'aid_last_{nth}_day_count'})
        df_aid_last_nth_day_features = df_aid_last_nth_day_features.reset_index()

        df_aid_last_nth_day_features[f'aid_last_{nth}_day_count_rank_pct'] = df_aid_last_nth_day_features[f'aid_last_{nth}_day_count'].rank(pct=True).astype(np.float32)
        df_aid_last_nth_day_features[f'aid_last_{nth}_day_session_nunique_rank_pct'] = df_aid_last_nth_day_features[f'aid_last_{nth}_day_session_nunique'].rank(pct=True).astype(np.float32)
        df_aid_last_nth_day_features[f'aid_last_{nth}_day_day_of_year_nunique_rank_pct'] = df_aid_last_nth_day_features[f'aid_last_{nth}_day_day_of_year_nunique'].rank(pct=True).astype(np.float32)
        df_aid_last_nth_day_features[f'aid_last_{nth}_day_is_session_start_count_rank_pct'] = df_aid_last_nth_day_features[f'aid_last_{nth}_day_is_session_start_count'].rank(pct=True).astype(np.float32)
        df_aid_last_nth_day_features[f'aid_last_{nth}_day_is_session_end_count_rank_pct'] = df_aid_last_nth_day_features[f'aid_last_{nth}_day_is_session_end_count'].rank(pct=True).astype(np.float32)
        df_aid_last_nth_day_features[f'aid_last_{nth}_day_ts_ratio'] = (df_aid_last_nth_day_features[f'aid_last_{nth}_day_ts_max'] / df_aid_last_nth_day_features[f'aid_last_{nth}_day_ts_min']).astype(np.float32)
        logging.info(f'Created additional last {nth} day features')

        df_aid_features = df_aid_features.merge(df_aid_last_nth_day_features.astype(np.float32), on='aid', how='left')
        del df_aid_last_nth_day_features
        logging.info(f'Merged aid last {nth} day features')

    group_ids = pd.MultiIndex.from_product([df['aid'].unique(), df['week_of_year'].unique(), [0, 1, 2]], names=['aid', 'week_of_year', 'type'])
    aid_counts = df.groupby(['aid', 'week_of_year', 'type'])['session'].count().rename('count')
    aid_counts = aid_counts.reindex(group_ids, fill_value=0).reset_index()
    aid_last_week_occurrence_ratio = (aid_counts.groupby(['aid', 'type'])['count'].last() / aid_counts.groupby(['aid', 'type'])['count'].sum()).fillna(0.).unstack('type')
    aid_last_week_occurrence_ratio.columns = [f'aid_{event_type}_last_week_occurrence_ratio' for event_type in ['click', 'cart', 'order']]
    df_aid_features = df_aid_features.merge(aid_last_week_occurrence_ratio.astype(np.float32).reset_index(), how='left', on='aid')
    del group_ids, aid_last_week_occurrence_ratio
    logging.info(f'Created last week occurrence features')

    aid_counts['pct_change'] = aid_counts.groupby(['aid', 'type'])['count'].pct_change()
    aid_last_week_pct_change = aid_counts.groupby(['aid', 'type'])['pct_change'].last().replace([np.inf, -np.inf], np.nan).unstack('type')
    aid_last_week_pct_change.columns = [f'aid_{event_type}_last_week_occurrence_pct_change' for event_type in ['click', 'cart', 'order']]
    df_aid_features = df_aid_features.merge(aid_last_week_pct_change.astype(np.float32).reset_index(), how='left', on='aid')
    del aid_counts, aid_last_week_pct_change
    logging.info(f'Created last week occurrence percentage change features')

    if args.mode == 'validation':
        df_aid_features.to_pickle(feature_engineering_directory / 'train_aid_features.pkl')
        logging.info(f'Saved train_aid_features.pkl to {feature_engineering_directory}')
    elif args.mode == 'submission':
        df_aid_features.to_pickle(feature_engineering_directory / 'train_and_test_aid_features.pkl')
        logging.info(f'Saved train_and_test_aid_features.pkl to {feature_engineering_directory}')
    else:
        raise ValueError('Invalid mode')
