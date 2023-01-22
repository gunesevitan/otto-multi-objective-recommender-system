import sys
import logging
import argparse
import pathlib
import pandas as pd
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    feature_engineering_directory = pathlib.Path(settings.DATA / 'feature_engineering')
    feature_engineering_directory.mkdir(parents=True, exist_ok=True)

    for dataset_event_type in ['click', 'cart', 'order']:

        if args.mode == 'validation':
            logging.info(f'Running {dataset_event_type} interaction feature engineering in validation mode')
            df_candidate = pl.from_pandas(pd.read_pickle(settings.DATA / 'candidate' / f'{dataset_event_type}_validation.pkl'))
        elif args.mode == 'submission':
            logging.info(f'Running {dataset_event_type} interaction feature engineering in submission mode')
            df_candidate = pl.from_pandas(pd.read_pickle(settings.DATA / 'candidate' / f'{dataset_event_type}_test.pkl'))
        else:
            raise ValueError('Invalid mode')

        df_candidate = df_candidate.unique()
        df_candidate = df_candidate.sort('session')
        df_candidate = df_candidate.with_columns([pl.col('session').cast(pl.Int32), pl.col('candidates').cast(pl.Int32)])
        logging.info(f'Candidate Dataset Shape: {df_candidate.shape} - Memory Usage: {df_candidate.estimated_size() / 1024 ** 2:.2f} MB')

        if args.mode == 'validation':
            df = pl.DataFrame(pd.concat((
                pd.read_parquet(settings.DATA / 'splits' / 'train.parquet'),
                pd.read_parquet(settings.DATA / 'splits' / 'val.parquet')
            ), axis=0, ignore_index=True).reset_index(drop=True))
        elif args.mode == 'submission':
            df = pl.from_pandas(pd.read_pickle(settings.DATA / 'test.pkl'))
            df['ts'] /= 1000
        else:
            raise ValueError('Invalid mode')

        df = df.sort(by=['session', 'ts'], reverse=[False, False])
        logging.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.estimated_size() / 1024 ** 2:.2f} MB')

        df_candidate = df_candidate.join(df_candidate.groupby('session').agg([
            pl.col('candidates').count().alias('session_candidate_count').cast(pl.UInt16),
            pl.col('candidate_scores').mean().alias('session_candidate_score_mean'),
            pl.col('candidate_scores').std().alias('session_candidate_score_std'),
            pl.col('candidate_scores').min().alias('session_candidate_score_min'),
            pl.col('candidate_scores').max().alias('session_candidate_score_max'),
        ]), on='session', how='left')

        df = df.join(df.groupby(['session', 'aid']).agg(pl.col('aid').count().alias('session_candidate_occurrence_count')), on=['session', 'aid'], how='left')
        df = df.join(df.groupby(['session', 'aid', 'type']).agg(pl.col('aid').count().alias('session_candidate_type_count')), on=['session', 'aid', 'type'], how='left')
        df = df.sort(by=['session', 'ts'], reverse=[False, False])

        df_candidate = df_candidate.join(df[['session', 'aid', 'session_candidate_occurrence_count']].rename({
            'aid': 'candidates'
        }), on=['session', 'candidates'], how='left')
        df_candidate = df_candidate.with_column(pl.col(f'session_candidate_occurrence_count').fill_null(0).cast(pl.UInt16))

        for event_type_value, event_type in enumerate(['click', 'cart', 'order']):
            df_candidate = df_candidate.join(df.filter(df['type'] == event_type_value).rename({
                'aid': 'candidates', 'session_candidate_type_count': f'session_candidate_{event_type}_occurrence_count'
            })['session', 'candidates', f'session_candidate_{event_type}_occurrence_count'], on=['session', 'candidates'], how='left')
            df_candidate = df_candidate.with_column(pl.col(f'session_candidate_{event_type}_occurrence_count').fill_null(0).cast(pl.UInt16))
            df_candidate = df_candidate.unique()

        if args.mode == 'validation':
            df_candidate.to_pandas().to_pickle(feature_engineering_directory / f'train_{dataset_event_type}_interaction_features.pkl')
            logging.info(f'Saved train_{dataset_event_type}_interaction_features.pkl to {feature_engineering_directory}')
        elif args.mode == 'submission':
            df_candidate.to_pandas().to_pickle(feature_engineering_directory / f'test_{dataset_event_type}_interaction_features.pkl')
            logging.info(f'Saved test_{dataset_event_type}_interaction_features.pkl to {feature_engineering_directory}')
        else:
            raise ValueError('Invalid mode')
