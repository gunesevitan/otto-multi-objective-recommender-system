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
            df = df.with_columns([
                (pl.col('ts') / 1000),
                pl.col('session').cast(pl.Int32),
                pl.col('aid').cast(pl.Int32)
            ])
        else:
            raise ValueError('Invalid mode')

        df = df.filter(df['session'].is_in(df_candidate['session']))
        df = df.sort(by=['session', 'ts'], reverse=[False, False])
        logging.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.estimated_size() / 1024 ** 2:.2f} MB')

        df_cumulative = df.groupby(['session']).agg([
            pl.col('aid').cumcount().alias('session_aid_cumcount').cast(pl.UInt16)
        ]).explode(['session_aid_cumcount']).sort(by=['session', 'session_aid_cumcount'])
        df_cumulative = df_cumulative.with_columns(pl.col('session_aid_cumcount') + 1)
        df = pl.concat([df, df_cumulative[['session_aid_cumcount']]], how='horizontal')
        del df_cumulative
        df = df.join(df.groupby(['session', 'aid']).agg([
            pl.col('session_aid_cumcount').last().alias('session_candidate_cumcount_last').cast(pl.UInt16),
        ]), on=['session', 'aid'], how='left')

        df = df.join(df.groupby(['session', 'aid']).agg(pl.col('aid').count().alias('session_candidate_occurrence_count')), on=['session', 'aid'], how='left')
        df = df.join(df.groupby(['session', 'aid', 'type']).agg(pl.col('aid').count().alias('session_candidate_type_occurrence_count')), on=['session', 'aid', 'type'], how='left')
        df = df.sort(by=['session', 'ts'], reverse=[False, False])

        df_candidate = df_candidate.join(df[[
            'session', 'aid', 'session_candidate_occurrence_count',
            'session_candidate_cumcount_last'
        ]].rename({
            'aid': 'candidates'
        }), on=['session', 'candidates'], how='left')
        df_candidate = df_candidate.with_column(pl.col(f'session_candidate_occurrence_count').fill_null(0).cast(pl.UInt16))

        for event_type_value, event_type in enumerate(['click', 'cart', 'order']):
            df_candidate = df_candidate.join(df.filter(df['type'] == event_type_value).rename({
                'aid': 'candidates',
                'session_candidate_type_occurrence_count': f'session_candidate_{event_type}_occurrence_count',
            })['session', 'candidates', f'session_candidate_{event_type}_occurrence_count'], on=['session', 'candidates'], how='left')
            df_candidate = df_candidate.with_column(pl.col(f'session_candidate_{event_type}_occurrence_count').fill_null(0).cast(pl.UInt16))
            df_candidate = df_candidate.unique()

        df_candidate_session_features = df_candidate.groupby('session').agg([
            pl.col('candidate_scores').mean().alias('session_candidate_score_mean').cast(pl.Float32),
            pl.col('candidate_scores').std().alias('session_candidate_score_std').cast(pl.Float32),
            pl.col('candidate_scores').min().alias('session_candidate_score_min').cast(pl.Float32),
            pl.col('candidate_scores').max().alias('session_candidate_score_max').cast(pl.Float32),
            pl.col('session_candidate_occurrence_count').mean().alias('session_candidate_occurrence_count_mean').cast(pl.Float32),
            pl.col('session_candidate_occurrence_count').sum().alias('session_candidate_occurrence_count_sum').cast(pl.UInt32),
            pl.col('session_candidate_occurrence_count').max().alias('session_candidate_occurrence_count_max').cast(pl.UInt16),
            pl.col('session_candidate_cumcount_last').mean().alias('session_candidate_cumcount_last_mean').cast(pl.Float32),
            pl.col('session_candidate_cumcount_last').sum().alias('session_candidate_cumcount_last_sum').cast(pl.UInt32),
            pl.col('session_candidate_cumcount_last').max().alias('session_candidate_cumcount_last_max').cast(pl.UInt16),
        ])
        df_candidate = df_candidate.join(df_candidate_session_features, on='session', how='left')
        del df_candidate_session_features

        df_candidate_aid_features = df_candidate.groupby('candidates').agg([
            pl.col('candidate_scores').mean().alias('aid_candidate_score_mean').cast(pl.Float32),
            pl.col('candidate_scores').std().alias('aid_candidate_score_std').cast(pl.Float32),
            pl.col('candidate_scores').max().alias('aid_candidate_score_max').cast(pl.Float32),
            pl.col('session_candidate_occurrence_count').mean().alias('aid_session_candidate_occurrence_count_mean').cast(pl.Float32),
            pl.col('session_candidate_occurrence_count').sum().alias('aid_session_candidate_occurrence_count_sum').cast(pl.UInt32),
            pl.col('session_candidate_occurrence_count').max().alias('aid_session_candidate_occurrence_count_max').cast(pl.UInt16),
            pl.col('session_candidate_cumcount_last').mean().alias('aid_session_candidate_cumcount_last_mean').cast(pl.Float32),
            pl.col('session_candidate_cumcount_last').sum().alias('aid_session_candidate_cumcount_last_sum').cast(pl.UInt32),
            pl.col('session_candidate_cumcount_last').max().alias('aid_session_candidate_cumcount_last_max').cast(pl.UInt16),
        ])
        df_candidate = df_candidate.join(df_candidate_aid_features, on='candidates', how='left')
        del df_candidate_aid_features

        if args.mode == 'validation':
            df_candidate.to_pandas().to_pickle(feature_engineering_directory / f'train_{dataset_event_type}_interaction_features.pkl')
            logging.info(f'Saved train_{dataset_event_type}_interaction_features.pkl to {feature_engineering_directory}')
        elif args.mode == 'submission':
            df_candidate.to_pandas().to_pickle(feature_engineering_directory / f'test_{dataset_event_type}_interaction_features.pkl')
            logging.info(f'Saved test_{dataset_event_type}_interaction_features.pkl to {feature_engineering_directory}')
        else:
            raise ValueError('Invalid mode')
