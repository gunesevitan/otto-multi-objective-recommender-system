import sys
import logging
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import GroupKFold
import xgboost as xgb

sys.path.append('..')
import settings
import visualization


if __name__ == '__main__':

    model_directory = Path(settings.MODELS / 'xgboost')
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    event_type = config['dataset']['event_type']
    features = config['dataset'][event_type]['features']
    target = config['dataset'][event_type]['target']

    df_candidate = pl.from_pandas(pd.read_pickle(settings.DATA / 'feature_engineering' / f'train_{event_type}_interaction_features.pkl'))
    df_candidate = df_candidate.with_columns([pl.col('session').cast(pl.Int32), pl.col('candidates').cast(pl.Int32)])
    df_candidate = df_candidate[
        ['candidates', 'session', 'candidate_labels'] + [column for column in df_candidate.columns if column in features]
    ]
    df_candidate = df_candidate.unique()
    df_candidate = df_candidate.sort('session')
    logging.info(f'Candidate Dataset Shape: {df_candidate.shape} - Memory Usage: {df_candidate.estimated_size() / 1024 ** 2:.2f} MB')

    # Load and merge aid features that are used on training
    df_aid_features = pl.from_pandas(pd.read_pickle(settings.DATA / 'feature_engineering' / 'train_aid_features.pkl'))
    aid_merge_columns = [column for column in df_aid_features.columns if column in features]
    if len(aid_merge_columns) > 0:
        df_candidate = df_candidate.join(df_aid_features.rename({'aid': 'candidates'})[['candidates'] + aid_merge_columns], how='left', on='candidates')
    del df_aid_features
    logging.info(f'Candidate Dataset + aid Features Shape: {df_candidate.shape} - Memory Usage: {df_candidate.estimated_size() / 1024 ** 2:.2f} MB')

    # Load and merge session features that are used on training
    df_session_features = pl.from_pandas(pd.read_pickle(settings.DATA / 'feature_engineering' / 'train_session_features.pkl'))
    session_merge_columns = [column for column in df_session_features.columns if column in features]
    if len(session_merge_columns) > 0:
        df_candidate = df_candidate.join(df_session_features[['session'] + session_merge_columns], how='left', on='session')
    del df_session_features
    logging.info(f'Candidate Dataset + session Features Shape: {df_candidate.shape} - Memory Usage: {df_candidate.estimated_size() / 1024 ** 2:.2f} MB')

    # Load and merge validation sessions and validation labels
    df_validation = pl.from_pandas(pd.read_parquet(settings.DATA / 'splits' / 'val.parquet'))
    df_validation = df_validation.groupby('session').agg([pl.col('aid'), pl.col('type'), pl.col('ts')])

    df_val_labels = pd.read_parquet(settings.DATA / 'splits' / 'val_labels.parquet')
    df_val_labels['type'] = df_val_labels['type'].map({'clicks': 0, 'carts': 1, 'orders': 2})
    df_val_labels = pl.from_pandas(df_val_labels)
    df_val_labels = df_val_labels.with_column(pl.col('session').cast(pl.Int32))
    df_validation = df_validation.join(
        df_val_labels.filter(df_val_labels['type'] == 0)[['session', 'ground_truth']].rename({'ground_truth': 'click_labels'}),
        on='session',
        how='left'
    )
    df_validation = df_validation.join(
        df_val_labels.filter(df_val_labels['type'] == 1)[['session', 'ground_truth']].rename({'ground_truth': 'cart_labels'}),
        on='session',
        how='left'
    )
    df_validation = df_validation.join(
        df_val_labels.filter(df_val_labels['type'] == 2)[['session', 'ground_truth']].rename({'ground_truth': 'order_labels'}),
        on='session',
        how='left'
    )
    del df_val_labels
    df_validation = df_validation.to_pandas()
    df_validation = df_validation.fillna(df_validation.notna().applymap(lambda x: x or []))
    df_validation['predictions'] = np.nan
    df_validation['predictions'] = df_validation['predictions'].astype(object)
    df_validation = df_validation.set_index('session')
    logging.info(f'Validation Labels Shape: {df_validation.shape} - Memory Usage: {df_validation.memory_usage().sum() / 1024 ** 2:.2f} MB')

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X=df_candidate, groups=df_candidate['session']), 1):
        df_candidate = df_candidate.with_column(pl.lit(0).alias(f'fold{fold}').cast(pl.UInt8))
        df_candidate[val_idx, f'fold{fold}'] = 1
    folds = [column for column in df_candidate.columns if column.startswith('fold')]

    logging.info(
        f'''
        Running LightGBM Ranker model for training
        Event Type: {event_type}
        Features: {features}
        Target: {target}
        Folds: {n_splits}
        '''
    )

    df_feature_importance_gain = pd.DataFrame(
        data=np.zeros((len(features), len(folds))),
        index=features,
        columns=folds
    )
    df_feature_importance_weight = pd.DataFrame(
        data=np.zeros((len(features), len(folds))),
        index=features,
        columns=folds
    )
    df_feature_importance_cover = pd.DataFrame(
        data=np.zeros((len(features), len(folds))),
        index=features,
        columns=folds
    )
    df_candidate = df_candidate.with_column(pl.lit(0).alias('predictions').cast(pl.Float32))
    df_candidate = df_candidate.join(df_candidate.groupby('session').agg(pl.col(target).cast(pl.UInt16).sum().alias('session_target_sum')), on='session', how='left')

    for fold in folds:

        train_idx = df_candidate[fold].to_pandas() == 0
        val_idx = np.where(df_candidate[fold] == 1)[0]

        # Index negatives in training set and sample from them
        train_negative_labels = df_candidate[target].to_pandas().loc[
            train_idx & (df_candidate[target].to_pandas() == 0) & (df_candidate['session_target_sum'].to_pandas() > 0)
        ]
        negative_idx = train_negative_labels.sample(frac=config['dataset'][event_type]['negative_sampling_ratio'], random_state=42).index.to_numpy()
        del train_negative_labels
        # Combine train positive index and sampled negative index
        train_idx = np.hstack((
            np.where(train_idx & (df_candidate[target].to_pandas() == 1))[0],
            negative_idx
        ))
        # Sort training index for retaining session order
        train_idx.sort()

        # Extract occurrence counts of groups in training and validation sets
        query_train = np.unique(df_candidate[train_idx, 'session'], return_counts=True)[1].astype(np.uint16)
        query_val = np.unique(df_candidate[val_idx, 'session'], return_counts=True)[1].astype(np.uint16)

        train_dataset = xgb.DMatrix(
            data=df_candidate[train_idx, features].to_pandas(),
            label=df_candidate[train_idx, target].to_pandas(),
            group=query_train
        )
        val_dataset = xgb.DMatrix(
            data=df_candidate[val_idx, features].to_pandas(),
            label=df_candidate[val_idx, target].to_pandas(),
            group=query_val
        )
        del query_train, query_val

        logging.info(
            f'''
            {fold}
            Training - Candidates: {train_idx.shape[0]} Sessions: {len(np.unique(df_candidate[train_idx, 'session']))} Target Mean: {df_candidate[train_idx, target].mean().to_numpy()[0][0]:.4f}
            Validation - Candidates: {val_idx.shape[0]} Sessions: {len(np.unique(df_candidate[val_idx, 'session']))} Target Mean: {df_candidate[val_idx, target].mean().to_numpy()[0][0]:.4f}
            '''
        )

        model = xgb.train(
            params=config['model'][event_type],
            dtrain=train_dataset,
            evals=[(val_dataset, 'val')],
            num_boost_round=config['fit'][event_type]['boosting_rounds'],
            early_stopping_rounds=config['fit'][event_type]['early_stopping_rounds'],
            verbose_eval=config['fit'][event_type]['verbose_eval'],
        )
        del train_dataset, val_dataset

        if config['persistence']['save_models']:
            model.save_model(
                model_directory / f'model_{event_type}_{fold}.xgb',
            )
            logging.info(f'model_{event_type}_{fold}.xgb is saved to {model_directory}')

        for feature, importance in model.get_score(importance_type='gain').items():
            df_feature_importance_gain.loc[feature, fold] += importance
        for feature, importance in model.get_score(importance_type='weight').items():
            df_feature_importance_weight.loc[feature, fold] += importance
        for feature, importance in model.get_score(importance_type='cover').items():
            df_feature_importance_cover.loc[feature, fold] += importance

        # Predict validation dataset and retrieve top 20 predictions with the highest score for every session
        df_candidate[val_idx, 'predictions'] = np.float32(model.predict(xgb.DMatrix(df_candidate[val_idx, features].to_pandas())))
        del model
        df_val_predictions = df_candidate[
            val_idx, ['session', 'candidates', 'predictions']
        ].sort(by=['session', 'predictions'], reverse=[False, True])[[
            'session', 'candidates'
        ]].groupby('session').head(20).groupby('session').agg(
            pl.col('candidates')
        ).with_column(pl.col('session').cast(pl.Int32)).to_pandas().set_index('session')
        df_validation.loc[df_val_predictions.index, 'predictions'] = df_val_predictions.values.reshape(-1)
        df_validation.loc[df_val_predictions.index, 'hits'] = pl.DataFrame(
            df_validation.loc[df_val_predictions.index, ['predictions', f'{event_type}_labels']]
        ).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        val_recall = df_validation.loc[df_val_predictions.index, 'hits'].sum() / df_validation.loc[df_val_predictions.index, f'{event_type}_labels'].apply(len).clip(0, 20).sum()
        logging.info(f'{fold} - Event: {event_type} - Recall@20: {val_recall:.6f}')

    oof_recall = df_validation['hits'].sum() / df_validation[f'{event_type}_labels'].apply(len).clip(0, 20).sum()
    logging.info(f'OOF - Event: {event_type} - Recall@20: {oof_recall:.6f}')
    del df_validation

    if config['persistence']['visualize_feature_importance']:
        for importance_type, df_feature_importance in zip(['gain', 'weight', 'cover'], [df_feature_importance_gain, df_feature_importance_weight, df_feature_importance_cover]):
            # Visualize calculated model feature importance
            df_feature_importance['mean'] = df_feature_importance[folds].mean(axis=1)
            df_feature_importance['std'] = df_feature_importance[folds].std(axis=1)
            df_feature_importance.sort_values(by='mean', ascending=False, inplace=True)

            visualization.visualize_feature_importance(
                df_feature_importance=df_feature_importance,
                path=model_directory / f'feature_importance_{importance_type}_{event_type}.png'
            )
            logging.info(f'Saved feature_importance_{importance_type}_{event_type}.png to {model_directory}')

    if config['persistence']['save_val_predictions']:
        df_candidate = df_candidate['session', 'candidates', 'predictions'].to_pandas()
        df_candidate.to_pickle(model_directory / f'val_predictions_{event_type}.pkl')
        logging.info(f'Saved val_predictions_{event_type}.pkl to {model_directory}')

    train_predictions = df_candidate['predictions'].to_numpy()
    del df_candidate

    if config['persistence']['save_test_predictions']:

        df_candidate = pl.from_pandas(pd.read_pickle(settings.DATA / 'feature_engineering' / f'test_{event_type}_interaction_features.pkl'))
        df_candidate = df_candidate[
            ['candidates', 'session'] + [column for column in df_candidate.columns if column in features]
        ]
        df_candidate = df_candidate.unique()
        df_candidate = df_candidate.sort('session')
        logging.info(f'Candidate Dataset Shape: {df_candidate.shape} - Memory Usage: {df_candidate.estimated_size() / 1024 ** 2:.2f} MB')

        # Load and merge aid features that are used on training
        df_aid_features = pl.from_pandas(pd.read_pickle(settings.DATA / 'feature_engineering' / 'train_and_test_aid_features.pkl'))
        aid_merge_columns = [column for column in df_aid_features.columns if column in features]
        if len(aid_merge_columns) > 0:
            df_candidate = df_candidate.join(df_aid_features.rename({'aid': 'candidates'})[['candidates'] + aid_merge_columns], how='left', on='candidates')
        del df_aid_features
        logging.info(f'Candidate Dataset + aid Features Shape: {df_candidate.shape} - Memory Usage: {df_candidate.estimated_size() / 1024 ** 2:.2f} MB')

        # Load and merge session features that are used on training
        df_session_features = pl.from_pandas(pd.read_pickle(settings.DATA / 'feature_engineering' / 'test_session_features.pkl'))
        session_merge_columns = [column for column in df_session_features.columns if column in features]
        if len(session_merge_columns) > 0:
            df_candidate = df_candidate.join(df_session_features[['session'] + session_merge_columns], how='left', on='session')
        del df_session_features
        logging.info(f'Candidate Dataset + session Features Shape: {df_candidate.shape} - Memory Usage: {df_candidate.estimated_size() / 1024 ** 2:.2f} MB')

        test_predictions = np.zeros(df_candidate.shape[0])
        chunk_size = test_predictions.shape[0] // 20
        chunks = range((test_predictions.shape[0] // chunk_size) + 1)

        for fold in folds:

            model = xgb.Booster(model_file=model_directory / f'model_{event_type}_{fold}.xgb')

            for chunk_idx in chunks:

                chunk_start = chunk_idx * chunk_size
                chunk_end = (chunk_idx + 1) * chunk_size

                test_predictions[chunk_start:chunk_end] += (np.float32(model.predict(xgb.DMatrix(df_candidate[chunk_start:chunk_end, features].to_pandas()))) / len(folds))

        df_candidate = df_candidate.with_column(pl.Series(name='predictions', values=test_predictions))
        df_candidate = df_candidate['session', 'candidates', 'predictions'].to_pandas()
        df_candidate.to_pickle(model_directory / f'test_predictions_{event_type}.pkl')
        logging.info(f'Saved test_predictions_{event_type}.pkl to {model_directory}')

        if config['persistence']['visualize_predictions']:
            visualization.visualize_predictions(
                train_predictions=train_predictions,
                test_predictions=test_predictions,
                path=model_directory / f'predictions_{event_type}.png'
            )
            logging.info(f'Saved predictions_{event_type}.png to {model_directory}')
