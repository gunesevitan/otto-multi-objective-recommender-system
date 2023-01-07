import sys
import logging
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

sys.path.append('..')
import settings
import metrics
import visualization


if __name__ == '__main__':

    model_directory = Path(settings.MODELS / 'lightgbm')
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df_candidate = pd.read_pickle(settings.DATA / 'candidate' / 'recency_weighted_validation.pkl')
    df_candidate.sort_values(by='session', inplace=True)
    df_candidate.reset_index(drop=True, inplace=True)
    df_candidate['session_length'] = df_candidate.groupby('session')['session'].transform('count')

    df_train_labels = pd.read_pickle(settings.DATA / 'train_labels.pkl')
    df_train_labels['click_predictions'] = np.nan
    df_train_labels['click_predictions'] = df_train_labels['click_predictions'].astype(object)
    df_train_labels['cart_predictions'] = np.nan
    df_train_labels['cart_predictions'] = df_train_labels['cart_predictions'].astype(object)
    df_train_labels['order_predictions'] = np.nan
    df_train_labels['order_predictions'] = df_train_labels['order_predictions'].astype(object)
    df_train_labels = df_train_labels.set_index('session')

    logging.info(f'Candidate Dataset Shape: {df_candidate.shape} - Memory Usage: {df_candidate.memory_usage().sum() / 1024 ** 2:.2f} MB')
    logging.info(f'Train Labels Dataset Shape: {df_train_labels.shape} - Memory Usage: {df_train_labels.memory_usage().sum() / 1024 ** 2:.2f} MB')

    event_type = config['dataset']['event_type']
    evaluation_metric_functions = {
        'click': metrics.click_recall,
        'cart': metrics.cart_order_recall,
        'order': metrics.cart_order_recall,
    }
    logging.info(f'Running LightGBM Ranker model for training (Event Type: {event_type})')

    features = config['dataset'][event_type]['features']
    target = config['dataset'][event_type]['target']
    n_splits = 5
    df_feature_importance_gain = pd.DataFrame(
        data=np.zeros((len(features), n_splits)),
        index=features
    )
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_candidate, groups=df_candidate['session'])):

        df_train = df_candidate.loc[train_idx]
        df_val = df_candidate.loc[val_idx]

        logging.info(
            f'''
            Fold {fold + 1}
            Training - Candidates: {df_train.shape[0]} Sessions: {df_train["session"].nunique()} Target Mean: {df_train[target].mean():.4f}
            Validation - Candidates: {df_val.shape[0]} Sessions: {df_val["session"].nunique()} Target Mean: {df_val[target].mean():.4f}
            '''
        )

        query_train = df_train.groupby('session')['session'].count().values.tolist()
        query_val = df_val.groupby('session')['session'].count().values.tolist()
        train_dataset = lgb.Dataset(data=df_train[features], label=df_train[target], group=query_train)
        val_dataset = lgb.Dataset(data=df_val[features], label=df_val[target], group=query_val)

        model = lgb.train(
            params=config['model'],
            train_set=train_dataset,
            valid_sets=[train_dataset, val_dataset],
            num_boost_round=config['fit']['boosting_rounds'],
            callbacks=[
                lgb.early_stopping(config['fit']['early_stopping_rounds']),
                lgb.log_evaluation(config['fit']['verbose_eval'])
            ]
        )

        if config['persistence']['save_models']:
            model.save_model(
                model_directory / f'model_{event_type}.lgb',
                num_iteration=None,
                start_iteration=0,
                importance_type='gain'
            )
            logging.info(f'model_{event_type}.lgb is saved to {model_directory}')

        df_feature_importance_gain[fold] = model.feature_importance(importance_type='gain')

        # Predict validation dataset and retrieve top 20 predictions with the highest score for every session
        df_val[f'{event_type}_predictions'] = model.predict(df_val[features])
        df_val_predictions = df_val.sort_values(
            by=['session', f'{event_type}_predictions'],
            ascending=[True, False]
        ).groupby('session').head(20).reset_index(drop=True)[[
            f'{event_type}_candidates',
            'session', f'{event_type}_predictions',
        ]].groupby('session')[f'{event_type}_candidates'].agg(list)

        df_train_labels.loc[df_val_predictions.index, f'{event_type}_predictions'] = df_val_predictions.values
        df_train_labels.loc[df_val_predictions.index, f'{event_type}_recall'] = df_train_labels.loc[
            df_val_predictions.index,
            [f'{event_type}_labels', f'{event_type}_predictions']
        ].apply(lambda x: evaluation_metric_functions[event_type](x[f'{event_type}_labels'], x[f'{event_type}_predictions']), axis=1)
        val_mean_event_recall = df_train_labels.loc[df_val_predictions.index, f'{event_type}_recall'].mean()

        logging.info(f'Fold {fold + 1} - Event: {event_type} - Recall@20: {val_mean_event_recall:.6f}')

    oof_mean_event_recall = df_train_labels[f'{event_type}_recall'].mean()
    logging.info(f'OOF - Event: {event_type} - Recall@20: {oof_mean_event_recall:.6f}')

    # Visualize calculated model feature importance
    df_feature_importance_gain['mean'] = df_feature_importance_gain[[fold for fold in range(n_splits)]].mean(axis=1)
    df_feature_importance_gain['std'] = df_feature_importance_gain[[fold for fold in range(n_splits)]].std(axis=1)
    df_feature_importance_gain.sort_values(by='mean', ascending=False, inplace=True)

    if config['persistence']['visualize_feature_importance']:
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance_gain,
            path=model_directory / f'{event_type}_feature_importance_gain.png'
        )
        logging.info(f'Saved feature_importance_gain.png to {model_directory}')
