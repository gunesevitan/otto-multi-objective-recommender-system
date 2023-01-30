import sys
import logging
import argparse
import pathlib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import RobustScaler

sys.path.append('..')
import settings


def read_predictions(predictions_file_paths, predictions_column, output_column, scale=True, file_format='parquet'):

    if len(predictions_file_paths) == 1:
        if file_format == 'parquet':
            df_predictions = pl.from_pandas(pd.read_parquet(predictions_file_paths[0]))
        elif file_format == 'pickle':
            df_predictions = pl.from_pandas(pd.read_pickle(predictions_file_paths[0]))
        else:
            raise ValueError(f'Invalid file format {file_format}')
    else:
        df_predictions = []
        for predictions_file_path in predictions_file_paths:
            if file_format == 'parquet':
                df_predictions.append(pl.from_pandas(pd.read_parquet(predictions_file_path)))
            elif file_format == 'pickle':
                df_predictions.append(pl.from_pandas(pd.read_pickle(predictions_file_path)))
            else:
                raise ValueError(f'Invalid file format {file_format}')

        df_predictions = pl.concat(df_predictions)

    if scale:
        df_predictions = df_predictions.with_column(
            pl.Series(
                name=predictions_column,
                values=RobustScaler().fit_transform(df_predictions[predictions_column].to_numpy().reshape(-1, 1)).reshape(-1)
            ).cast(pl.Float32)
        )

    if 'candidates' in df_predictions.columns:
        df_predictions = df_predictions.rename({'candidates': 'aid'})

    df_predictions = df_predictions.sort(by=['session', predictions_column], reverse=[False, True]).rename({
        predictions_column: output_column,
    })[['session', 'aid', output_column]]

    return df_predictions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    weights = {
        'click': {
            'tetsuro_lightgbm_click': 0.75,
            'gunes_lightgbm_click': 0.125,
            'gunes_xgboost_click': 0.125
        },
        'cart': {
            'tetsuro_lightgbm_cart': 0.1,
            'tetsuro_lightgbm_stack_cart': 0.8,
            'gunes_lightgbm_cart': 0.05,
            'gunes_xgboost_cart': 0.05
        },
        'order': {
            'tetsuro_lightgbm_order': 0.05,
            'tetsuro_lightgbm_stack_order': 0.85,
            'gunes_lightgbm_order': 0.05,
            'gunes_xgboost_order': 0.05,
        }
    }

    if args.mode == 'validation':

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
        df_validation['click_predictions'] = np.nan
        df_validation['click_predictions'] = df_validation['click_predictions'].astype(object)
        df_validation['cart_predictions'] = np.nan
        df_validation['cart_predictions'] = df_validation['cart_predictions'].astype(object)
        df_validation['order_predictions'] = np.nan
        df_validation['order_predictions'] = df_validation['order_predictions'].astype(object)
        df_validation = df_validation.set_index('session')
        logging.info(f'Validation Labels Shape: {df_validation.shape} - Memory Usage: {df_validation.memory_usage().sum() / 1024 ** 2:.2f} MB')

        df_tetsuro_lightgbm_click_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'tetsuro' / 'oof_lgbm_clicks.parquet'],
            predictions_column='score',
            output_column='tetsuro_lightgbm_click_predictions',
            scale=True,
            file_format='parquet'
        )
        logging.info(f'Tetsuro LightGBM Click Predictions Shape: {df_tetsuro_lightgbm_click_predictions.shape} - Memory Usage: {df_tetsuro_lightgbm_click_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_lightgbm_click_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'lightgbm' / 'val_predictions_click.pkl'],
            predictions_column='predictions',
            output_column='gunes_lightgbm_click_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes LightGBM Click Predictions Shape: {df_gunes_lightgbm_click_predictions.shape} - Memory Usage: {df_gunes_lightgbm_click_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_xgboost_click_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'xgboost' / 'val_predictions_click.pkl'],
            predictions_column='predictions',
            output_column='gunes_xgboost_click_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes XGBoost Click Predictions Shape: {df_gunes_xgboost_click_predictions.shape} - Memory Usage: {df_gunes_xgboost_click_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_click_predictions = df_gunes_lightgbm_click_predictions.join(df_gunes_xgboost_click_predictions, how='left', on=['session', 'aid'])
        df_click_predictions = df_click_predictions.join(df_tetsuro_lightgbm_click_predictions, on=['session', 'aid'], how='outer')
        df_click_predictions = df_click_predictions.fill_null(0)
        del df_gunes_lightgbm_click_predictions, df_gunes_xgboost_click_predictions, df_tetsuro_lightgbm_click_predictions

        df_click_predictions = df_click_predictions.with_columns(
            (
                    pl.col('gunes_lightgbm_click_predictions') * weights['click']['gunes_lightgbm_click'] +
                    pl.col('gunes_xgboost_click_predictions') * weights['click']['gunes_xgboost_click'] +
                    pl.col('tetsuro_lightgbm_click_predictions') * weights['click']['tetsuro_lightgbm_click']
            ).alias('predictions')
        )['session', 'aid', 'predictions']
        df_click_predictions = df_click_predictions.sort(by=['session', 'predictions'], reverse=[False, True])
        df_click_predictions = df_click_predictions.groupby('session').agg(pl.col('aid').head(20)).to_pandas().set_index('session')
        df_validation.loc[df_click_predictions.index, 'click_predictions'] = df_click_predictions.values.reshape(-1)
        df_validation['click_hits'] = pl.DataFrame(df_validation[['click_predictions', 'click_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        click_recall = df_validation['click_hits'].sum() / df_validation[f'click_labels'].apply(len).clip(0, 20).sum()

        df_tetsuro_lightgbm_cart_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'tetsuro' / 'oof_lgbm_carts.parquet'],
            predictions_column='score',
            output_column='tetsuro_lightgbm_cart_predictions',
            scale=True,
            file_format='parquet'
        )
        logging.info(f'Tetsuro LightGBM Cart Predictions Shape: {df_tetsuro_lightgbm_cart_predictions.shape} - Memory Usage: {df_tetsuro_lightgbm_cart_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_tetsuro_lightgbm_stack_cart_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'tetsuro' / 'oof_lgbm_carts_stack.parquet'],
            predictions_column='score',
            output_column='tetsuro_lightgbm_stack_cart_predictions',
            scale=True,
            file_format='parquet'
        )
        logging.info(f'Tetsuro LightGBM Stack Cart Predictions Shape: {df_tetsuro_lightgbm_stack_cart_predictions.shape} - Memory Usage: {df_tetsuro_lightgbm_stack_cart_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_lightgbm_cart_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'lightgbm' / 'val_predictions_cart.pkl'],
            predictions_column='predictions',
            output_column='gunes_lightgbm_cart_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes LightGBM Cart Predictions Shape: {df_gunes_lightgbm_cart_predictions.shape} - Memory Usage: {df_gunes_lightgbm_cart_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_xgboost_cart_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'xgboost' / 'val_predictions_cart.pkl'],
            predictions_column='predictions',
            output_column='gunes_xgboost_cart_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes XGBoost Cart Predictions Shape: {df_gunes_xgboost_cart_predictions.shape} - Memory Usage: {df_gunes_xgboost_cart_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_cart_predictions = df_gunes_lightgbm_cart_predictions.join(df_gunes_xgboost_cart_predictions, how='left', on=['session', 'aid'])
        df_cart_predictions = df_cart_predictions.join(df_tetsuro_lightgbm_cart_predictions, on=['session', 'aid'], how='outer')
        df_cart_predictions = df_cart_predictions.join(df_tetsuro_lightgbm_stack_cart_predictions, on=['session', 'aid'], how='outer')
        df_cart_predictions = df_cart_predictions.fill_null(0)
        del df_gunes_lightgbm_cart_predictions, df_gunes_xgboost_cart_predictions
        del df_tetsuro_lightgbm_cart_predictions, df_tetsuro_lightgbm_stack_cart_predictions

        df_cart_predictions = df_cart_predictions.with_columns(
            (
                    pl.col('gunes_lightgbm_cart_predictions') * weights['cart']['gunes_lightgbm_cart'] +
                    pl.col('gunes_xgboost_cart_predictions') * weights['cart']['gunes_xgboost_cart'] +
                    pl.col('tetsuro_lightgbm_cart_predictions') * weights['cart']['tetsuro_lightgbm_cart'] +
                    pl.col('tetsuro_lightgbm_stack_cart_predictions') * weights['cart']['tetsuro_lightgbm_stack_cart']
            ).alias('predictions')
        )['session', 'aid', 'predictions']
        df_cart_predictions = df_cart_predictions.sort(by=['session', 'predictions'], reverse=[False, True])
        df_cart_predictions = df_cart_predictions.groupby('session').agg(pl.col('aid').head(20)).to_pandas().set_index('session')
        df_validation.loc[df_cart_predictions.index, 'cart_predictions'] = df_cart_predictions.values.reshape(-1)
        df_validation['cart_hits'] = pl.DataFrame(df_validation[['cart_predictions', 'cart_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        cart_recall = df_validation['cart_hits'].sum() / df_validation[f'cart_labels'].apply(len).clip(0, 20).sum()

        df_tetsuro_lightgbm_order_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'tetsuro' / 'oof_lgbm_orders.parquet'],
            predictions_column='score',
            output_column='tetsuro_lightgbm_order_predictions',
            scale=True,
            file_format='parquet'
        )
        logging.info(f'Tetsuro LightGBM Order Predictions Shape: {df_tetsuro_lightgbm_order_predictions.shape} - Memory Usage: {df_tetsuro_lightgbm_order_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_tetsuro_lightgbm_stack_order_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'tetsuro' / 'oof_lgbm_orders_stack.parquet'],
            predictions_column='score',
            output_column='tetsuro_lightgbm_stack_order_predictions',
            scale=True,
            file_format='parquet'
        )
        logging.info(f'Tetsuro LightGBM Stack Order Predictions Shape: {df_tetsuro_lightgbm_stack_order_predictions.shape} - Memory Usage: {df_tetsuro_lightgbm_stack_order_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_lightgbm_order_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'lightgbm' / 'val_predictions_order.pkl'],
            predictions_column='predictions',
            output_column='gunes_lightgbm_order_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes LightGBM Order Predictions Shape: {df_gunes_lightgbm_order_predictions.shape} - Memory Usage: {df_gunes_lightgbm_order_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_xgboost_order_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'xgboost' / 'val_predictions_order.pkl'],
            predictions_column='predictions',
            output_column='gunes_xgboost_order_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes XGBoost Order Predictions Shape: {df_gunes_xgboost_order_predictions.shape} - Memory Usage: {df_gunes_lightgbm_order_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_order_predictions = df_gunes_lightgbm_order_predictions.join(df_gunes_xgboost_order_predictions, how='left', on=['session', 'aid'])
        df_order_predictions = df_order_predictions.join(df_tetsuro_lightgbm_order_predictions, on=['session', 'aid'], how='outer')
        df_order_predictions = df_order_predictions.join(df_tetsuro_lightgbm_stack_order_predictions, on=['session', 'aid'], how='outer')
        df_order_predictions = df_order_predictions.fill_null(0)
        del df_gunes_lightgbm_order_predictions, df_gunes_xgboost_order_predictions, df_tetsuro_lightgbm_order_predictions

        df_order_predictions = df_order_predictions.with_columns(
            (
                    pl.col('gunes_lightgbm_order_predictions') * weights['order']['gunes_lightgbm_order'] +
                    pl.col('gunes_xgboost_order_predictions') * weights['order']['gunes_xgboost_order'] +
                    pl.col('tetsuro_lightgbm_order_predictions') * weights['order']['tetsuro_lightgbm_order'] +
                    pl.col('tetsuro_lightgbm_stack_order_predictions') * weights['order']['tetsuro_lightgbm_stack_order']
            ).alias('predictions')
        )['session', 'aid', 'predictions']
        df_order_predictions = df_order_predictions.sort(by=['session', 'predictions'], reverse=[False, True])
        df_order_predictions = df_order_predictions.groupby('session').agg(pl.col('aid').head(20)).to_pandas().set_index('session')
        df_validation.loc[df_order_predictions.index, 'order_predictions'] = df_order_predictions.values.reshape(-1)
        df_validation['order_hits'] = pl.DataFrame(df_validation[['order_predictions', 'order_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        order_recall = df_validation['order_hits'].sum() / df_validation[f'order_labels'].apply(len).clip(0, 20).sum()
        weighted_recall = (click_recall * 0.1) + (cart_recall * 0.3) + (order_recall * 0.6)

        logging.info(
            f'''
            Validation scores
            clicks - recall@20: {click_recall:.6f}
            carts - recall@20: {cart_recall:.6f}
            orders - recall@20: {order_recall:.6f}
            weighted recall@20: {weighted_recall:.6f}
            '''
        )

    elif args.mode == 'submission':

        # Create a directory for saving submission file
        submissions_directory = pathlib.Path(settings.DATA / 'submissions')
        submissions_directory.mkdir(parents=True, exist_ok=True)

        df_tetsuro_lightgbm_click_predictions = read_predictions(
            predictions_file_paths=[
                settings.MODELS / 'tetsuro' / 'test_predictions_clicks_0_.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_clicks_1_.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_clicks_2_.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_clicks_3_.parquet',
            ],
            predictions_column='score',
            output_column='tetsuro_lightgbm_click_predictions',
            scale=True,
            file_format='parquet'
        )
        logging.info(f'Tetsuro LightGBM Click Predictions Shape: {df_tetsuro_lightgbm_click_predictions.shape} - Memory Usage: {df_tetsuro_lightgbm_click_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_lightgbm_click_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'lightgbm' / 'test_predictions_click.pkl'],
            predictions_column='predictions',
            output_column='gunes_lightgbm_click_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes LightGBM Click Predictions Shape: {df_gunes_lightgbm_click_predictions.shape} - Memory Usage: {df_gunes_lightgbm_click_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_xgboost_click_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'xgboost' / 'test_predictions_click.pkl'],
            predictions_column='predictions',
            output_column='gunes_xgboost_click_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes XGBoost Click Predictions Shape: {df_gunes_xgboost_click_predictions.shape} - Memory Usage: {df_gunes_xgboost_click_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_click_predictions = df_gunes_lightgbm_click_predictions.join(df_gunes_xgboost_click_predictions, how='left', on=['session', 'aid'])
        df_click_predictions = df_click_predictions.join(df_tetsuro_lightgbm_click_predictions, on=['session', 'aid'], how='outer')
        df_click_predictions = df_click_predictions.fill_null(0)
        del df_gunes_lightgbm_click_predictions, df_gunes_xgboost_click_predictions, df_tetsuro_lightgbm_click_predictions
        logging.info(f'Merged Click Predictions Shape: {df_click_predictions.shape} - Memory Usage: {df_click_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_click_predictions = df_click_predictions.with_columns(
            (
                    pl.col('gunes_lightgbm_click_predictions') * weights['click']['gunes_lightgbm_click'] +
                    pl.col('gunes_xgboost_click_predictions') * weights['click']['gunes_xgboost_click'] +
                    pl.col('tetsuro_lightgbm_click_predictions') * weights['click']['tetsuro_lightgbm_click']
            ).alias('predictions')
        )['session', 'aid', 'predictions']
        df_click_predictions = df_click_predictions.sort(by=['session', 'predictions'], reverse=[False, True])
        df_click_predictions = df_click_predictions.groupby('session').agg(pl.col('aid').head(20))
        logging.info(f'Filtered Click Predictions Shape: {df_click_predictions.shape} - Memory Usage: {df_click_predictions.estimated_size() / 1024 ** 2:.2f} MB')
        df_click_predictions = df_click_predictions.apply(
            lambda x: (str(x[0]) + '_clicks', ' '.join([str(aid) for aid in x[1]]))
        ).rename({'column_0': 'session_type', 'column_1': 'labels'})

        df_tetsuro_lightgbm_cart_predictions = read_predictions(
            predictions_file_paths=[
                settings.MODELS / 'tetsuro' / 'test_predictions_carts_0_.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_carts_1_.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_carts_2_.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_carts_3_.parquet',
            ],
            predictions_column='score',
            output_column='tetsuro_lightgbm_cart_predictions',
            scale=True,
            file_format='parquet'
        )
        logging.info(f'Tetsuro LightGBM Cart Predictions Shape: {df_tetsuro_lightgbm_cart_predictions.shape} - Memory Usage: {df_tetsuro_lightgbm_cart_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_tetsuro_lightgbm_stack_cart_predictions = read_predictions(
            predictions_file_paths=[
                settings.MODELS / 'tetsuro' / 'test_predictions_carts_0_stack.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_carts_1_stack.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_carts_2_stack.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_carts_3_stack.parquet',
            ],
            predictions_column='score',
            output_column='tetsuro_lightgbm_stack_cart_predictions',
            scale=True,
            file_format='parquet'
        )
        logging.info(f'Tetsuro LightGBM Stack Cart Predictions Shape: {df_tetsuro_lightgbm_stack_cart_predictions.shape} - Memory Usage: {df_tetsuro_lightgbm_stack_cart_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_lightgbm_cart_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'lightgbm' / 'test_predictions_cart.pkl'],
            predictions_column='predictions',
            output_column='gunes_lightgbm_cart_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes LightGBM Cart Predictions Shape: {df_gunes_lightgbm_cart_predictions.shape} - Memory Usage: {df_gunes_lightgbm_cart_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_xgboost_cart_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'xgboost' / 'test_predictions_cart.pkl'],
            predictions_column='predictions',
            output_column='gunes_xgboost_cart_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes XGBoost Cart Predictions Shape: {df_gunes_xgboost_cart_predictions.shape} - Memory Usage: {df_gunes_xgboost_cart_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_cart_predictions = df_gunes_lightgbm_cart_predictions.join(df_gunes_xgboost_cart_predictions, how='left', on=['session', 'aid'])
        df_cart_predictions = df_cart_predictions.join(df_tetsuro_lightgbm_cart_predictions, on=['session', 'aid'], how='outer')
        df_cart_predictions = df_cart_predictions.join(df_tetsuro_lightgbm_stack_cart_predictions, on=['session', 'aid'], how='outer')
        df_cart_predictions = df_cart_predictions.fill_null(0)
        del df_gunes_lightgbm_cart_predictions, df_gunes_xgboost_cart_predictions
        del df_tetsuro_lightgbm_cart_predictions, df_tetsuro_lightgbm_stack_cart_predictions
        logging.info(f'Merged Cart Predictions Shape: {df_cart_predictions.shape} - Memory Usage: {df_cart_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_cart_predictions = df_cart_predictions.with_columns(
            (
                    pl.col('gunes_lightgbm_cart_predictions') * weights['cart']['gunes_lightgbm_cart'] +
                    pl.col('gunes_xgboost_cart_predictions') * weights['cart']['gunes_xgboost_cart'] +
                    pl.col('tetsuro_lightgbm_cart_predictions') * weights['cart']['tetsuro_lightgbm_cart'] +
                    pl.col('tetsuro_lightgbm_stack_cart_predictions') * weights['cart']['tetsuro_lightgbm_stack_cart']
            ).alias('predictions')
        )['session', 'aid', 'predictions']
        df_cart_predictions = df_cart_predictions.sort(by=['session', 'predictions'], reverse=[False, True])
        df_cart_predictions = df_cart_predictions.groupby('session').agg(pl.col('aid').head(20))
        logging.info(f'Filtered Cart Predictions Shape: {df_cart_predictions.shape} - Memory Usage: {df_cart_predictions.estimated_size() / 1024 ** 2:.2f} MB')
        df_cart_predictions = df_cart_predictions.apply(
            lambda x: (str(x[0]) + '_carts', ' '.join([str(aid) for aid in x[1]]))
        ).rename({'column_0': 'session_type', 'column_1': 'labels'})

        df_tetsuro_lightgbm_order_predictions = read_predictions(
            predictions_file_paths=[
                settings.MODELS / 'tetsuro' / 'test_predictions_orders_0_.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_orders_1_.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_orders_2_.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_orders_3_.parquet',
            ],
            predictions_column='score',
            output_column='tetsuro_lightgbm_order_predictions',
            scale=True,
            file_format='parquet'
        )
        logging.info(f'Tetsuro LightGBM Order Predictions Shape: {df_tetsuro_lightgbm_order_predictions.shape} - Memory Usage: {df_tetsuro_lightgbm_order_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_tetsuro_lightgbm_stack_order_predictions = read_predictions(
            predictions_file_paths=[
                settings.MODELS / 'tetsuro' / 'test_predictions_orders_0_stack.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_orders_1_stack.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_orders_2_stack.parquet',
                settings.MODELS / 'tetsuro' / 'test_predictions_orders_3_stack.parquet',
            ],
            predictions_column='score',
            output_column='tetsuro_lightgbm_stack_order_predictions',
            scale=True,
            file_format='parquet'
        )
        logging.info(f'Tetsuro LightGBM Stack Order Predictions Shape: {df_tetsuro_lightgbm_stack_order_predictions.shape} - Memory Usage: {df_tetsuro_lightgbm_stack_order_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_lightgbm_order_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'lightgbm' / 'test_predictions_order.pkl'],
            predictions_column='predictions',
            output_column='gunes_lightgbm_order_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes LightGBM Order Predictions Shape: {df_gunes_lightgbm_order_predictions.shape} - Memory Usage: {df_gunes_lightgbm_order_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_gunes_xgboost_order_predictions = read_predictions(
            predictions_file_paths=[settings.MODELS / 'xgboost' / 'test_predictions_order.pkl'],
            predictions_column='predictions',
            output_column='gunes_xgboost_order_predictions',
            scale=True,
            file_format='pickle'
        )
        logging.info(f'Gunes XGBoost Order Predictions Shape: {df_gunes_xgboost_order_predictions.shape} - Memory Usage: {df_gunes_lightgbm_order_predictions.estimated_size() / 1024 ** 2:.2f} MB')

        df_order_predictions = df_gunes_lightgbm_order_predictions.join(df_gunes_xgboost_order_predictions, how='left', on=['session', 'aid'])
        df_order_predictions = df_order_predictions.join(df_tetsuro_lightgbm_order_predictions, on=['session', 'aid'], how='outer')
        df_order_predictions = df_order_predictions.join(df_tetsuro_lightgbm_stack_order_predictions, on=['session', 'aid'], how='outer')
        df_order_predictions = df_order_predictions.fill_null(0)
        del df_gunes_lightgbm_order_predictions, df_gunes_xgboost_order_predictions
        del df_tetsuro_lightgbm_order_predictions, df_tetsuro_lightgbm_stack_order_predictions
        logging.info(f'Merged Order Predictions Shape: {df_order_predictions.shape} - Memory Usage: {df_order_predictions.estimated_size() / 1024 ** 2:.2f} MB')
        df_order_predictions = df_order_predictions.with_columns(
            (
                    pl.col('gunes_lightgbm_order_predictions') * weights['order']['gunes_lightgbm_order'] +
                    pl.col('gunes_xgboost_order_predictions') * weights['order']['gunes_xgboost_order'] +
                    pl.col('tetsuro_lightgbm_order_predictions') * weights['order']['tetsuro_lightgbm_order'] +
                    pl.col('tetsuro_lightgbm_stack_order_predictions') * weights['order']['tetsuro_lightgbm_stack_order']
            ).alias('predictions')
        )['session', 'aid', 'predictions']
        df_order_predictions = df_order_predictions.sort(by=['session', 'predictions'], reverse=[False, True])
        df_order_predictions = df_order_predictions.groupby('session').agg(pl.col('aid').head(20))
        logging.info(f'Filtered Order Predictions Shape: {df_order_predictions.shape} - Memory Usage: {df_order_predictions.estimated_size() / 1024 ** 2:.2f} MB')
        df_order_predictions = df_order_predictions.apply(
            lambda x: (str(x[0]) + '_orders', ' '.join([str(aid) for aid in x[1]]))
        ).rename({'column_0': 'session_type', 'column_1': 'labels'})

        df_predictions = pl.concat([df_click_predictions, df_cart_predictions, df_order_predictions])
        del df_click_predictions, df_cart_predictions, df_order_predictions
        df_predictions.to_pandas().to_csv(submissions_directory / 'blend_submission.csv.gz', index=False, compression='gzip')
        logging.info(f'blend_submission.csv.gz is saved to {submissions_directory}')
