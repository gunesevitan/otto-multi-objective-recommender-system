import sys
import logging
import argparse
import pathlib
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn
import torch.optim as optim
from merlin.loader.torch import Loader
from merlin.io import Dataset

sys.path.append('..')
import settings
import torch_modules
import torch_utils
import metrics
import visualization


def train(train_loader, model, criterion, optimizer, device, scheduler=None):

    """
    Train given model on given data loader

    Parameters
    ----------
    train_loader: torch.utils.data.DataLoader
        Training set data loader

    model: torch.nn.Module
        Model to train

    criterion: torch.nn.Module
        Loss function

    optimizer: torch.optim.Optimizer
        Optimizer

    device: torch.device
        Location of the model and inputs

    scheduler: torch.optim.LRScheduler or None
        Learning rate scheduler

    Returns
    -------
    train_loss: float
        Average training loss after model is fully trained on training set data loader
    """

    model.train()
    progress_bar = tqdm(train_loader)
    losses = []

    for inputs, _ in progress_bar:
        if isinstance(model, torch_modules.CollaborativeFiltering):
            # Pass aid pairs to collaborative filtering model
            x1, x2, targets = inputs['x1'].to(device), inputs['x2'].to(device), inputs['target'].float().to(device)
            outputs = model(x1, x2)
            loss = criterion(outputs, targets)
        elif isinstance(model, torch_modules.MatrixFactorization):
            # Pass session and aid to matrix factorization model
            sessions, aids, targets = inputs['session'].to(device), inputs['aid'].to(device), inputs['target'].float().to(device)
            outputs = model(sessions, aids)
            loss = criterion(outputs, targets)
        else:
            raise ValueError('Invalid model')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.detach().item())
        average_loss = np.mean(losses)
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        progress_bar.set_description(f'train_loss: {average_loss:.6f} - lr: {lr:.8f}')

    train_loss = np.mean(losses)
    return train_loss


def validate(val_loader, model, criterion, device, scores=False):

    """
    Validate given model on given data loader

    Parameters
    ----------
    val_loader: torch.utils.data.DataLoader
        Validation set data loader

    model: torch.nn.Module
        Model to validate

    criterion: torch.nn.Module
        Loss function

    device: torch.device
        Location of the model and inputs

    scores: bool
        Whether to calculate validation scores or not

    Returns
    -------
    val_loss: float
        Average validation loss after model is fully validated on validation set data loader

    val_scores: dict or None
        Dictionary of metric scores after model is fully validated on validation set data loader
    """

    model.eval()
    progress_bar = tqdm(val_loader)
    losses = []
    if scores:
        ground_truth = []
        predictions = []

    with torch.no_grad():
        for inputs, _ in progress_bar:
            if isinstance(model, torch_modules.CollaborativeFiltering):
                # Pass session or aid pairs to collaborative filtering model
                x1, x2, targets = inputs['x1'].to(device), inputs['x2'].to(device), inputs['target'].float().to(device)
                outputs = model(x1, x2)
                loss = criterion(outputs, targets)
            elif isinstance(model, torch_modules.MatrixFactorization):
                # Pass session and aid to matrix factorization model
                sessions, aids, targets = inputs['session'].to(device), inputs['aid'].to(device), inputs['target'].float().to(device)
                outputs = model(sessions, aids)
                loss = criterion(outputs, targets)
            else:
                raise ValueError('Invalid model')

            losses.append(loss.detach().item())
            average_loss = np.mean(losses)
            progress_bar.set_description(f'val_loss: {average_loss:.6f}')
            if scores:
                ground_truth += [(targets.detach().cpu())]
                predictions += [(outputs.detach().cpu())]

    val_loss = np.mean(losses)
    if scores:
        ground_truth = torch.cat(ground_truth, dim=0).numpy()
        if isinstance(model, torch_modules.CollaborativeFiltering):
            predictions = torch.sigmoid(torch.cat(predictions, dim=0)).numpy()
            val_scores = metrics.classification_scores(y_true=ground_truth, y_pred=predictions, threshold=0.5)
        elif isinstance(model, torch_modules.MatrixFactorization):
            predictions = torch.cat(predictions, dim=0).numpy()
            val_scores = metrics.regression_scores(y_true=ground_truth, y_pred=predictions)
        else:
            raise ValueError('Invalid model')
    else:
        val_scores = None

    return val_loss, val_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    config = yaml.load(open(settings.MODELS / args.config_path, 'r'), Loader=yaml.FullLoader)

    df = pd.concat((
        pd.read_pickle(settings.DATA / 'train.pkl'),
        pd.read_pickle(settings.DATA / 'test.pkl')
    ), axis=0, ignore_index=True)
    logging.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    if config['model']['model_class'] == 'CollaborativeFiltering':

        # Create directory for model specific dataset
        dataset_root_directory = pathlib.Path(settings.DATA / 'collaborative_filtering')
        dataset_root_directory.mkdir(parents=True, exist_ok=True)

        if config['dataset']['load_dataset']:
            # Load pre-computed dataset for aid x aid collaborative filtering model
            logging.info(f'Using pre-computed dataset from {dataset_root_directory / "aid_pairs.parquet"}')
            df_aid_pairs = pd.read_parquet(str(dataset_root_directory / 'aid_pairs.parquet'))
        else:
            # Create dataset for aid x aid collaborative filtering model
            if config['dataset']['sampling_strategy'] == 'time':

                df_aid_pairs = pd.DataFrame(columns=['aid_x', 'aid_y', 'target'], dtype=int)
                df.index = pd.MultiIndex.from_frame(df[['session']])

                sessions = df['session'].unique()
                logging.info(f'Creating aid pairs dataset from {len(sessions)} sessions and {df.shape[0]} events')

                for i in tqdm(range(0, sessions.shape[0], config['dataset']['chunk_size'])):
                    # Index a chunk of sessions
                    df_sessions = df.loc[sessions[i]:sessions[min(sessions.shape[0] - 1, i + config['dataset']['chunk_size'] - 1)]].reset_index(drop=True).sample(frac=0.15)
                    # Merge groups of sessions with itself for vectorized comparisons
                    df_aid_pairs_ = df_sessions.merge(df_sessions, on='session')
                    # Drop same aid pairs
                    df_aid_pairs_ = df_aid_pairs_.loc[df_aid_pairs_['aid_x'] != df_aid_pairs_['aid_y'], :]
                    # Calculate hour difference between aid pairs and sample positives/negatives based on the specified condition
                    df_aid_pairs_['hours_elapsed'] = (df_aid_pairs_['ts_y'] - df_aid_pairs_['ts_x']).dt.seconds / 3600
                    df_aid_pairs_['target'] = 0
                    df_aid_pairs_.loc[
                        (df_aid_pairs_['hours_elapsed'] > 0) & (df_aid_pairs_['hours_elapsed'] <= config['dataset']['hour_difference']),
                        'target'
                    ] = 1
                    df_aid_pairs = pd.concat((
                        df_aid_pairs,
                        df_aid_pairs_[['aid_x', 'aid_y', 'target']]
                    ), axis=0, ignore_index=True)

                if config['dataset']['target_aggregation'] == 'mean':
                    # Assign positive label to aid pairs if their target mean is greater than 0.5
                    df_aid_pairs = df_aid_pairs.groupby(['aid_x', 'aid_y'])['target'].mean().reset_index()
                    df_aid_pairs['target'] = (df_aid_pairs['target'] >= 0.5).astype(int)
                elif config['dataset']['target_aggregation'] == 'max':
                    # Assign positive label to aid pairs if their target is positive at least once
                    df_aid_pairs = df_aid_pairs.groupby(['aid_x', 'aid_y'])['target'].max().reset_index()
                else:
                    raise ValueError('Invalid target aggregation')

                df_aid_pairs.rename(columns={'aid_x': 'x1', 'aid_y': 'x2'}, inplace=True)

            elif config['dataset']['sampling_strategy'] == 'diff':
                # Positive aid pairs are aid shifted by -1
                # Negative aid pairs are random aids within each session
                df_aid_pairs = pl.DataFrame(df).groupby('session').agg([
                    pl.col('aid').alias('x1'),
                    pl.col('aid').shift(-1).alias('x2'),
                    pl.col('aid').shuffle().alias('x3')
                ]).explode(['x1', 'x2', 'x3']).drop_nulls()

                # Filter out same positive/negative aid pairs and create negative target column
                df_negative_aid_pairs = df_aid_pairs.filter(
                    (pl.col('x2') != pl.col('x3')) & (pl.col('x1') != pl.col('x3')) & (pl.col('x1') != pl.col('x3'))
                ).select(['x1', 'x3'])
                df_negative_aid_pairs.columns = ['x1', 'x2']
                df_negative_aid_pairs = df_negative_aid_pairs.unique(subset=['x1', 'x2'])
                df_negative_aid_pairs = df_negative_aid_pairs.with_column(pl.lit(0).cast(pl.Int8).alias('target'))
                # Filter out same positive/negative aid pairs and create positive target column
                df_aid_pairs = df_aid_pairs.filter(
                    (pl.col('x2') != pl.col('x3')) & (pl.col('x1') != pl.col('x2')) & (pl.col('x1') != pl.col('x3'))
                ).select(['x1', 'x2'])
                df_aid_pairs = df_aid_pairs.unique(subset=['x1', 'x2'])
                df_aid_pairs = df_aid_pairs.with_column(pl.lit(1).cast(pl.Int8).alias('target'))
                # Concatenate positive/negative aid pairs and duplicates
                df_aid_pairs = pl.concat((df_aid_pairs, df_negative_aid_pairs))
                del df_negative_aid_pairs
                df_aid_pairs = df_aid_pairs.unique(subset=['x1', 'x2'])
                df_aid_pairs = df_aid_pairs.to_pandas()
            else:
                raise ValueError('Invalid sampling strategy')

            # Cast created aid pair dataset to int data type and save it as a parquet file
            df_aid_pairs.astype(int).to_parquet(dataset_root_directory / 'aid_pairs.parquet')
            logging.info(f'aid_pairs.parquet is saved to {dataset_root_directory}')

        del df
        n_pairs = df_aid_pairs.shape[0]
        n_aids = max(df_aid_pairs['x1'].max(), df_aid_pairs['x2'].max())

        logging.info(
            f'''
            aid pairs dataset - pairs: {n_pairs} aids: {n_aids})
            {np.sum(df_aid_pairs['target'] == 1)} ({((np.sum(df_aid_pairs['target'] == 1) / n_pairs) * 100):.2f}%) positive aid pairs
            {np.sum(df_aid_pairs['target'] == 0)} ({((np.sum(df_aid_pairs['target'] == 0) / n_pairs) * 100):.2f}%) negative aid pairs
            '''
        )

    elif config['model']['model_class'] == 'MatrixFactorization':

        # Create directory for model specific dataset
        dataset_root_directory = pathlib.Path(settings.DATA / 'matrix_factorization')
        dataset_root_directory.mkdir(parents=True, exist_ok=True)

        if config['dataset']['load_dataset']:
            # Load pre-computed dataset for aid x aid collaborative filtering model
            logging.info(f'Using pre-computed dataset from {dataset_root_directory / "sessions_aids.parquet"}')
            df_sessions_aids = pd.read_parquet(str(dataset_root_directory / 'sessions_aids.parquet'))
        else:
            df_session_aids = df.rename(columns={'type': 'target'})[['session', 'aid', 'target']]
            df_session_aids.astype(int).to_parquet(dataset_root_directory / 'sessions_aids.parquet')

        del df
        n_samples = df_sessions_aids.shape[0]
        n_sessions = df_sessions_aids['session'].max()
        n_aids = df_sessions_aids['aid'].max()

        logging.info(
            f'''
            sessions and aids dataset - sessions: {n_sessions} aids: {n_aids})
            {np.sum(df_sessions_aids['target'] == 0)} ({((np.sum(df_sessions_aids['target'] == 0) / n_samples) * 100):.2f}%) clicks
            {np.sum(df_sessions_aids['target'] == 1)} ({((np.sum(df_sessions_aids['target'] == 1) / n_samples) * 100):.2f}%) carts
            {np.sum(df_sessions_aids['target'] == 2)} ({((np.sum(df_sessions_aids['target'] == 2) / n_samples) * 100):.2f}%) orders
            '''
        )
    else:
        raise ValueError('Invalid model')

    # Create training and validation datasets and data loaders
    if config['model']['model_class'] == 'CollaborativeFiltering':
        train_dataset_filename = 'aid_pairs.parquet'
        val_dataset_filename = 'aid_pairs.parquet'
    elif config['model']['model_class'] == 'MatrixFactorization':
        train_dataset_filename = 'sessions_aids.parquet'
        val_dataset_filename = 'sessions_aids.parquet'
    else:
        raise ValueError('Invalid model')

    train_dataset = Dataset(path_or_source=str(dataset_root_directory / train_dataset_filename), engine='parquet')
    val_dataset = Dataset(path_or_source=str(dataset_root_directory / val_dataset_filename), engine='parquet')
    train_loader = Loader(dataset=train_dataset, batch_size=config['training']['training_batch_size'], shuffle=True, drop_last=False)
    val_loader = Loader(dataset=val_dataset, batch_size=config['training']['validation_batch_size'], shuffle=True, drop_last=False)

    # Create directory for models and visualizations
    model_root_directory = pathlib.Path(settings.MODELS / config['persistence']['model_directory'])
    model_root_directory.mkdir(parents=True, exist_ok=True)

    # Set model, loss function, device and seed for reproducible results
    torch_utils.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
    device = torch.device(config['training']['device'])
    criterion = getattr(torch.nn, config['training']['loss_function'])(**config['training']['loss_args'])

    if config['model']['model_class'] == 'CollaborativeFiltering':
        model = torch_modules.CollaborativeFiltering(
            n_embeddings=config['model']['n_embeddings'],
            n_factors=config['model']['n_factors'],
            sparse=config['model']['sparse'],
            dropout_probability=config['model']['dropout_probability']
        )
    elif config['model']['model_class'] == 'MatrixFactorization':
        model = torch_modules.MatrixFactorization(
            n_sessions=config['model']['n_sessions'],
            n_aids=config['model']['n_aids'],
            n_factors=config['model']['n_factors'],
            sparse=config['model']['sparse'],
            dropout_probability=config['model']['dropout_probability']
        )
    else:
        raise ValueError('Invalid model')

    if config['model']['model_checkpoint_path'] is not None:
        model.load_state_dict(torch.load(config['model']['model_checkpoint_path']))
    model.to(device)

    # Set optimizer, learning rate scheduler and stochastic weight averaging
    optimizer = getattr(optim, config['training']['optimizer'])(model.parameters(), **config['training']['optimizer_args'])
    scheduler = getattr(optim.lr_scheduler, config['training']['lr_scheduler'])(optimizer, **config['training']['lr_scheduler_args'])

    early_stopping = False

    if config['model']['model_class'] == 'CollaborativeFiltering':
        summary = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_roc_auc': [],
        }
    elif config['model']['model_class'] == 'MatrixFactorization':
        summary = {
            'train_loss': [],
            'val_loss': [],
            'val_mean_absolute_error': [],
            'val_mean_squared_error': [],
        }
    else:
        raise ValueError('Invalid model')

    for epoch in range(1, config['training']['epochs'] + 1):

        if early_stopping:
            break

        if config['training']['lr_scheduler'] == 'ReduceLROnPlateau':
            # Step on validation loss if learning rate scheduler is ReduceLROnPlateau
            train_loss = train(train_loader, model, criterion, optimizer, device, scheduler=None)
            val_loss, val_scores = validate(val_loader, model, criterion, device, scores=config['training']['scores'])
            scheduler.step(val_loss)
        else:
            # Learning rate scheduler works in training function if it is not ReduceLROnPlateau
            train_loss = train(train_loader, model, criterion, optimizer, device, scheduler)
            val_loss, val_scores = validate(val_loader, model, criterion, device, scores=config['training']['scores'])

        if config['model']['model_class'] == 'CollaborativeFiltering':
            logging.info(
                f'''
                Epoch {epoch} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}
                Validation Scores
                Accuracy: {val_scores["accuracy"]:.4f} ROC AUC: {val_scores["roc_auc"]:.4f}
                '''
            )
        elif config['model']['model_class'] == 'MatrixFactorization':
            logging.info(
                f'''
                Epoch {epoch} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}
                Validation Scores
                Mean Absolute Error: {val_scores["mean_absolute_error"]:.4f} Mean Squared Error: {val_scores["mean_squared_error"]:.4f}
                '''
            )
        else:
            raise ValueError('Invalid model')

        if epoch in config['persistence']['save_epoch_model']:
            # Save model if current epoch is specified to be saved
            torch.save(model.state_dict(), model_root_directory / f'model_epoch_{epoch}.pt')
            logging.info(f'Saved model_epoch_{epoch}.pt to {model_root_directory}')

        best_val_loss = np.min(summary['val_loss']) if len(summary['val_loss']) > 0 else np.inf
        if val_loss < best_val_loss:
            # Save model if validation loss improves
            torch.save(model.state_dict(), model_root_directory / f'model_best.pt')
            logging.info(f'Saved model_best.pt (validation loss decreased from {best_val_loss:.6f} to {val_loss:.6f})\n')

        summary['train_loss'].append(train_loss)
        summary['val_loss'].append(val_loss)
        if config['model']['model_class'] == 'CollaborativeFiltering':
            summary['val_accuracy'].append(val_scores['accuracy'])
            summary['val_roc_auc'].append(val_scores['roc_auc'])
        elif config['model']['model_class'] == 'MatrixFactorization':
            summary['val_mean_absolute_error'].append(val_scores['mean_absolute_error'])
            summary['val_mean_squared_error'].append(val_scores['mean_squared_error'])
        else:
            raise ValueError('Invalid model')

        best_epoch = np.argmin(summary['val_loss']) + 1
        if config['training']['early_stopping_patience'] > 0:
            # Trigger early stopping if early stopping patience is greater than 0
            if len(summary['val_loss']) - best_epoch >= config['training']['early_stopping_patience']:

                early_stopping = True

                if config['model']['model_class'] == 'CollaborativeFiltering':
                    logging.info(
                        f'''
                        Early Stopping (validation loss didn\'t improve for {config['training']["early_stopping_patience"]} epochs)
                        Best Epoch ({best_epoch + 1}) Validation Loss: {summary["val_loss"][best_epoch]:.4f}
                        Validation Scores
                        Accuracy: {summary["val_accuracy"][best_epoch]:.4f} ROC AUC: {summary["val_roc_auc"][best_epoch]:.4f}
                        '''
                    )
                    scores = {
                        'val_loss': summary['val_loss'][best_epoch],
                        'val_accuracy': summary['val_accuracy'][best_epoch],
                        'val_roc_auc': summary['val_roc_auc'][best_epoch]
                    }
                elif config['model']['model_class'] == 'MatrixFactorization':
                    logging.info(
                        f'''
                        Early Stopping (validation loss didn\'t improve for {config['training']["early_stopping_patience"]} epochs)
                        Best Epoch ({best_epoch + 1}) Validation Loss: {summary["val_loss"][best_epoch]:.4f}
                        Validation Scores
                        Mean Absolue Error: {summary["mean_absolute_error"][best_epoch]:.4f} Mean Squared Error: {summary["mean_squared_error"][best_epoch]:.4f}
                        '''
                    )
                    scores = {
                        'val_loss': summary['val_loss'][best_epoch],
                        'mean_absolute_error': summary['mean_absolute_error'][best_epoch],
                        'mean_squared_error': summary['mean_squared_error'][best_epoch]
                    }
                else:
                    raise ValueError('Invalid model')
        else:
            if epoch == config['training']['epochs']:
                if config['model']['model_class'] == 'CollaborativeFiltering':
                    scores = {
                        'val_loss': summary['val_loss'][best_epoch],
                        'val_accuracy': summary['val_accuracy'][best_epoch],
                        'val_roc_auc': summary['val_roc_auc'][best_epoch]
                    }
                elif config['model']['model_class'] == 'MatrixFactorization':
                    scores = {
                        'val_loss': summary['val_loss'][best_epoch],
                        'mean_absolute_error': summary['mean_absolute_error'][best_epoch],
                        'mean_squared_error': summary['mean_squared_error'][best_epoch]
                    }
                else:
                    raise ValueError('Invalid model')

    if config['persistence']['visualize_learning_curve']:

        if config['model']['model_class'] == 'CollaborativeFiltering':
            visualization_validation_scores = {
                'val_accuracy': summary['val_accuracy'],
                'val_roc_auc': summary['val_roc_auc']
            }
        elif config['model']['model_class'] == 'MatrixFactorization':
            visualization_validation_scores = {
                'val_mean_absolute_error': summary['mean_absolute_error'],
                'val_mean_squared_error': summary['mean_squared_error']
            }
        else:
            raise ValueError('Invalid model')

        visualization.visualize_learning_curve(
            training_losses=summary['train_loss'],
            validation_losses=summary['val_loss'],
            validation_scores=visualization_validation_scores,
            path=str(model_root_directory / f'learning_curve.png')
        )
        logging.info(f'Saved learning_curve.png to {model_root_directory}')
