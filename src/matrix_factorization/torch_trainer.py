import sys
import logging
import pathlib
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
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

        optimizer.zero_grad()

        if isinstance(model, torch_modules.CollaborativeFiltering):
            # Pass session or aid pairs to collaborative filtering model
            x1, x2, targets = inputs['x1'].to(device), inputs['x2'].to(device), inputs['target'].to_device()
            outputs = model(x1, x2)
            loss = criterion(outputs, targets)
        elif isinstance(model, torch_modules.MatrixFactorization):
            # Pass session and aid to matrix factorization model
            session, aid, type = inputs['session'].to(device), inputs['aid'].to(device), inputs['type'].to(device)
            outputs = model(session, aid)

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


def validate(val_loader, model, criterion, device):

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

    Returns
    -------
    val_loss: float
        Average validation loss after model is fully validated on validation set data loader

    val_scores: dict
        Dictionary of metric scores after model is fully validated on validation set data loader
    """

    model.eval()
    progress_bar = tqdm(val_loader)
    losses = []
    ground_truth = []
    predictions = []

    with torch.no_grad():
        for inputs, _ in progress_bar:

            session, aid = inputs['session'].to(device), inputs['aid'].to(device)
            positive_outputs = model(session, aid)
            negative_outputs = model(session, aid[torch.randperm(aid.shape[0])])
            outputs = torch.cat([positive_outputs, negative_outputs])
            targets = torch.cat([torch.ones_like(positive_outputs), torch.zeros_like(negative_outputs)])

            loss = criterion(outputs, targets)
            losses.append(loss.detach().item())
            average_loss = np.mean(losses)
            progress_bar.set_description(f'val_loss: {average_loss:.6f}')
            ground_truth += [(targets.detach().cpu())]
            predictions += [(outputs.detach().cpu())]

    val_loss = np.mean(losses)
    ground_truth = torch.cat(ground_truth, dim=0).numpy()
    predictions = torch.sigmoid(torch.cat(predictions, dim=0)).numpy()
    val_scores = metrics.classification_scores(y_true=ground_truth, y_pred=predictions, threshold=0.5)

    return val_loss, val_scores


if __name__ == '__main__':

    config = yaml.load(open(settings.MODELS / 'aid_collaborative_filtering' / 'config.yaml', 'r'), Loader=yaml.FullLoader)

    df = pd.concat((
        pd.read_pickle(settings.DATA / 'train.pkl'),
        pd.read_pickle(settings.DATA / 'test.pkl')
    ), axis=0, ignore_index=True).loc[:500000]
    logging.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    if config['model']['model_class'] == 'CollaborativeFiltering':

        # Create directory for datasets
        dataset_root_directory = pathlib.Path(settings.DATA / 'collaborative_filtering')
        dataset_root_directory.mkdir(parents=True, exist_ok=True)

        if config['dataset']['pairs'] == 'aid':

            if config['dataset']['load_dataset']:
                logging.info(f'Using pre-computed dataset from {dataset_root_directory / "aid_pairs.parquet"}')
                df_aid_pairs = pd.read_parquet(dataset_root_directory / "aid_pairs.parquet")
            else:
                # Create a dataset for aid x aid collaborative filtering model
                df.index = pd.MultiIndex.from_frame(df[['session']])
                df_aid_pairs = pd.DataFrame(columns=['aid_x', 'aid_y', 'target'], dtype=int)
                sessions = df['session'].unique()
                logging.info(f'Creating aid pairs dataset from {len(sessions)} sessions and {df.shape[0]} events')

                for i in tqdm(range(0, sessions.shape[0], config['dataset']['chunk_size'])):
                    # Index a chunk of sessions
                    df_sessions = df.loc[sessions[i]:sessions[min(sessions.shape[0] - 1, i + config['dataset']['chunk_size'] - 1)]].reset_index(drop=True)
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

                df_aid_pairs['aid_x'] = df_aid_pairs['aid_x'].astype(np.uint32)
                df_aid_pairs['aid_y'] = df_aid_pairs['aid_y'].astype(np.uint32)
                df_aid_pairs['target'] = df_aid_pairs['target'].astype(np.uint8)

                if config['dataset']['target_aggregation'] == 'mean':
                    # Assign positive label to aid pairs if their target mean is greater than 0.5
                    df_aid_pairs = df_aid_pairs.groupby(['aid_x', 'aid_y'])['target'].mean().reset_index()
                    df_aid_pairs['target'] = (df_aid_pairs['target'] >= 0.5).astype(np.uint8)
                elif config['dataset']['target_aggregation'] == 'max':
                    # Assign positive label to aid pairs if their target is positive at least once
                    df_aid_pairs = df_aid_pairs.groupby(['aid_x', 'aid_y'])['target'].max().reset_index()
                else:
                    raise ValueError('Invalid target aggregation')

                df_aid_pairs.rename(columns={'aid_x': 'x1', 'aid_y': 'x2'}, inplace=True)
                df_aid_pairs.to_parquet(dataset_root_directory / 'aid_pairs.parquet')
                logging.info(f'aid_pairs.parquet is saved to {dataset_root_directory}')

            del df
            n_pairs = df_aid_pairs.shape[0]
            n_aids = max(df_aid_pairs['x1'].max(), df_aid_pairs['x2'].max())

            logging.info(
                f'''
                aid pairs dataset - pairs: {n_pairs} aids: {n_aids})
                {np.sum(df_aid_pairs['target'] == 1)} ({((np.sum(df_aid_pairs['target'] == 1) / n_pairs) * 100):.2f}%) positive aid pairs (hour difference <= {config['dataset']['hour_difference']})
                {np.sum(df_aid_pairs['target'] == 0)} ({((np.sum(df_aid_pairs['target'] == 0) / n_pairs) * 100):.2f}%) negative aid pairs (hour difference > {config['dataset']['hour_difference']})
                '''
            )

        elif config['dataset']['pairs'] == 'session':

            if config['dataset']['load_dataset']:
                logging.info(f'Using pre-computed dataset from {dataset_root_directory / "session_pairs.parquet"}')
                df_session_pairs = pd.read_parquet(dataset_root_directory / "session_pairs.parquet")
            else:
                # Create a dataset for session x session collaborative filtering model
                # ?????????????????????????????
                # sessions with aid intersection > 0.5 are positive?
                # sessions with events at similar time are positive?
                # sessions with similar number of events are similar?
                pass

        else:
            raise ValueError('Invalid pairs')

    elif config['model']['model_class'] == 'MatrixFactorization':
        pass

    else:
        raise ValueError('Invalid model')
    exit()

    # Create training and validation datasets and data loaders
    train_dataset = Dataset(path_or_source=str(dataset_root_directory / 'all.parquet'), engine='parquet')
    val_dataset = Dataset(path_or_source=str(dataset_root_directory / 'all.parquet'), engine='parquet')
    train_loader = Loader(dataset=train_dataset, batch_size=config['training']['training_batch_size'], shuffle=True, drop_last=False)
    val_loader = Loader(dataset=val_dataset, batch_size=config['training']['validation_batch_size'], shuffle=True, drop_last=False)

    # Create directory for models and visualizations
    model_root_directory = pathlib.Path(settings.MODELS / 'matrix_factorization')
    model_root_directory.mkdir(parents=True, exist_ok=True)

    # Set model, loss function, device and seed for reproducible results
    torch_utils.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
    device = torch.device(config['training']['device'])
    criterion = getattr(torch.nn, config['training']['loss_function'])(**config['training']['loss_args'])

    model = torch_modules.MatrixFactorization(
        n_sessions=n_sessions,
        n_aids=n_aids,
        embedding_dim=config['model']['embedding_dim']
    )
    if config['model']['model_checkpoint_path'] is not None:
        model.load_state_dict(torch.load(config['model']['model_checkpoint_path']))
    model.to(device)

    # Set optimizer, learning rate scheduler and stochastic weight averaging
    optimizer = getattr(optim, config['training']['optimizer'])(model.parameters(), **config['training']['optimizer_args'])
    scheduler = getattr(optim.lr_scheduler, config['training']['lr_scheduler'])(optimizer, **config['training']['lr_scheduler_args'])

    early_stopping = False
    summary = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_roc_auc': [],
    }

    for epoch in range(1, config['training']['epochs'] + 1):

        if early_stopping:
            break

        if config['training']['lr_scheduler'] == 'ReduceLROnPlateau':
            # Step on validation loss if learning rate scheduler is ReduceLROnPlateau
            train_loss = train(train_loader, model, criterion, optimizer, device, scheduler=None)
            val_loss, val_scores = validate(val_loader, model, criterion, device)
            scheduler.step(val_loss)
        else:
            # Learning rate scheduler works in training function if it is not ReduceLROnPlateau
            train_loss = train(train_loader, model, criterion, optimizer, device, scheduler)
            val_loss, val_scores = validate(val_loader, model, criterion, device)

        logging.info(
            f'''
            Epoch {epoch} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}
            Validation Scores
            Accuracy: {val_scores["accuracy"]:.4f} ROC AUC: {val_scores["roc_auc"]:.4f}
            '''
        )

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
        summary['val_accuracy'].append(val_scores['accuracy'])
        summary['val_roc_auc'].append(val_scores['roc_auc'])

        best_epoch = np.argmin(summary['val_loss']) + 1
        if config['training']['early_stopping_patience'] > 0:
            # Trigger early stopping if early stopping patience is greater than 0
            if len(summary['val_loss']) - best_epoch >= config['training']['early_stopping_patience']:
                logging.info(
                    f'''
                    Early Stopping (validation loss didn\'t improve for {config['training']["early_stopping_patience"]} epochs)
                    Best Epoch ({best_epoch + 1}) Validation Loss: {summary["val_loss"][best_epoch]:.4f}
                    Validation Scores
                    Accuracy: {summary["val_accuracy"][best_epoch]:.4f} ROC AUC: {summary["val_roc_auc"][best_epoch]:.4f}
                    '''
                )
                early_stopping = True
                scores = {
                    'val_loss': summary['val_loss'][best_epoch],
                    'val_accuracy': summary['val_accuracy'][best_epoch],
                    'val_roc_auc': summary['val_roc_auc'][best_epoch]
                }
        else:
            if epoch == config['training']['epochs']:
                scores = {
                    'val_loss': summary['val_loss'][best_epoch],
                    'val_accuracy': summary['val_accuracy'][best_epoch],
                    'val_roc_auc': summary['val_roc_auc'][best_epoch]
                }

    if config['persistence']['visualize_learning_curve']:
        visualization.visualize_learning_curve(
            training_losses=summary['train_loss'],
            validation_losses=summary['val_loss'],
            validation_scores={
                'val_accuracy': summary['val_accuracy'],
                'val_roc_auc': summary['val_roc_auc']
            },
            path=model_root_directory / f'learning_curve.png'
        )
        logging.info(f'Saved learning_curve.png to {model_root_directory}')
