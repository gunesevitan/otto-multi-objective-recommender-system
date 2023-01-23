import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import settings


def visualize_categorical_feature_distribution(df, feature, path=None):

    """
    Visualize distribution of given categorical column in given dataframe

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples, 4)
        Dataframe with session, aid, ts, type columns

    feature: str
        Name of the categorical feature

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, df[feature].value_counts().shape[0] + 4), dpi=100)
    sns.barplot(
        y=df[feature].value_counts().values,
        x=df[feature].value_counts().index,
        color='tab:blue',
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([
        f'{x} ({value_count:,})' for value_count, x in zip(
            df[feature].value_counts().values,
            df[feature].value_counts().index
        )
    ])
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(f'Value Counts {feature}', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_continuous_feature_distribution(df_train, df_test, feature, path=None):

    """
    Visualize distribution of given continuous column in given dataframe

    Parameters
    ----------
    df_train: pandas.DataFrame of shape (n_samples, 4)
        Training dataframe with session, aid, ts, type columns

    df_test: pandas.DataFrame of shape (n_samples, 4)
        Test dataframe with session, aid, ts, type columns

    feature: str
        Name of the continuous feature

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, 6), dpi=100)
    sns.kdeplot(df_train[feature], label='train', fill=True, log_scale=True, ax=ax)
    sns.kdeplot(df_test[feature], label='test', fill=True, log_scale=True, ax=ax)
    ax.tick_params(axis='x', labelsize=12.5)
    ax.tick_params(axis='y', labelsize=12.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(prop={'size': 15})
    title = f'''
    {feature}
    Mean - Train: {df_train[feature].mean():.2f} |  Test: {df_test[feature].mean():.2f}
    Median - Train: {df_train[feature].median():.2f} |  Test: {df_test[feature].median():.2f}
    Std - Train: {df_train[feature].std():.2f} |  Test: {df_test[feature].std():.2f}
    Min - Train: {df_train[feature].min():.2f} |  Test: {df_test[feature].min():.2f}
    Max - Train: {df_train[feature].max():.2f} |  Test: {df_test[feature].max():.2f}
    '''
    ax.set_title(title, size=20, pad=12.5)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


def visualize_session(df, session, path=None):

    """
    Visualize given session in given dataframe as a sequence

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples, 4)
        Training or test dataframe with session, aid, ts, type columns

    session: int
        Unique ID of the session

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    df_session = df.loc[df['session'] == session, :]

    fig, ax = plt.subplots(figsize=(24, 6))
    ax.plot(df_session.set_index('ts')['type'], 'o-')
    ax.set_yticks(range(3), ['Click (0)', 'Cart (1)', 'Order (2)'])
    ax.tick_params(axis='x', labelsize=12.5)
    ax.tick_params(axis='y', labelsize=12.5)
    ax.set_xlabel('Timestamps', fontsize=15, labelpad=12.5)
    ax.set_ylabel('Events', fontsize=15, labelpad=12.5)
    title = f'''
    Session: {session}
    Clicks: {(df_session['type'] == 0).sum()} |  Carts: {(df_session['type'] == 1).sum()} | Orders: {(df_session['type'] == 2).sum()}
    '''
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_aid_frequencies(aid_frequencies, title, path=None):

    """
    Visualize aids and their frequencies in given dictionary

    Parameters
    ----------
    aid_frequencies: dict
        Dictionary of aids and their frequencies

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, 20), dpi=100)
    ax.barh(range(len(aid_frequencies)), aid_frequencies.values(), align='center')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks(range(len(aid_frequencies)))
    ax.set_yticklabels([f'{x} ({value_count:,})' for x, value_count in aid_frequencies.items()])
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)
    plt.gca().invert_yaxis()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_feature_importance(df_feature_importance, path=None):

    """
    Visualize feature importance in descending order

    Parameters
    ----------
    df_feature_importance: pandas.DataFrame of shape (n_features, n_splits)
        Dataframe of feature importance

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    n_splits = df_feature_importance.shape[1] - 2

    fig, ax = plt.subplots(figsize=(24, 20), dpi=100)
    ax.barh(
        range(len(df_feature_importance)),
        df_feature_importance['mean'],
        xerr=df_feature_importance['std'],
        ecolor='black',
        capsize=10,
        align='center',
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks(range(len(df_feature_importance)))
    ax.set_yticklabels([f'{k} ({v:.2f})' for k, v in df_feature_importance['mean'].to_dict().items()])
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(f'Mean and Std Feature Importance of {n_splits} Models', size=20, pad=15)
    plt.gca().invert_yaxis()

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


def visualize_predictions(train_predictions, test_predictions, path=None):

    """
    Visualize train and test predictions histograms

    Parameters
    ----------
    train_predictions: numpy.ndarray of shape (n_predictions)
        Array of train predictions

    test_predictions: numpy.ndarray of shape (n_predictions)
        Array of test predictions

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, 8), dpi=100)
    ax.hist(train_predictions, label='train', alpha=0.5,  bins=50)
    ax.hist(test_predictions, label='test', alpha=0.5, bins=50)
    ax.tick_params(axis='x', labelsize=12.5)
    ax.tick_params(axis='y', labelsize=12.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(prop={'size': 15})
    title = f'''
        Predictions
        Mean - Train: {train_predictions.mean():.2f} |  Test: {test_predictions.mean():.2f}
        Std - Train: {train_predictions.std():.2f} |  Test: {test_predictions.std():.2f}
        Min - Train: {train_predictions.min():.2f} |  Test: {test_predictions.min():.2f}
        Max - Train: {train_predictions.max():.2f} |  Test: {test_predictions.max():.2f}
        '''
    ax.set_title(title, size=20, pad=12.5)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':

    df_train = pd.read_pickle(settings.DATA / 'train.pkl')
    df_test = pd.read_pickle(settings.DATA / 'test.pkl')

    logging.info(f'Training Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    VISUALIZE_TYPE_DISTRIBUTION = False
    VISUALIZE_SESSION_COUNT_DISTRIBUTION = False
    VISUALIZE_AID_COUNT_DISTRIBUTION = False
    VISUALIZE_AID_FREQUENCIES = True

    if VISUALIZE_TYPE_DISTRIBUTION:

        visualize_categorical_feature_distribution(
            df=df_train,
            feature='type',
            path=settings.EDA / 'train_type_distribution.png'
        )
        visualize_categorical_feature_distribution(
            df=df_test,
            feature='type',
            path=settings.EDA / 'test_type_distribution.png'
        )
        logging.info(f'Saved type distribution visualization to {settings.EDA}')

    if VISUALIZE_SESSION_COUNT_DISTRIBUTION:

        df_train_sessions = df_train.groupby('session')[['session']].count().reset_index(drop=True)
        df_test_sessions = df_test.groupby('session')[['session']].count().reset_index(drop=True)
        visualize_continuous_feature_distribution(
            df_train=df_train_sessions,
            df_test=df_test_sessions,
            feature='session',
            path=settings.EDA / 'session_count_distribution.png'
        )
        logging.info(f'Saved session count distribution visualization to {settings.EDA}')

    if VISUALIZE_AID_COUNT_DISTRIBUTION:

        df_train_aids = df_train.groupby('aid')[['aid']].count().reset_index(drop=True)
        df_test_aids = df_test.groupby('aid')[['aid']].count().reset_index(drop=True)
        visualize_continuous_feature_distribution(
            df_train=df_train_aids,
            df_test=df_test_aids,
            feature='aid',
            path=settings.EDA / 'aid_count_distribution.png'
        )
        logging.info(f'Saved aid count distribution visualization to {settings.EDA}')

    if VISUALIZE_AID_FREQUENCIES:

        for dataset in ['train', 'test', 'all']:

            with open(settings.DATA / 'statistics' / f'{dataset}_20_most_frequent_aids.json') as f:
                aid_frequencies = json.load(f)

            visualize_aid_frequencies(
                aid_frequencies=aid_frequencies,
                title=f'Top 20 Most Frequent aids in {dataset.capitalize()}',
                path=settings.EDA / f'{dataset}_top_20_most_frequent_aids.png'
            )
            logging.info(f'Saved {dataset}_top_20_most_frequent_aids.png to {settings.EDA}')

            for event_type in ['click', 'cart', 'order']:

                with open(settings.DATA / 'statistics' / f'{dataset}_20_most_frequent_{event_type}_aids.json') as f:
                    aid_frequencies = json.load(f)

                visualize_aid_frequencies(
                    aid_frequencies=aid_frequencies,
                    title=f'Top 20 Most Frequent {event_type.capitalize()} aids in {dataset.capitalize()}',
                    path=settings.EDA / f'{dataset}_top_20_most_frequent_{event_type}_aids.png'
                )
                logging.info(f'Saved {dataset}_top_20_most_frequent_{event_type}_aids.png to {settings.EDA}')
