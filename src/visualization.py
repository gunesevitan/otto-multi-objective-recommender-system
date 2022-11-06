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


if __name__ == '__main__':

    df_train = pd.read_pickle(settings.DATA / 'train.pkl')
    df_test = pd.read_pickle(settings.DATA / 'test.pkl')

    logging.info(f'Training Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    VISUALIZE_TYPE_DISTRIBUTION = False
    VISUALIZE_SESSION_COUNT_DISTRIBUTION = False
    VISUALIZE_AID_COUNT_DISTRIBUTION = False

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
