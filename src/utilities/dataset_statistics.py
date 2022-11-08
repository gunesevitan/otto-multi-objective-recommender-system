import sys
import logging
import pathlib
import json
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    df_train = pd.read_pickle(settings.DATA / 'train.pkl')
    df_test = pd.read_pickle(settings.DATA / 'test.pkl')

    logging.info(f'Training Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Create a directory for saving extracted statistics
    statistics_directory = pathlib.Path(settings.DATA / 'statistics')
    statistics_directory.mkdir(parents=True, exist_ok=True)

    # Extract top 20 most frequent aids globally
    df_train_aid_counts = df_train.groupby('aid')[['aid']].count()
    df_test_aid_counts = df_test.groupby('aid')[['aid']].count()

    df_train_aid_counts = df_train_aid_counts.rename(columns={'aid': 'count'}).reset_index()
    df_test_aid_counts = df_test_aid_counts.rename(columns={'aid': 'count'}).reset_index()
    df_all_aid_counts = pd.concat((df_train_aid_counts, df_test_aid_counts), axis=0, ignore_index=True).reset_index(drop=True)
    df_all_aid_counts = df_all_aid_counts.groupby('aid')['count'].sum().reset_index()

    df_train_aid_counts.sort_values(by='count', ascending=False, inplace=True)
    df_test_aid_counts.sort_values(by='count', ascending=False, inplace=True)
    df_all_aid_counts.sort_values(by='count', ascending=False, inplace=True)

    # Convert top 20 most frequent aids to dictionaries and save them
    train_20_most_frequent_aids = df_train_aid_counts.set_index('aid').head(20).to_dict()['count']
    test_20_most_frequent_aids = df_test_aid_counts.set_index('aid').head(20).to_dict()['count']
    all_20_most_frequent_aids = df_all_aid_counts.set_index('aid').head(20).to_dict()['count']
    del df_train_aid_counts, df_test_aid_counts, df_all_aid_counts

    with open(str(statistics_directory / 'train_20_most_frequent_aids.json'), 'w') as f:
        json.dump(train_20_most_frequent_aids, f, indent=2)
    logging.info(f'Saved train_20_most_frequent_aids.json to {statistics_directory}')

    with open(str(statistics_directory / 'test_20_most_frequent_aids.json'), 'w') as f:
        json.dump(test_20_most_frequent_aids, f, indent=2)
    logging.info(f'Saved test_20_most_frequent_aids.json to {statistics_directory}')

    with open(str(statistics_directory / 'all_20_most_frequent_aids.json'), 'w') as f:
        json.dump(all_20_most_frequent_aids, f, indent=2)
    logging.info(f'Saved all_20_most_frequent_aids.json to {statistics_directory}')

    # Extract top 20 most frequent aids for every event type
    df_train_aids_by_types = df_train.groupby(['type', 'aid'])[['aid']].count()
    df_test_aids_by_types = df_test.groupby(['type', 'aid'])[['aid']].count()
    df_train_click_aid_counts = df_train_aids_by_types.loc[0].rename(columns={'aid': 'count'}).reset_index()
    df_train_cart_aid_counts = df_train_aids_by_types.loc[1].rename(columns={'aid': 'count'}).reset_index()
    df_train_order_aid_counts = df_train_aids_by_types.loc[2].rename(columns={'aid': 'count'}).reset_index()
    df_test_click_aid_counts = df_test_aids_by_types.loc[0].rename(columns={'aid': 'count'}).reset_index()
    df_test_cart_aid_counts = df_test_aids_by_types.loc[1].rename(columns={'aid': 'count'}).reset_index()
    df_test_order_aid_counts = df_test_aids_by_types.loc[2].rename(columns={'aid': 'count'}).reset_index()
    del df_train_aids_by_types, df_test_aids_by_types
    df_all_click_aid_counts = pd.concat((df_train_click_aid_counts, df_test_click_aid_counts), axis=0, ignore_index=True).reset_index(drop=True)
    df_all_click_aid_counts = df_all_click_aid_counts.groupby('aid')['count'].sum().reset_index()
    df_all_cart_aid_counts = pd.concat((df_train_cart_aid_counts, df_test_cart_aid_counts), axis=0, ignore_index=True).reset_index(drop=True)
    df_all_cart_aid_counts = df_all_cart_aid_counts.groupby('aid')['count'].sum().reset_index()
    df_all_order_aid_counts = pd.concat((df_train_order_aid_counts, df_test_order_aid_counts), axis=0, ignore_index=True).reset_index(drop=True)
    df_all_order_aid_counts = df_all_order_aid_counts.groupby('aid')['count'].sum().reset_index()

    df_train_click_aid_counts.sort_values(by='count', ascending=False, inplace=True)
    df_test_click_aid_counts.sort_values(by='count', ascending=False, inplace=True)
    df_all_click_aid_counts.sort_values(by='count', ascending=False, inplace=True)

    df_train_cart_aid_counts.sort_values(by='count', ascending=False, inplace=True)
    df_test_cart_aid_counts.sort_values(by='count', ascending=False, inplace=True)
    df_all_cart_aid_counts.sort_values(by='count', ascending=False, inplace=True)

    df_train_order_aid_counts.sort_values(by='count', ascending=False, inplace=True)
    df_test_order_aid_counts.sort_values(by='count', ascending=False, inplace=True)
    df_all_order_aid_counts.sort_values(by='count', ascending=False, inplace=True)

    # Convert top 20 most frequent aids to dictionaries and save them
    train_20_most_frequent_click_aids = df_train_click_aid_counts.set_index('aid').head(20).to_dict()['count']
    test_20_most_frequent_click_aids = df_test_click_aid_counts.set_index('aid').head(20).to_dict()['count']
    all_20_most_frequent_click_aids = df_all_click_aid_counts.set_index('aid').head(20).to_dict()['count']
    del df_train_click_aid_counts, df_test_click_aid_counts, df_all_click_aid_counts

    train_20_most_frequent_cart_aids = df_train_cart_aid_counts.set_index('aid').head(20).to_dict()['count']
    test_20_most_frequent_cart_aids = df_test_cart_aid_counts.set_index('aid').head(20).to_dict()['count']
    all_20_most_frequent_cart_aids = df_all_cart_aid_counts.set_index('aid').head(20).to_dict()['count']
    del df_train_cart_aid_counts, df_test_cart_aid_counts, df_all_cart_aid_counts

    train_20_most_frequent_order_aids = df_train_order_aid_counts.set_index('aid').head(20).to_dict()['count']
    test_20_most_frequent_order_aids = df_test_order_aid_counts.set_index('aid').head(20).to_dict()['count']
    all_20_most_frequent_order_aids = df_all_order_aid_counts.set_index('aid').head(20).to_dict()['count']
    del df_train_order_aid_counts, df_test_order_aid_counts, df_all_order_aid_counts

    with open(str(statistics_directory / 'train_20_most_frequent_click_aids.json'), 'w') as f:
        json.dump(train_20_most_frequent_click_aids, f, indent=2)
    logging.info(f'Saved train_20_most_frequent_click_aids.json to {statistics_directory}')

    with open(str(statistics_directory / 'test_20_most_frequent_click_aids.json'), 'w') as f:
        json.dump(test_20_most_frequent_click_aids, f, indent=2)
    logging.info(f'Saved test_20_most_frequent_click_aids.json to {statistics_directory}')

    with open(str(statistics_directory / 'all_20_most_frequent_click_aids.json'), 'w') as f:
        json.dump(all_20_most_frequent_click_aids, f, indent=2)
    logging.info(f'Saved all_20_most_frequent_click_aids.json to {statistics_directory}')

    with open(str(statistics_directory / 'train_20_most_frequent_cart_aids.json'), 'w') as f:
        json.dump(train_20_most_frequent_cart_aids, f, indent=2)
    logging.info(f'Saved train_20_most_frequent_cart_aids.json to {statistics_directory}')

    with open(str(statistics_directory / 'test_20_most_frequent_cart_aids.json'), 'w') as f:
        json.dump(test_20_most_frequent_cart_aids, f, indent=2)
    logging.info(f'Saved test_20_most_frequent_cart_aids.json to {statistics_directory}')

    with open(str(statistics_directory / 'all_20_most_frequent_cart_aids.json'), 'w') as f:
        json.dump(all_20_most_frequent_cart_aids, f, indent=2)
    logging.info(f'Saved all_20_most_frequent_cart_aids.json to {statistics_directory}')

    with open(str(statistics_directory / 'train_20_most_frequent_order_aids.json'), 'w') as f:
        json.dump(train_20_most_frequent_order_aids, f, indent=2)
    logging.info(f'Saved train_20_most_frequent_order_aids.json to {statistics_directory}')

    with open(str(statistics_directory / 'test_20_most_frequent_order_aids.json'), 'w') as f:
        json.dump(test_20_most_frequent_order_aids, f, indent=2)
    logging.info(f'Saved test_20_most_frequent_order_aids.json to {statistics_directory}')

    with open(str(statistics_directory / 'all_20_most_frequent_order_aids.json'), 'w') as f:
        json.dump(all_20_most_frequent_order_aids, f, indent=2)
    logging.info(f'Saved all_20_most_frequent_order_aids.json to {statistics_directory}')
