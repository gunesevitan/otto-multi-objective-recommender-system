import sys
import logging
import argparse
import pathlib
import yaml
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import *
from recbole.model.sequential_recommender import *
from recbole.trainer import Trainer

sys.path.append('..')
import settings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    config = yaml.load(open(settings.MODELS / args.config_path, 'r'), Loader=yaml.FullLoader)

    # Create directory for models
    model_root_directory = pathlib.Path(settings.MODELS / config['persistence']['model_directory'])
    model_root_directory.mkdir(parents=True, exist_ok=True)

    recbole_config = Config(
        model=config['model']['model_name'],
        dataset='data',
        config_dict=config['model']['model_args']
    )
    logging.info(recbole_config)
    dataset = create_dataset(recbole_config)
    logging.info(dataset)

    training_data, validation_data, test_data = data_preparation(recbole_config, dataset)
    model = eval(config['model']['model_name'])(recbole_config, training_data.dataset).to(recbole_config['device'])
    trainer = Trainer(recbole_config, model)
    best_valid_score, best_valid_result = trainer.fit(training_data, validation_data)
    logging.info(
        f'''
        Finished training {config["model"]["model_name"]} model
        Best valid score: {best_valid_score:.4f}
        Best valid results: {best_valid_result}
        '''
    )
