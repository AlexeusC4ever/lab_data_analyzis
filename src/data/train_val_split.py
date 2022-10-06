# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, 'D:\git_repos\lab_data_science')

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import src.config as cfg
from src.utils import save_as_pickle, load_as_pickle
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath_data', type=click.Path(exists=True))
@click.argument('input_filepath_target', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(input_filepath_data, input_filepath_target, output_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Splitting data to train - val samples')
    
    train_data, target_data = load_as_pickle(input_filepath_data), load_as_pickle(input_filepath_target)
    x_train, x_test, y_train, y_test = train_test_split(train_data, target_data, test_size=0.25)
    
    
    path_to_splitted_train_data = output_dir + '/train.pkl'
    path_to_splitted_train_data_target = output_dir + '/train_target.pkl'
    path_to_splitted_val_data = output_dir + '/val.pkl'
    path_to_splitted_val_data_target = output_dir + '/val_target.pkl'
    
    save_as_pickle(x_train, path_to_splitted_train_data)
    save_as_pickle(y_train, path_to_splitted_train_data_target)
    save_as_pickle(x_test, path_to_splitted_val_data)
    save_as_pickle(y_test, path_to_splitted_val_data_target)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
