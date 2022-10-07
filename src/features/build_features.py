# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, 'D:\git_repos\lab_data_science')

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import src.config as cfg
from src.utils import save_as_pickle, load_as_pickle
from src.config import (path_to_splitted_train_data,
                path_to_splitted_train_data_target,
                path_to_splitted_val_data,
                path_to_splitted_val_data_target)
from utils import TargetEncoder, empty_column_filler
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('output_dir', type=click.Path())
def main(output_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Building features')
    
    x_train = load_as_pickle(path_to_splitted_train_data)
    y_train = load_as_pickle(path_to_splitted_train_data_target)
    x_val = load_as_pickle(path_to_splitted_val_data)
    y_val = load_as_pickle(path_to_splitted_val_data_target)
    
    ecf = empty_column_filler()
    te = TargetEncoder(cfg.CAT_COLS, 5)
    
    x_train = ecf.fit_transform(x_train)
    x_val = ecf.fit_transform(x_val)
    
    # x_train_t_encoded = te.fit(x_train, y_train)
    # x_val_t_encoded = te.transform(x_val)
    
    save_as_pickle(x_train, output_dir + '/featured_train_data.pkl')
    save_as_pickle(x_val, output_dir + '/featured_val_data.pkl')
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
