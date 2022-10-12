# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, 'D:\git_repos\lab_data_science')

import click
import logging
import catboost
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, precision_score, recall_score, roc_auc_score, f1_score

from src.features.utils import empty_column_filler
import src.config as cfg
from src.utils import save_as_pickle, load_as_pickle, save_model_as_pickle, load_model_as_pickle
from src.config import (path_to_splitted_train_data,
                path_to_splitted_train_data_target,
                path_to_splitted_val_data,
                path_to_splitted_val_data_target)
from utils import get_indexes_of_cat_columns
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('test_data_file_path', type=click.Path())
@click.argument('output_file_path', type=click.Path())
@click.argument('model_path', type=click.Path())
def main(test_data_file_path, output_file_path, model_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Building features')
    
    x_test = pd.read_csv(test_data_file_path)
    model = load_model_as_pickle(model_path)
    
    x_test = x_test.drop('ID', axis=1)
    ecf = empty_column_filler()
    
    x_test = ecf.transform(x_test)
    
    prediction = model.predict(x_test)
    prediction = pd.DataFrame(prediction, columns=cfg.TARGET_COLS)
    
    prediction.to_csv(output_file_path)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
