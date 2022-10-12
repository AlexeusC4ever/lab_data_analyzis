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
@click.argument('input_dir', type=click.Path())
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir='src/models/'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Building features')
    
    x_train = load_as_pickle(input_dir + cfg.name_of_featured_train_data)
    x_train_t_encoded = load_as_pickle(input_dir + cfg.name_of_featured_t_encoded_train_data)
    y_train = load_as_pickle(path_to_splitted_train_data_target)
    x_val = load_as_pickle(input_dir + cfg.name_of_featured_val_data)
    x_val_t_encoded = load_as_pickle(input_dir + cfg.name_of_featured_t_encoded_val_data)
    y_val = load_as_pickle(path_to_splitted_val_data_target)
    
    
    
    cat_idx = get_indexes_of_cat_columns(x_train, cfg.CAT_COLS)
    
    # catboost_model = catboost.CatBoostClassifier(
                                                # loss_function='MultiLogloss',
                                                # cat_features=cat_idx,
                                                # verbose=50,
                                                # iterations=1000
                                           # )
                                            
    # catboost_model.fit(x_train, y_train)           

    # LrCl_target_encode = MultiOutputClassifier(LogisticRegression(max_iter=10000)).fit(x_train_t_encoded, y_train)
    
    
    max_depth_values = [100, 500, 1000]

    best_score = 0
    best_model = None
    
    param_grid = {
        'model_size_reg': [0, 1],
        'n_estimators' : [100, 500, 1000],
        'one_hot_max_size' : [0, 1000],
        'max_ctr_complexity': [1, 6]
    }

    grid = GridSearchCV(catboost.CatBoostClassifier(cat_features=cat_idx, loss_function='MultiLogloss', verbose=500), param_grid, cv=3, scoring='recall', 
                        verbose=0)

    grid.fit(x_train, y_train)
    
    catboost_model = catboost.CatBoostClassifier(cat_features=cat_idx, loss_function='MultiLogloss', verbose=100, **grid.best_params_)
    catboost_model.fit(x_train, y_train)

    for i in max_depth_values:
        rfC = MultiOutputClassifier(RandomForestClassifier(max_depth = i))

        rfC.fit(x_train_t_encoded, y_train)
        y_pred = rfC.predict(x_val_t_encoded)
        score = recall_score(y_val, y_pred, average='micro')
        if score > best_score:
            best_score = score
            best_model = rfC
    
    
    save_model_as_pickle(catboost_model, output_dir + cfg.name_catboost_model)
    save_model_as_pickle(best_model, output_dir + cfg.name_lr_model)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
