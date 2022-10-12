# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, 'D:\git_repos\lab_data_science')

import click
import logging
import catboost
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

import src.config as cfg
from src.utils import save_as_pickle, load_as_pickle, save_model_as_pickle, load_model_as_pickle
import src.features.utils as feat_utils
from src.config import (path_to_splitted_train_data,
                        path_to_splitted_train_data_target)
import pandas as pd


@click.command()
@click.argument('input_dir', type=click.Path())
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir='src/models/'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating Pipelines')
    
    x_train = load_as_pickle(path_to_splitted_train_data)
    y_train = load_as_pickle(path_to_splitted_train_data_target)
    
    catboost_model = load_model_as_pickle(input_dir + cfg.name_catboost_model)
    RFC = load_model_as_pickle(input_dir + cfg.name_lr_model)
    
    catboost_pipeline = Pipeline([('column_filler', feat_utils.empty_column_filler()),
                                ('model', catboost_model)])
                     
    RFC_pipeline = Pipeline([('column_filler', feat_utils.empty_column_filler()),
                                ('target_encoder', feat_utils.TargetEncoder(cfg.CAT_COLS, 5)),
                                ('model', RFC)])
                 
    # pipeline.predict(x_train)
    # catboost_pipeline.fit(x_train, y_train)
    ecf = feat_utils.empty_column_filler()
    te = feat_utils.TargetEncoder(cfg.CAT_COLS, 5)
    x_train_t_encoded = te.fit_transform(ecf.fit_transform(x_train), y_train)
    
    RFC_pipeline.fit(x_train, y_train)
    RFC_pipeline['model'].estimator.fit(x_train_t_encoded, y_train)
    catboost_pipeline.fit(x_train, y_train)
    
    save_model_as_pickle(catboost_pipeline, output_dir + cfg.name_catboost_pipeline)
    save_model_as_pickle(RFC_pipeline, output_dir + cfg.name_RFC_pipeline)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
