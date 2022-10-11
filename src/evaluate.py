import json
import math
import os
import pickle
import sys
sys.path.insert(1, 'D:\git_repos\lab_data_science')

import logging
import pandas as pd
import config as cfg
import utils
from pathlib import Path
from sklearn import tree
from dvclive import Live
from matplotlib import pyplot as plt
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import r2_score, mean_squared_error, precision_score, recall_score, roc_auc_score, f1_score

live = Live("reports/report")
    
    
def count_metrics(y_val, preds, model):
    recall = recall_score(y_val, preds, average='micro')
    precision = precision_score(y_val, preds, average='micro')
    roc_auc = roc_auc_score(y_val, preds, average='micro')

    live.log(model + "_avg_recall", recall)
    live.log(model + "_avg_prec", precision)
    live.log(model + "_roc_auc", roc_auc)
    
    return {'recall': recall, 'precision': precision, 'roc_auc': roc_auc }
    
# @click.command()
# @click.argument('input_dir', type=click.Path())
# @click.argument('output_dir', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Count metrics')
    
    x_val = utils.load_as_pickle(cfg.path_to_splitted_val_data)
    y_val = utils.load_as_pickle(cfg.path_to_splitted_val_data_target)
    
    catboost_pipeline = utils.load_model_as_pickle(cfg.path_to_models + cfg.name_catboost_pipeline)
    RFC_pipeline = utils.load_model_as_pickle(cfg.path_to_models + cfg.name_RFC_pipeline)
    
    predictions_catboost = catboost_pipeline.predict(x_val)
    predictions_RFC = RFC_pipeline.predict(x_val)
    
    metrics = {}
    
    metrics['catboost'] = count_metrics(y_val, predictions_catboost, 'catboost')
    metrics['RFC'] = count_metrics(y_val, predictions_RFC, 'RFC')
    
    # recall = recall_score(y_val, predictions_catboost, average='micro')

    # precision = precision_score(y_val, predictions_catboost, average='micro')

    # roc_auc = recall_score(y_val, predictions_catboost, average='micro')

    # prc_points = list(zip(precision, recall, roc_auc))
    # prc_file = os.path.join("evaluation", "plots", "precision_recall.json")
    # with open(prc_file, "w") as fd:
        # json.dump(
            # {
                # "prc": [
                    # {"precision": p, "recall": r}
                    # for p, r in prc_points
                # ]
            # },
            # fd,
            # indent=4,
        # )


    # ... and finally, we can dump an image, it's also supported:
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    importances = catboost_pipeline['model'].feature_importances_
    feature_names = x_val.columns
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=30)
    axes.set_ylabel("Mean decrease in impurity")
    forest_importances.plot.bar(ax=axes)
    fig.savefig(os.path.join("reports/figures/", "importance_catboost.png"))
    
    
    x_val_encoded = RFC_pipeline['target_encoder'].transform(x_val)
    importances = RFC_pipeline['model'].estimator.feature_importances_
    feature_names = x_val_encoded.columns
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=30)
    axes.set_ylabel("Mean decrease in impurity")
    forest_importances.plot.bar(ax=axes)
    fig.savefig(os.path.join("reports/figures/", "importance_RFC.png"))

    importances = []
    feature_names = []
    for cl, mets in metrics.items():
        for score, value in mets.items():
            feature_names.append(cl + "_" + score)
            importances.append(value)
            
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=30)
    axes.set_ylabel("Value of metric")
    forest_importances.plot.bar(ax=axes)
    fig.savefig(os.path.join("reports/figures/", "Metrics.png"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


# ... confusion matrix plot
# live.log_plot("confusion_matrix", labels.squeeze(), predictions_by_class.argmax(-1))

