# %%
import os
from pathlib import Path
import shutil
from typing import Dict, List, Any

import tensorflow as tf

from py.oct.imports import *

from envvars import ENV

# %%
class EvalExampleCombs(ParamsCombinations):
    def get_combinations(self):
        param_combs = []
        param_combs.append({})
        return param_combs

class GenerateDataset(EvalConfig):
    """
    Offline preprocessing pipeline for anomaly datasets. Steps:

    """
    def create_evaluation_pipeline(self,
                                   params: Params,
                                   model: tf.keras.Model,
                                   metrics: Dict[str, tf.keras.metrics.Metric]) -> tf.data.Dataset:
        

        # Prepare output dir
        save_ds_path = params['run_output_folder'] / Path(params['dataset_name'])
        if os.path.exists(save_ds_path):
            shutil.rmtree(save_ds_path)
        os.makedirs(save_ds_path)

        ds = tf.data.TFRecordDataset(anom_tfrecs_paths_str)
  
        ds = Save(save_dir=str(save_ds_path), save_name='anom-data', return_empty_ds=False)(ds)
        
        return ds

parameters = {
    'project_id': 'Diagnosing',
    'experiment_name': 'Test-eval',
    'comment': 'Testing octopus evaluation',
    'db_handle': ExperimentDatabase('10.130.20.43:5432', 'test_usr', '975ZuGM1XTEg', 'experiments_tests'),
    'dry_run': True,
    'num_gpus': 1,
    'n_threads': 6,
    'out_dir': ENV.EXPERIMENTS_DIR,
    'run_output_folder': ENV.DATA_DIR
}

parameters = EvaluationExperimentParams(eval_config=GenerateDataset(),
                                        combinations=EvalExampleCombs(),
                                        common_params=CommonExpParams(**parameters))

from py import AttrDict
params = AttrDict(**dict(parameters.items()), **parameters.combinations.get_combination_by_ind(0))