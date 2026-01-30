from jan.py.oct.imports import *
from jan.py.AtrrDict import AttrDict
from jan.py.py_utils import _get_target_namespace
from envvars import ENV

# %%

LOCAL_RUN = False
VISUALIZE = False
# DUMP_VAL_RESULTS_TO_NPY = False
EXP_DATASET_PATH = str(Path(ENV.DATA_DIR) / 'datasets' / 'DS_27-08-2025')
TRAIN_OUT_DIR = Path(ENV.EXPERIMENTS_DIR) / 'Train_with_CNV_BEST'

pipe_name: str='train'
backup_list: List[str] | None = None
visualize = VISUALIZE

def init(parameters):
    
    # parameters = ipy.user_ns.get("parameters", None)
    if parameters is not None:
        
        parameters['data_path'] = str(Path(EXP_DATASET_PATH) / 'anom-test')
        parameters['data_path_train'] = str(Path(EXP_DATASET_PATH) / 'anom-train')
        parameters['data_path_val'] = str(Path(EXP_DATASET_PATH) / 'anom-val')
        parameters['data_dir'] = EXP_DATASET_PATH
        parameters['dataset_records'] = (EXP_DATASET_PATH, 'anom-test')
        parameters['train_out_dir'] = TRAIN_OUT_DIR
        params = dict(parameters.items())
        params['run_output_folder'] = params['out_dir']
        if parameters.combinations.get_combinations():
            combination = parameters.combinations.get_combination_by_ind(0)
            combination = combination if combination else {}
            params = {**combination, **params}
        
        # # remove duplicates
        # for k in list(combination.keys()):
        #     if k in params:
        #         del params[k]
        
        ns = _get_target_namespace()
        ns['params'] = AttrDict(**params)
    