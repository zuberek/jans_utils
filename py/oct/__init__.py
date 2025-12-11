from jan.py.oct.imports import *
from jan.py.AtrrDict import AttrDict
from jan.py.py_utils import _get_target_namespace
from envvars import ENV

# %%

LOCAL_RUN = False
VISUALIZE = False
DUMP_VAL_RESULTS_TO_NPY = False
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
        param_dict = dict(parameters.items())
        combination = parameters.combinations.get_combination_by_ind(0)
        combination = combination if combination else {}
        full_comb = {**combination, **param_dict}
        
        # # remove duplicates
        # for k in list(combination.keys()):
        #     if k in param_dict:
        #         del param_dict[k]
        
        ns = _get_target_namespace()
        ns['params'] = AttrDict(**full_comb)
    