from jan.py.oct.imports import *
from jan.py.AtrrDict import AttrDict
from envvars import ENV

# %%

LOCAL_RUN = True
VISUALIZE = False
DUMP_VAL_RESULTS_TO_NPY = False
EXP_DATASET_PATH = str(Path(ENV.DATA_DIR) / 'datasets' / 'DS_27-08-2025')
TRAIN_OUT_DIR = Path(ENV.EXPERIMENTS_DIR) / 'TEST_MODELS'

pipe_name: str='train'
backup_list: List[str] | None = None
visualize = VISUALIZE

def init(parameters):
    from IPython import get_ipython
    ipy = get_ipython()
    
    # if ipy is None:
    #     return
    
    # parameters = ipy.user_ns.get("parameters", None)
    if parameters is not None:
        
        parameters['data_path'] = Path(EXP_DATASET_PATH) / 'anom-test'
        parameters['data_path_train'] = Path(EXP_DATASET_PATH) / 'anom-train'
        parameters['data_path_val'] = Path(EXP_DATASET_PATH) / 'anom-val'
        parameters['data_dir'] = EXP_DATASET_PATH
        parameters['dataset_records'] = (EXP_DATASET_PATH, 'anom-test')
        parameters['train_out_dir'] = TRAIN_OUT_DIR
        global params
        param_dict = dict(parameters.items())
        combination = parameters.combinations.get_combination_by_ind(0)
        combination = combination if combination else {}
        full_comb = {**param_dict, **combination}
        
        # # remove duplicates
        # for k in list(combination.keys()):
        #     if k in param_dict:
        #         del param_dict[k]
        
        params = AttrDict(**full_comb)
        if ipy: ipy.user_ns['params'] = params
    