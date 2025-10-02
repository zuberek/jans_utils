from py.oct.imports import *
from py.AtrrDict import AttrDict
from envvars import ENV

LOCAL_RUN = True
VISUALIZE = False
DUMP_VAL_RESULTS_TO_NPY = False
EXP_DATASET_PATH = str(Path(ENV.DATA_DIR) / 'datasets' / 'DS_27-08-2025')
TRAIN_OUT_DIR = Path(ENV.EXPERIMENTS_DIR) / 'TEST2'

pipe_name: str='train'
backup_list: List[str] | None = None
visualize = VISUALIZE

def init():
    from IPython import get_ipython
    ipy = get_ipython()
    
    if ipy is None:
        return
    
    parameters = ipy.user_ns.get("parameters", None)
    if parameters is not None:
        global params
        param_dict = dict(parameters.items())
        combination = parameters.combinations.get_combination_by_ind(0)
        
        # remove duplicates
        for k in list(combination.keys()):
            if k in param_dict:
                del combination[k]
        
        params = AttrDict(**param_dict, **combination)
        ipy.user_ns['params'] = params
    