import time
import numpy as np
from tqdm import tqdm
import io
from contextlib import redirect_stdout, redirect_stderr
import importlib

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # suppress TF INFO/WARN/ERROR
os.environ["PYTHONWARNINGS"] = "ignore"       # suppress Python warnings

"""

python /home/jdabrowski/code/jan/jan/py/benchmark_test_import.py

"""

module_paths = [
    "jan.py.benchmark_scripts.test_load_basic",
    # "jan.py.benchmark_scripts.test_load_round_051125",
    # "jan.py.benchmark_scripts.test_load_seq_051125",
    # "jan.py.benchmark_scripts.test_load_specific_shard_051125",
]

def load_func(mod_path):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        mod = importlib.import_module(mod_path)
        benchmark_func = getattr(mod, "benchmark_func")
        return benchmark_func

N_RUNS = 20
N_ITERATIONS = [10, 20, 80]

for mod_path in module_paths:
    module_name = mod_path.split('.')[-1]
    benchmark_func = load_func(mod_path)
    
    print(f'{module_name}')
    for n_iter in N_ITERATIONS:
        times = []
        for _ in tqdm(range(N_RUNS), desc=f"{n_iter=}", unit="run", leave=False):
            start = time.perf_counter()
            benchmark_func(n_iter)
            times.append(time.perf_counter() - start)

        times = np.array(times)
        print(f"{n_iter=} | mean: {times.mean():.3f}s  ±  std: {times.std():.3f}s")
    
