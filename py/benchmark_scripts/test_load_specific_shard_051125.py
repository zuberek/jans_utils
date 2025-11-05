# %%
from scripts.data_integration.integration_utils import load_exams
from pathlib import Path

input_path = Path('/home/jdabrowski/data/tfrecs/CNV-1')
exam_names = list(input_path.rglob('*.tfrecord'))
exam_ids = [Path(p).stem for p in exam_names][3:4]
dpath = Path('/home/jdabrowski/data/datasets/external/CNV-1')

# %%

def benchmark_func(iters=10):
    ds = load_exams(dpath, exam_ids)
    for x in ds.take(iters):
        _ = x['exam_id']  # don’t print, just iterate