# %%
from pathlib import Path
from framework.data.pipelines.load import Load

dpath = Path('/home/jdabrowski/data/datasets/external/CNV-1')

# %%
def benchmark_func(iters=10):
    ds = Load(
        save_dir=str(dpath.parent), 
        save_name=dpath.name,
        reader_func=lambda shards: shards.flat_map(lambda x: x))()
    for x in ds.take(iters):
        _ = x['exam_id']  # don’t print, just iterate