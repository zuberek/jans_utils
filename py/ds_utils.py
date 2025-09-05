# %%
import tensorflow as tf
import pandas as pd

# %%


def get_ds_info(ds: tf.data.Dataset):
    bscan_count = 0
    exam_ids = set()

    for x in ds:
        bscan_count += 1
        exam_ids.add(x['exam_id'].numpy().decode("utf-8"))

    info = {
        "bscan_count": [bscan_count],
        "exam_count": [len(exam_ids)],
    }
    return pd.DataFrame(info)
