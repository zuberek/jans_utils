# %%

from typing import Any, Dict, List, Union
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Any

# %%


def opt_status(opts: Dict[str, Any], fields=[]) -> List[Any]:
    out: List[Any] = [
        tf.as_string(tf.timestamp(), precision=3),
        opts["exam_id"],
        opts["bscan_index"],
    ]
    for f in fields:
        out.extend([f, tf.shape(opts[f])])
    return out


def opt_status_pretty(opts: Dict[str, Any], fields=None, exam_chars: int = 12) -> pd.DataFrame:

    if fields:
        fields = [k for k in fields if k in opts.keys()]

    if not fields:
        fields = [k for k in opts.keys() if k not in (
            "exam_id", "bscan_index")]

    # unpack opt_status into pieces
    status = opt_status(opts, fields=fields)
    ts, exam_ids, bscan_indices, *field_pairs = status

    # unify exam_ids and indices into arrays
    if exam_ids.shape.rank == 0:
        exam_ids = tf.expand_dims(exam_ids, 0)
        bscan_indices = tf.expand_dims(bscan_indices, 0)

    exam_ids_np = exam_ids.numpy()
    bscan_indices_np = bscan_indices.numpy()

    # prepare a dict mapping field->shape tensor
    field_dict = {name: shape for name, shape in zip(
        field_pairs[0::2], field_pairs[1::2])}

    def make_row(eid: bytes, idx: int) -> dict:
        row = {
            "exam": eid.decode()[:exam_chars],
            "idx": int(idx),
        }
        for name, shape_t in field_dict.items():
            # row[f'{name}_shape'] = str(tuple(shape_t.numpy().tolist()))
            row[name] = str(tuple(shape_t.numpy().tolist()))
        return row

    rows = [make_row(eid, idx)
            for eid, idx in zip(exam_ids_np, bscan_indices_np)]
    df = pd.DataFrame(rows)
    df = df.set_index(["exam", "idx"])
    return df


def print_progress():
    def fn(opts: Dict[str, Any]) -> Dict[str, Any]:
        tf.print(opt_status(opts))
        return opts
    return fn
