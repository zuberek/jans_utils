import tensorflow as tf
import pandas as pd
from typing import Any, Dict



class OPT:
    """Wrapper for a single dataset element with dict + attribute access."""
    
    def __init__(self, ds: tf.data.Dataset):
        """Create OPT from first element of a tf.data.Dataset."""
        data = next(iter(ds))  # eager execution
        self._data = data
        
        # Inject into interactive namespace
        from IPython import get_ipython
        ipy = get_ipython()
        ipy.user_ns['opts'] = self

    # def __init__(self, data: Dict[str, Any]):
    #     self._data = data

    # --- dict-like ---
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def get(self, key, default=None):
        return self._data.get(key, default)

    # --- attribute-like ---
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    # --- printing ---
    def __repr__(self):
        return repr(self._data)


    def print(self, exam_chars: int = 12, time: bool = False, dtype: bool = False):
        """Pretty-print all fields with shape + dtype, supports SparseTensors and batching."""
        ts = pd.to_datetime(float(tf.timestamp().numpy()), unit="s").floor("s")

        def fmt(t):
            if isinstance(t, tf.SparseTensor):
                shape = tuple(t.dense_shape.numpy().tolist())
                return f"{shape}, {t.dtype.name}" if dtype else f"{shape}"
            if isinstance(t, tf.Tensor):
                shape = tuple(t.shape.as_list())
                return f"{shape}, {t.dtype.name}" if dtype else f"{shape}"
            return str(type(t))

        # detect batching from exam_id (string tensor)
        exam_id = self._data.get("exam_id", None)
        if isinstance(exam_id, tf.Tensor) and exam_id.shape.rank > 0 and exam_id.shape[0] > 0:
            rows = []
            for i in range(exam_id.shape[0]):
                row = {"time": ts} if time else {}
                for k, v in self._data.items():
                    if isinstance(v, tf.Tensor):
                        if v.shape.rank == 0 and v.dtype == tf.string:
                            row[k] = v.numpy().decode()[:exam_chars]
                        elif v.shape.rank == 0:
                            row[k] = int(v.numpy())
                        elif v.dtype == tf.string and v.shape.rank > 0:
                            row[k] = v[i].numpy().decode()[:exam_chars]
                        elif v.shape.rank > 0 and v.shape[0] == exam_id.shape[0]:
                            # batched vector → show scalar value
                            val = v[i].numpy()
                            if val.ndim == 0:
                                row[k] = int(val)
                            else:
                                row[k] = f"{tuple(val.shape)}, {v.dtype.name}" if dtype else f"{tuple(val.shape)}"
                        else:
                            row[k] = fmt(v)
                    elif isinstance(v, tf.SparseTensor):
                        row[k] = fmt(v)
                    else:
                        row[k] = str(v)
                rows.append(row)
            return pd.DataFrame(rows)

        # unbatched (single example)
        row = {"time": ts} if time else {}
        for k, v in self._data.items():
            if isinstance(v, tf.Tensor) and v.shape.rank == 0 and v.dtype == tf.string:
                row[k] = v.numpy().decode()[:exam_chars]
            elif isinstance(v, tf.Tensor) and v.shape.rank == 0:
                row[k] = int(v.numpy())
            else:
                row[k] = fmt(v)
        return pd.DataFrame([row])


class DS:
    """Debugging wrapper for tf.data.Dataset."""

    def __init__(self, dataset: tf.data.Dataset):
        self._dataset = dataset

    def __getattr__(self, name):
        # delegate everything else to the underlying dataset
        return getattr(self._dataset, name)

    def next(self) -> OPT:
        """Return next element wrapped as OPT."""
        return OPT(next(iter(self._dataset)))
