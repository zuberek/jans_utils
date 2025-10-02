from typing import Any, Dict, List
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from py import plotHW, plotCW


class OPTS:
    """Wrapper for a single dataset element with dict + attribute access."""
    
    def __init__(self, dataset: tf.data.Dataset = None):
        """Create OPT from first element of a tf.data.Dataset."""
        from IPython import get_ipython
        ipy = get_ipython()

        if dataset is None:
            dataset = ipy.user_ns.get("ds", None)
            if dataset is None:
                print("No dataset provided and no 'ds' variable in user namespace.")
                return

        data = next(iter(dataset))  # eager execution
        self._data = data
        self._is_batched = self._check_batched()

        # Inject into interactive namespace
        ipy.user_ns['opts'] = self


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OPTS":
        """Create OPTS directly from a dictionary."""
        obj = cls.__new__(cls)   # bypass __init__
        obj._data = data
        obj._is_batched = obj._check_batched()
        return obj
    
    def _check_batched(self) -> bool:
        """Check if all entries are batched with the same first dimension."""
        batch_sizes = []
        for v in self._data.values():
            if hasattr(v, "shape") and len(v.shape) > 0:
                batch_sizes.append(v.shape[0])
            # else:
            #     return False  # scalar/no leading dim → not batched
        return len(set(batch_sizes)) == 1 if batch_sizes else False

    # --- dict-like ---
    def __getitem__(self, key):
        # dict-like
        if isinstance(key, str):
            return self._data[key]

        # index-like
        if isinstance(key, int):
            if not self._is_batched:
                print('Trying to index unbatched opts')
                return
            new_data = {}
            for k, v in self._data.items():
                if hasattr(v, "shape") and len(v.shape) > 0:
                    new_data[k] = v[key]
                else:
                    new_data[k] = v  # rest unchanged
            return OPTS.from_dict(new_data)

        raise TypeError(f"Invalid key type: {type(key)}")

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
    
    def plotHW(
        self, 
        field: str, 
        borders=True, 
        channel: int = None, 
        legend=False,
        step=1,
    ):
        shape = self._data[field].shape
        rank = len(shape)
        assert rank == 3 or rank == 4, "This is not an (N)HWC tensor!"
        is_batched = rank == 4
        is_multichannel = shape[-1] != 1
        tensor = self._data[field]
        if is_batched:
            tensor = tensor[0] # HWC
        if is_multichannel:
            if channel: tensor = tensor[..., channel]
            else: tensor = tensor[..., 0] # HW1
            
        ax = plotHW(tensor, legend=False)
        
        if borders and 'borders' in self._data:
            borders_tensor = self._data['borders']
            ax = plotCW(ax, borders_tensor, step=step, legend=legend)
            
        plt.show()
            

    def print(self, 
              fields: List[str] | None = None, 
              exam_chars: int = 12, 
              time: bool = False, 
              dtype: bool = False
        ):
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
                        # elif v.dtype == tf.string and v.shape.rank > 0:
                        #     row[k] = _decode_str(v[i], exam_chars)
                        # elif v.shape.rank > 0 and v.shape[0] == exam_id.shape[0]:
                        #     # batched vector → show scalar value
                        #     val = v[i].numpy()
                        #     if val.ndim == 0:
                        #         row[k] = int(val)
                        #     else:
                        #         row[k] = f"{tuple(val.shape)}, {v.dtype.name}" if dtype else f"{tuple(val.shape)}"
                        else:
                            row[k] = fmt(v)
                    elif isinstance(v, tf.SparseTensor):
                        row[k] = fmt(v)
                    else:
                        row[k] = str(v)
                rows.append(row)
            return pd.DataFrame(rows)

        # unbatched (single example)
        rows = []
        for k,v in self._data.items():
            if fields and k not in fields: continue # skip if not wanted
            rows.append(format_item(k,v, exam_chars))
        return pd.DataFrame(rows)

def _decode_str(arr, exam_chars):
    val = arr.numpy()
    if isinstance(val, bytes):        # scalar tf.string
        return val.decode()[:exam_chars]
    if val.ndim == 0:                 # 0-d array of bytes
        return val.item().decode()[:exam_chars]
    if val.ndim > 0:                  # vector of strings
        return [x.decode()[:exam_chars] for x in val.tolist()]
    return str(val)



def format_item(k,v, exam_chars=12):
        # determine preview
    if isinstance(v, tf.Tensor) and v.shape.rank == 0:
        if v.dtype == tf.string:
            preview = v.numpy().decode()[:exam_chars]
        else:
            preview = int(v.numpy())
    elif isinstance(v, tf.Tensor):
        arr = v.numpy()
        preview = np.array2string(arr.flatten()[:3], separator=", ") + (" ..." if arr.size > 3 else "")
    elif isinstance(v, np.ndarray):
        preview = np.array2string(v.flatten()[:3], separator=", ") + (" ..." if v.size > 3 else "")
    else:
        preview = str(v)

    # determine shape + dtype
    if isinstance(v, tf.SparseTensor):
        shape = tuple(v.dense_shape.numpy().tolist())
        dtype_str = v.dtype.name
    elif isinstance(v, tf.Tensor):
        shape = tuple(v.shape.as_list())
        dtype_str = v.dtype.name
    elif isinstance(v, np.ndarray):
        shape = v.shape
        dtype_str = v.dtype.name
    else:
        shape = None
        dtype_str = type(v).__name__
        
    return {"name": k, "preview": preview, "shape": shape, "dtype": dtype_str}

class DS:
    """Debugging wrapper for tf.data.Dataset."""

    def __init__(self, dataset: tf.data.Dataset):
        self._dataset = dataset

    def __getattr__(self, name):
        # delegate everything else to the underlying dataset
        return getattr(self._dataset, name)

    def next(self) -> OPTS:
        """Return next element wrapped as OPT."""
        return OPTS(next(iter(self._dataset)))
