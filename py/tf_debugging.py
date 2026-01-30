# %%
from typing import Any, Dict, List
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from jan.py import plotHW, plotCW, plotHWC
from jan.py.py_utils import _get_target_namespace
from tqdm import tqdm
from pathlib import Path



class OPTS:
    """Wrapper for a single dataset element with dict + attribute access."""
    
    _data: Dict[str, Any]
    
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
        
        namespace = _get_target_namespace()
        namespace['opts'] = obj
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
    
    def _repr_html_(self):
        return self.print()._repr_html_()
    
    
    def plot(
        self, 
        field: str='input', 
        borders=True,
        output=True, 
        input=True,
        channel: int | None = None, 
        legend=True,
        step=1,
        title_suffix='',
        save: Path | None = None,
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
        
        fig, ax = plt.subplots()      
          
        if input:    
            ax = plotHW(tensor, ax=ax,  legend=False)
        
        if output and ('output' in self._data or 'model_output' in self._data):
            output_key = 'output' if 'output' in self._data else 'model_output'
            output_tensor = self._data[output_key]
            output_classes = self._data['output_classes'].numpy() if 'output_classes' in self._data else None
            output_classes = [c.decode() for c in output_classes] if output_classes is not None else None
            ax = plotHWC(output_tensor, ax, threshold=0.1, labels=output_classes, legend=legend)
        
        if borders and 'borders' in self._data:
            borders_tensor = self._data['borders']
            border_names = self._data['border_names'].numpy() if 'border_names' in self._data else None
            border_names = [c.decode() for c in border_names] if border_names is not None else None
            ax = plotCW(borders_tensor, ax, border_names, step=step, legend=legend)
            
        exam = self._data['exam_id'].numpy().decode()
        bscan = self._data['bscan_index'].numpy()
        x_size_px = self._data['x_size_px'].numpy()
        z_size_px = self._data['z_size_px'].numpy()
        ax.set_title(f'{exam=}\n{bscan=} | {x_size_px=} | {z_size_px=}'+title_suffix)
        
        if save:
            save.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(save/f'{exam}_{bscan}.png', dpi=150, bbox_inches="tight")
            plt.close(ax.figure)
        else:
            return ax
            

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


def find_example(ds: tf.data.Dataset, conditions: dict, max_iter=10_000):
    """
    Iterate over dataset until example matches all conditions.

    Args:
        ds: tf.data.Dataset
        conditions: dict of {key: expected_value}, compared via equality.
        max_iter: safety stop to avoid infinite loops.

    Returns:
        tf.data.Dataset with one element or a single example dict, or None if not found.
    """
    # Try to get cardinality
    try:
        card = tf.data.experimental.cardinality(ds).numpy()
        if card < 0:  # -1 or -2 => unknown / infinite
            card = None
    except Exception:
        card = None

    it = iter(ds)
    for i in tqdm(range(max_iter), total=card, desc="Searching dataset"):
        try:
            ex = next(it)
        except StopIteration:
            break

        match = True
        for key, expected in conditions.items():
            val = ex[key].numpy()
            if hasattr(val, "item") and val.shape == ():  # scalar tensor
                val = val.item()
            if isinstance(val, (bytes, bytearray)):
                val = val.decode("utf-8")
            if val != expected:
                match = False
                break

        if match:
            print(f'Found specified example at position {i}')
            return tf.data.Dataset.from_tensors(ex).repeat()

    print("Did not find specified example")
    return ds

from pathlib import Path
# from utils import get_shard_idxs, select_shards # diagnosing
from framework.data.pipelines.load import Load

class DS:
    """Debugging wrapper for tf.data.Dataset."""
    
    _dataset: tf.data.Dataset

    def __init__(
        self, 
        dataset: tf.data.Dataset | Path | str | None = None,
        exam_names = None,
    ):
        namespace = _get_target_namespace()
        
        if isinstance(dataset, Path) or isinstance(dataset, str):
            dpath = Path(dataset)
            
            # def reader_func(shards):
            #     if exam_names:
            #         shard_idxs = get_shard_idxs(dpath, exam_names)
            #         shards = select_shards(shards, shard_idxs)
            #     shards = shards.flat_map(lambda x: x)
            #     return shards
            
            dataset = Load(str(dpath.parent), dpath.name, 
                # reader_func=reader_func
                )()
        
        if dataset is None:
            dataset = namespace.get('ds')
            if dataset is None:
                print("No dataset provided and no 'ds' variable in user namespace.")
                return
            
        self._dataset = dataset
        self.ds = dataset

    def __repr__(self):
        return repr(self._dataset)

    def __getattr__(self, name):
        # delegate everything else to the underlying dataset
        return getattr(self._dataset, name)
    
    def get(self, bscan_index) -> OPTS:
        return OPTS.from_dict(next(iter(self._dataset.skip(bscan_index-1).take(1))))
    
    def vis(self, bscan_index, field='input', **plot_kwargs):
        opts = self.get(bscan_index)
        opts.plot(field, **plot_kwargs)

    def next(self) -> OPTS:
        """Return next element wrapped as OPT."""
        return OPTS.from_dict(next(iter(self._dataset)))
    
    def head(self, n_elem, keys=['exam_id', 'bscan_index']):
        for x in self._dataset.take(n_elem):
            vals = [x[key].numpy() for key in keys]
            vals = [val.decode() if isinstance(val, bytes) else val for val in vals]
            print(vals)
    
    def find(self, conditions: dict, max_iter=10_000):
        """
        Iterate over dataset until example matches all conditions.

        Args:
            ds: tf.data.Dataset
            conditions: dict of {key: expected_value}, compared via equality.
            max_iter: safety stop to avoid infinite loops.

        Returns:
            tf.data.Dataset with one element or a single example dict, or None if not found.
        """
        # Try to get cardinality
        try:
            card = tf.data.experimental.cardinality(self._dataset).numpy()
            if card < 0:  # -1 or -2 => unknown / infinite
                card = None
        except Exception:
            card = None

        it = iter(self._dataset)
        for i in tqdm(range(max_iter), total=card, desc="Searching dataset"):
            try:
                ex = next(it)
            except StopIteration:
                break

            match = True
            for key, expected in conditions.items():
                val = ex[key].numpy()
                if hasattr(val, "item") and val.shape == ():  # scalar tensor
                    val = val.item()
                if isinstance(val, (bytes, bytearray)):
                    val = val.decode("utf-8")
                if val != expected:
                    match = False
                    break

            if match:
                print(f'Found specified example at position {i}')
                return OPTS.from_dict(ex)

        print("Did not find specified example")
