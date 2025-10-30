import tensorflow as tf


def shapes_compatible(shape1, shape2):
    """Return True if shapes are equal or differ only by None vs int."""
    s1, s2 = shape1.as_list(), shape2.as_list()
    if len(s1) != len(s2):
        return False
    for d1, d2 in zip(s1, s2):
        if d1 is None or d2 is None:
            continue
        if d1 != d2:
            return False
    return True


def compare_specs(ds1: tf.data.Dataset, ds2: tf.data.Dataset):
    spec1 = ds1.element_spec
    spec2 = ds2.element_spec
    keys1, keys2 = set(spec1.keys()), set(spec2.keys())

    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2

    if only_in_1:
        print("Only in spec1:", only_in_1)
    if only_in_2:
        print("Only in spec2:", only_in_2)

    for k in sorted(common):
        s1, s2 = spec1[k], spec2[k]
        diffs = []

        if type(s1) != type(s2):
            diffs.append(f"type mismatch: {type(s1).__name__} vs {type(s2).__name__}")
        elif not (isinstance(s1, tf.TensorSpec) and isinstance(s2, tf.TensorSpec)):
            diffs.append(f"non-TensorSpec type: {type(s1).__name__}")
        else:
            if not shapes_compatible(s1.shape, s2.shape):
                diffs.append(f"shape {s1.shape} != {s2.shape}")
            if s1.dtype != s2.dtype:
                diffs.append(f"dtype {s1.dtype} != {s2.dtype}")

        if diffs:
            print(f"{k}: " + "; ".join(diffs))

    print('Check done')
