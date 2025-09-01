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
        if s1.__class__ != s2.__class__:
            diffs.append(
                f"class {s1.__class__.__name__} != {s2.__class__.__name__}")
        if not shapes_compatible(s1.shape, s2.shape):
            diffs.append(f"shape {s1.shape} != {s2.shape}")
        if s1.dtype != s2.dtype:
            diffs.append(f"dtype {s1.dtype} != {s2.dtype}")

    print('Check done')
