# %%

def get_nested(d: dict, path: str, sep: str = '.', default=None):
    """
    Access nested dictionary keys with a path string.

    Example:
        d = {"a": {"b": {"c": 42}}}
        get_nested(d, "a.b.c")  # -> 42
        get_nested(d, "a.b.x", default="missing")  # -> "missing"

    Args:
        d: The dictionary to access.
        path: Key path string, e.g. "a.b.c".
        sep: Separator between keys (default=".").
        default: Value to return if path not found.

    Returns:
        The value at the nested path, or default if missing.
    """
    cur = d
    for k in path.split(sep):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur
