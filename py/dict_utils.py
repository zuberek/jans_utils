# %%

def get_nested(d: dict, path: str, sep: str = '.', default=None):
    cur = d
    for k in path.split(sep):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def pop_nested(d: dict, path: str, sep='.', default=None):
    parts = path.split(sep)
    cur = d
    for k in parts[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur.pop(parts[-1], default)

