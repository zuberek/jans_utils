import shlex
import os
from types import SimpleNamespace


class SafeNamespace(SimpleNamespace):
    def __getattr__(self, name):
        return None  # Return None if attribute doesn't exist


def parse(cli_command: str) -> SafeNamespace:
    # Expand env vars and split
    parts = [os.path.expandvars(p)
             for p in shlex.split(cli_command, posix=True)]

    # Dictionary for results
    vars_dict = {}
    i = 0
    while i < len(parts):
        if parts[i].startswith("--"):
            name = parts[i][2:].replace("-", "_")
            values = []
            j = i + 1
            while j < len(parts) and not parts[j].startswith("--"):
                values.append(parts[j])
                j += 1
            if values:
                # single value → str, multiple → list
                vars_dict[name] = values if len(values) > 1 else values[0]
            else:
                vars_dict[name] = True
            i = j
        else:
            i += 1

    args = SafeNamespace(**vars_dict)
    vars_dict["args"] = args

    # Inject into interactive namespace
    from IPython import get_ipython
    ipy = get_ipython()
    for k, v in vars_dict.items():
        ipy.user_ns[k] = v

    # Return as SimpleNamespace
    return args
