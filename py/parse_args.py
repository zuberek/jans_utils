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
            if i+1 < len(parts) and not parts[i+1].startswith("--"):
                vars_dict[name] = parts[i+1]
                i += 2
            else:
                vars_dict[name] = True
                i += 1
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
