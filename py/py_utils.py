from IPython import get_ipython

def inject_to_ipy(d: dict):
    """Inject dict keys/values into the current IPython user namespace."""
    ipy = get_ipython()
    if ipy is None:
        raise RuntimeError("Not running inside IPython.")
    ipy.user_ns.update(d)
