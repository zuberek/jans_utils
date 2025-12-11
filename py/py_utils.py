# %%
from IPython import get_ipython
from types import SimpleNamespace
import ast, inspect, builtins
from typing import Callable

# %%

def inject_to_ipy(d: dict):
    """Inject dict keys/values into the current IPython user namespace."""
    ipy = get_ipython()
    if ipy is None:
        raise RuntimeError("Not running inside IPython.")
    ipy.user_ns.update(d)

def fake_self():
    namespace = _get_target_namespace()
    namespace['self'] = SimpleNamespace()

def _get_target_namespace():
    """Return caller's namespace (works for Jupyter too)."""
    ipy = get_ipython()
    if ipy:
        return ipy.user_ns  # live Jupyter namespace
    # fallback: caller's frame globals
    return inspect.currentframe().f_back.f_back.f_globals


def inject(target: str | Callable):
    """Detect whether the input is a function definition or a function call and inject accordingly."""
    
    if isinstance(target, str):
        code = target
    
        if code.endswith("):") or code.endswith("):\n"):
            code += "\n    pass"
        
        tree = ast.parse(code.strip())
        has_func_def = any(isinstance(n, ast.FunctionDef) for n in tree.body)
        has_call = any(isinstance(n, ast.Call) for n in ast.walk(tree))

        if has_func_def:
            inject_str_defaults(code)
        if has_call:
            inject_call_args(code)
            
    if isinstance(target, Callable):
        inject_callable_defaults(target)
    
def inject_callable_defaults(target: Callable):
    namespace = _get_target_namespace()
    print('### defaults')
    sig = inspect.signature(target.__init__ if inspect.isclass(target) else target)
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty and name not in namespace:
            namespace[name] = param.default
            print(f"{name} = {param.default!r}")

def inject_str_defaults(code: str):
    """Inject all parameters with default values from a function definition string."""
    namespace = _get_target_namespace()
    
    print('### defaults')
    
    if code.endswith("):") or code.endswith("):\n"):
        code += "\n    pass"

    tree = ast.parse(code.strip())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    for arg, default in zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults):
        name = arg.arg
        value = ast.literal_eval(default)
        namespace[name] = value
        print(f"{name} = {value!r}")
        
def inject_call_args(code: str):
    """Inject keyword args from a function call string into globals()."""
    namespace = _get_target_namespace()
    tree = ast.parse(code.strip())
    call = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.Call)), None
    )
    if not call:
        raise ValueError("No function call found")
    
    func_obj = None
    if isinstance(call.func, ast.Name):
        fn_name = call.func.id
        func_obj = namespace.get(fn_name)
    if not callable(func_obj):
        print("Warning: could not resolve callable from code.")
        func_obj = None
    
    if func_obj:
        inject_callable_defaults(func_obj)
        
    if func_obj:
        print('### args')
        
        sig = inspect.signature(func_obj)
        param_names = list(sig.parameters.keys())
        
        for node, name in zip(call.args, param_names):
            value = _safe_eval(node)
            namespace[name] = value
            print(f"{name} = {value!r}")

    if(call.keywords): print('### kwargs')
    for kw in call.keywords:
        value = _safe_eval(kw.value)
        namespace[kw.arg] = value
        print(f"{kw.arg} = {value!r}")

def _safe_eval(node):
    """Try to resolve the node safely to a real value if possible."""
    namespace = _get_target_namespace()
    try:
        # literal constants
        return ast.literal_eval(node)
    except Exception:
        # variables like foo, or attributes like self.bar
        expr = _expr_to_str(node)
        try:
            return eval(expr, namespace)
        except Exception:
            return expr  # fallback to string if not resolvable
        
def _expr_to_str(node: ast.AST) -> str:
    """Safely convert an AST node (like self.x) into a source string."""
    try:
        return ast.unparse(node)
    except Exception:
        return str(node)