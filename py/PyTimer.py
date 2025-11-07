# import time
# from jan.py.py_utils import _get_target_namespace

# class timer:
#     def __init__(self, name: str | None = None):
#         self.name = name
#         self._start = None
        
#         ns = _get_target_namespace()
        
#         if 'pytimer' in ns:
#             timer = ns['pytimer']
#             elapsed = time.perf_counter() - timer._start
#             label = f"[{timer.name}] " if timer.name else ""
#             print(f"{label}Elapsed: {elapsed:.3f} s")
#             del ns['pytimer']
#         else:    
#             self._start = time.perf_counter()
#             ns['pytimer'] = self
            
import time
from jan.py.py_utils import _get_target_namespace

class timer:
    def __init__(self, name: str = "", extra_print = ""):
        ns = _get_target_namespace()
        timers = ns.setdefault("pytimers", {})

        # existing timer: stop, print, delete
        if name in timers:
            start_time = timers.pop(name)
            elapsed = time.perf_counter() - start_time
            print(f"{name}{extra_print} Elapsed: {elapsed:.3f} s", flush=True)

        # new timer: start and store
        else:
            timers[name] = time.perf_counter()
            if extra_print: print(extra_print, flush=True)
