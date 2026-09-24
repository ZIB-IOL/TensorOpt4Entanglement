"""Per-table generators.

Each module in this package owns one family of tables and exposes:

    TABLES : dict  name -> spec, where spec has at least `m` (subsystem count)
                   and `rows` (list of (algo_code, label, emphasise))
    build(name, spec, ctx) -> str   the LaTeX body

`ctx` is a `tables.common.Context` carrying the search paths and the loaded
instances. `make_tables.py` discovers the modules listed in `MODULES` and
dispatches by table name.
"""
from . import common, main, lowrank, gapclosing, ddps, memory

# size.py is retained for reference but no longer registered: its numbers
# are summarised in the caption of tab.mem, so the per-m tables it emitted
# were never cited by the paper.
MODULES = (main, lowrank, gapclosing, ddps, memory)

# table name -> (owning module, spec)
REGISTRY = {}
for _mod in MODULES:
    for _name, _spec in _mod.TABLES.items():
        if _name in REGISTRY:
            raise RuntimeError(f"table {_name!r} defined by two modules")
        REGISTRY[_name] = (_mod, _spec)
