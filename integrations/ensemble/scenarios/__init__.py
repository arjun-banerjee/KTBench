"""Auto-import every scenario module so its ``@scenario`` decorator fires."""

from importlib import import_module
from pathlib import Path

_PKG = Path(__file__).resolve().parent
for _p in sorted(_PKG.glob("*.py")):
    if _p.name == "__init__.py" or _p.name.startswith("_"):
        continue
    import_module(f"{__name__}.{_p.stem}")
