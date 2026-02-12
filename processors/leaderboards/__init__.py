__all__ = ["main", "run_analysis"]

def main(*args, **kwargs):
    from .main import main as _main
    return _main(*args, **kwargs)

def run_analysis(*args, **kwargs):
    from .main import run_analysis as _run
    return _run(*args, **kwargs)
