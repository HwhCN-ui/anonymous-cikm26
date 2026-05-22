
import os

__version__ = "0.0.2"



def root_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    return project_root
