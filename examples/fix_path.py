"""
Helper module to add the project root to Python path.
Import this at the beginning of your notebook to fix path issues.
I'm so dumb
"""
import os
import sys

def add_project_root_to_path():
    """Add the project root directory to Python path"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_root = os.path.dirname(current_dir)
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to Python path")
    else:
        print(f"{project_root} already in Python path")
        
    return project_root 