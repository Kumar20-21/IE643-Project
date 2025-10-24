import os
import sys
from typing import List

class import_repos:
    def __init__(self, repo_paths: List[str]) -> None:
        """
        Constructor to add multiple repo paths to sys.path for dynamic imports.
        
        Args:
        - repo_paths (List[str]): A list of paths to repositories to add to sys.path.
        """
        self.repos = repo_paths
        for repo in repo_paths:
            sys.path.insert(0, os.path.abspath(repo))
    
    def pop_repo(self) -> None:
        """
        Removes the repo paths from sys.path after imports are complete.
        """
        for repo in self.repos: 
            sys.path.pop(0)