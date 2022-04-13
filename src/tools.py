"""
Library of general purpose functions
"""

from pathlib import Path, PurePath, PosixPath
import os

root_directory = "loan_defaults"


def get_path_root() -> PosixPath:
    """
    This function is used to avoid path issues when calling the code from fom different places
    :return:
    absolut path to project directory "loan_defaults"
    """
    current_path = Path(os.path.realpath(__file__)).resolve()
    path = current_path
    while path.name != root_directory:
        path = path.parent
    return path


def get_absolute_path(local_path: str) -> PosixPath:
    """
    Return absolute path corresponding to a local path within the project
    :param local_path: local path within the project eg: 'data/accounts.csv'
    :return:
    Absolute path to the file in local_path
    """
    root_path = get_path_root()
    abs_path = PurePath.joinpath(root_path, local_path)
    return abs_path
