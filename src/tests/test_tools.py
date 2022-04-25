"""
Tests for tools.py
"""

from src.tools import get_path_root, root_directory, get_absolute_path


def test_get_path_root():
    """
    Test that path returned is the absolut path to the project directory
    """
    path_root = get_path_root()
    assert path_root.name == root_directory
    assert path_root.is_absolute()


def test_get_absolute_path():
    local_path = "data/accounts.csv"
    absolute_path = get_absolute_path(local_path)
    assert absolute_path.is_absolute()
    assert absolute_path.name == 'accounts.csv'
    assert absolute_path.parents[1].name == root_directory
