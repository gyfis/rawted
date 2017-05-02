import os


_abs_dir = os.path.dirname(__file__)


def absolute_path(*path: str) -> str:
    return os.path.join(_abs_dir, *path)


def path_exists(path: str) -> bool:
    return os.path.exists(path)


def check_or_create_absolute_dir(dir_path: str) -> None:
    absolute_dir_path = absolute_path(dir_path)
    if not os.path.isdir(absolute_dir_path):
        if path_exists(absolute_dir_path):
            raise NotADirectoryError()
        os.mkdir(absolute_dir_path)
