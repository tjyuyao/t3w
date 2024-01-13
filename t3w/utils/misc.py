import os
import uuid
import shutil
from contextlib import contextmanager
from datetime import datetime


def generate_run_hash(hash_length=24) -> str:
    full_hash = datetime.now().strftime("%y%m%d%H%M%S") + uuid.uuid4().hex
    return full_hash[:hash_length]


@contextmanager
def work_directory(path: str):
    curr = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(curr)


def rm_r(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
