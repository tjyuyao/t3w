import os
import time
import uuid
from contextlib import contextmanager


def base32_timestamp():
    cur = time.localtime()
    base32 = "0123456789abcdefghjklmnpqrtuvwxy"
    y = base32[cur.tm_year%10]
    m = base32[cur.tm_mon]
    d = base32[cur.tm_mday]
    h = base32[cur.tm_hour]
    M = base32[cur.tm_min//2]
    s = base32[time.time_ns()%1024//32] + base32[time.time_ns()%32]
    return f"{y}{m}{d}{h}{M}{s}"


def generate_run_hash(hash_length=24) -> str:
    full_hash = base32_timestamp() + uuid.uuid4().hex
    return full_hash[:hash_length]


@contextmanager
def work_directory(path: str):
    curr = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(curr)
