"""Extractor functions that misbehave, run in a child by test_extract_isolation.

A module rather than lambdas because the child imports the function by name.
"""

import os
import signal
import time


def hard_exit(path, config):
    os._exit(7)


def segfault(path, config):
    os.kill(os.getpid(), signal.SIGSEGV)


def hang(path, config):
    time.sleep(60)


def hog(path, config):
    return str(len(bytearray(2**30)))


def raises(path, config):
    raise ValueError("bad pdf")


def chatty(path, config):
    print("progress noise on stdout")
    return "text \ud800 with a lone surrogate"


def dies_on_a(path, config):
    if "test_doc_a" in path:
        os._exit(9)
    return "extracted fine"
