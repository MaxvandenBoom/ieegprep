"""
Unit test to measure the performance of listing subject ('sub-' prefix) directories from an input directory

Conlusions:
  - os.scandir and next(os.walk(directory))[1] are fastest

=====================================================
Adapted from a StackOverflow by users 'user136036' and 'poppie' (https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory)
Max van den Boom: The 'run_os_walk()' function ended up doing a recursive walk, added the 'next(os.walk(directory))[1]' to the test
"""

import time
import os
from glob import glob
from pathlib import Path

#
# configuration
#
directory = '~/Documents/ccepAge'
RUNS = 1000


# retrieve the absolute/resolved path
directory = os.path.abspath(os.path.expanduser(os.path.expandvars(directory)))


def run_os_scandir():
    a = time.time_ns()
    for i in range(RUNS):
        fu = [f.name for f in os.scandir(directory) if f.is_dir() and f.name.lower().startswith('sub-')]
    print(f"os.scandir\t\t\ttook {(time.time_ns() - a) / 1000:.2f}\t\t{(time.time_ns() - a) / 1000 / 1000 / RUNS:.0f} ms.\tFound dirs: {len(fu)}")


def run_os_walk_next():
    a = time.time_ns()
    for i in range(RUNS):
        fu = next(os.walk(directory))[1]
        fu = [f for f in fu if f.lower().startswith('sub-')]
    print(f"os.walk_next\t\ttook {(time.time_ns() - a) / 1000:.2f}\t\t{(time.time_ns() - a) / 1000 / 1000 / RUNS:.0f} ms.\tFound dirs: {len(fu)}")


def run_glob():
    a = time.time_ns()
    for i in range(RUNS):
        fu = glob(directory + "/sub-*/")
    print(f"glob.glob\t\t\ttook {(time.time_ns() - a) / 1000:.2f}\t\t{(time.time_ns() - a) / 1000 / 1000 / RUNS:.0f} ms.\tFound dirs: {len(fu)}")


def run_pathlib_iterdir():
    a = time.time_ns()
    for i in range(RUNS):
        dirname = Path(directory)
        fu = [f.name for f in dirname.iterdir() if f.is_dir() and f.name.lower().startswith('sub-')]
    print(f"pathlib.iterdir\t\ttook {(time.time_ns() - a) / 1000:.2f}\t\t{(time.time_ns() - a) / 1000 / 1000 / RUNS:.0f} ms.\tFound dirs: {len(fu)}")


def run_os_listdir():
    a = time.time_ns()
    for i in range(RUNS):
        fu = [o for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o)) and o.lower().startswith('sub-')]
    print(f"os.listdir\t\t\ttook {(time.time_ns() - a) / 1000:.2f}\t\t{(time.time_ns() - a) / 1000 / 1000 / RUNS:.0f} ms.\tFound dirs: {len(fu)}")


if __name__ == '__main__':
    run_os_scandir()
    run_os_walk_next()
    run_glob()
    run_pathlib_iterdir()
    run_os_listdir()
