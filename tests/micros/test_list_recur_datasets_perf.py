"""
Unit test to measure the performance of recursively listing datasets from an input directory


=====================================================
Timing method adapted from a StackOverflow by users 'user136036' and 'poppie' (https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory)
"""

import time
import os
import fnmatch
from glob import glob
from pathlib import Path
from ieegprep.fileio import VALID_FORMAT_EXTENSIONS

#
# configuration
#
extensions = VALID_FORMAT_EXTENSIONS
directory = '~/Documents/ccepAge'
#bids_folder = '~/Documents/ERDetect/'
subset_patterns = ('lin_run-021050','lin_run-021050','lin_run-021050','lin_run-021050','lin_run-021050','lin_run-021050')
#subset_patterns = ('lin_run-021050',)
RUNS = 1
modalities = ('eeg', 'ieeg')  # ieeg and eeg

# retrieve the absolute/resolved path
directory = os.path.abspath(os.path.expanduser(os.path.expandvars(directory)))

def run_os_walk():
    a = time.time_ns()
    for i in range(RUNS):
        fu = []

        for root, dirs, files in os.walk(directory):
            for extension in extensions:
                for subset_pattern in subset_patterns:
                    subset_pattern = '*' + subset_pattern + '*' if subset_pattern else '*'
                    fu.extend([x for x in fnmatch.filter(files, subset_pattern + extension)])
                    fu.extend([x for x in fnmatch.filter(dirs, subset_pattern + extension)])

        #for f in fu:
        #    print(f)


    print(f"os.walk\t\t\ttook {(time.time_ns() - a) / 1000 / 1000 / RUNS:.0f} ms. Found dirs: {len(fu)}")

def run_os_walk_with_modality():
    a = time.time_ns()
    for i in range(RUNS):
        fu = []

        for root, dirs, files in os.walk(directory):
            if modalities is None or root.lower().endswith(modalities):
                for extension in extensions:
                    for subset_pattern in subset_patterns:
                        subset_pattern = '*' + subset_pattern + '*' if subset_pattern else '*'
                        fu.extend([os.path.join(root, x) for x in fnmatch.filter(files, subset_pattern + extension)])
                        fu.extend([os.path.join(root, x) for x in fnmatch.filter(dirs, subset_pattern + extension)])

        #for f in fu:
        #    print(f)


    print(f"os.walk_with_m\t\t\ttook {(time.time_ns() - a) / 1000 / 1000 / RUNS:.0f} ms. Found dirs: {len(fu)}")

def run_single_glob():
    a = time.time_ns()
    for i in range(RUNS):
        fu = []

        for modality in modalities:
            for subset_pattern in subset_patterns:
                subset_pattern = '*' + subset_pattern + '*' if subset_pattern else '*'
                fu.extend([p.resolve() for p in Path(directory).glob('**/' + subset_pattern) if p.suffix in extensions])


    #for f in fu:
    #    print(f)

    print(f"run_single_glob\t\ttook {(time.time_ns() - a) / 1000:.2f}\t{(time.time_ns() - a) / 1000 / 1000 / RUNS:.0f} ms.\tFound dirs: {len(fu)}")

def run_multi_glob():
    a = time.time_ns()
    for i in range(RUNS):
        fu = []

        subsets = []
        for extension in extensions:
            for modality in modalities:
                for subset_pattern in subset_patterns:
                    subset_pattern = '*' + subset_pattern + '*' if subset_pattern else '*'
                    fu.extend([x for x in Path(directory).glob('**/' + subset_pattern + extension)])
        #fu = [*set(fu)]
    #for f in fu:
    #    print(f)

    print(f"run_multi_glob\t\ttook {(time.time_ns() - a) / 1000:.2f}\t{(time.time_ns() - a) / 1000 / 1000 / RUNS:.0f} ms.\tFound dirs: {len(subsets)}")


if __name__ == '__main__':
    run_os_walk()
    run_os_walk_with_modality()
    run_single_glob()
    run_multi_glob()

