"""
Functions to search and list BIDS datasets


=====================================================
Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import re
import posixpath
import fnmatch
import logging
from ieegprep.fileio import VALID_FORMAT_EXTENSIONS


def list_bids_datasets(bids_search_directory, dataset_extensions=None, subjects_filter=None, subset_search_pattern=None, strict_search=False, only_subjects_with_subsets=False):
    """
    Search and list the subjects and datasets for a given BIDS directory. The BIDS directory functions as a search
    starting point and can point to  any depth of the BIDS structure. Only datasets that are found (recursively) from
    that search depth are included.

    Args:
        bids_search_directory (str):        The BIDS directory to search subject and datasets in
        dataset_extensions (list/tuple):    The file extensions that will be considered as datasets. If set to None (default)
                                            all the readable dataset formats will be considered (EDF, BrainVision, MEF3)
        subjects_filter (list/tuple):       Optional list of subjects to include. If set, then any subject found that is
                                            not included in this list will be excluded from the results. Pass
                                            None (default) to include all found subjects.
        subset_search_pattern (str/list):   Optional search string or tuple/list of search strings. If a dataset name
                                            contains any of these subset patterns, then it will be included in the search
                                            results. Set to None (default) to include all datasets.
        strict_search (bool):               List only subjects and datasets that abide (more) strictly to the BIDS folder
                                            and filename structure, such as whether to require a dataset to be in a modality
                                            folder and matching of the subject-label between dataset names and their directories.
                                            Set to True to be more strict, False (default) for less strict searching.
        only_subjects_with_subsets (bool):  Whether to list a subject in the results that has no listed datasets after
                                            applying the search arguments (dataset_extensions, subjects_filter and
                                            subset_search_pattern) in it. Set to True in order to only add subjects to
                                            the results that have any datasets. False (default) to also list subjects
                                            without datasets.

    Returns:
        A dictionary with subjects and their corresponding datasets. The keys of the dictionary represent the
        subject-labels. Each dictionary value holds a list with paths to datasets for that specific subject.

    Raises:
        NotADirectoryError:                 If the BIDS input path does not exist or is not a directory
        ValueError:                         Invalid subset_search_pattern or dataset_extensions parameter
    """

    # retrieve the absolute/resolved path
    bids_search_directory = os.path.abspath(os.path.expanduser(os.path.expandvars(bids_search_directory)))
    if not os.path.isdir(bids_search_directory):
        logging.error('The BIDS input path \'' + bids_search_directory + '\' does not exists or is not a directory')
        raise NotADirectoryError('The BIDS input path does not exists or is not a directory')

    # prepare the search patterns parameter
    if subset_search_pattern is None or not subset_search_pattern:
        subset_patterns = ('',)
    elif isinstance(subset_search_pattern, str):
        subset_patterns = (subset_search_pattern.lower(), )
    elif isinstance(subset_search_pattern, list) or isinstance(subset_search_pattern, tuple):
        subset_patterns = subset_search_pattern
        subset_patterns = list(set(subset_patterns))            # make unique
        subset_patterns = [x.lower() for x in subset_patterns]  # lower-case
    else:
        logging.error('Invalid subset_search_pattern parameter, this argument can either be None (default), a single search string or a tuple/list of strings')
        raise ValueError('Invalid subset_search_pattern parameter')

    # prepare the data formats parameter
    if dataset_extensions is not None and dataset_extensions:
        extensions = dataset_extensions
        for extension in extensions:
            if not any(extension in x for x in VALID_FORMAT_EXTENSIONS):
                logging.error('Invalid dataset_extensions parameter, this argument can either be None (default), or a tuple/list of combining one or more valid extensions (valid extension are: ' + ', '.join(VALID_FORMAT_EXTENSIONS) + ').')
                raise ValueError('Invalid dataset_extensions parameter')
    else:
        extensions = VALID_FORMAT_EXTENSIONS

    # prepare subject filter parameter
    if not subjects_filter is None and subjects_filter:

        # if subject should be filtered, then lower-case the filter-subjects and remove the 'sub-' prefix
        subjects_filter = [s.lower().replace('sub-', '') for s in subjects_filter]


    #
    # inventorize the subject(s)
    #

    datasets = dict()
    subject_dirs = dict()

    # check if there as subject ('sub-') directories in the input directory
    #
    # Note:   First check root, and only then single directory. This will ensure that even something unlikely
    #         as '/desktop/sub-test/BIDS_root' will work as an input path
    # Note 2: next/os.walk is a fast method to find subject folders, see test_list_subject_dirs_perf.py
    sub_folders = next(os.walk(bids_search_directory))[1]
    sub_folders = [f for f in sub_folders if f.lower().startswith('sub-')]
    if sub_folders:
        # the input directory is in a BIDS root

        # apply subject filter if needed
        if not subjects_filter is None and subjects_filter:

            # filter and add each subject as a subject directory (with the subject folder for further searching of subsets)
            for subject_label in sub_folders:
                if subject_label[4:].lower() in subjects_filter:
                    subject_dirs[subject_label] = os.path.join(bids_search_directory, subject_label)
        else:

            # add each subject as a subject directory (with the subject folder for further searching of subsets)
            for subject_label in sub_folders:
                subject_dirs[subject_label] = os.path.join(bids_search_directory, subject_label)

    else:
        # perhaps the input path is in a subject (sub)directory (or a directory with data files where we can extract the subject name from)

        # subtract the name (returns without 'sub-' prefix)
        subject_label = _extract_subject_label_from_path(bids_search_directory)
        if subject_label is None:
            # no subject name could be derived from the input path, this already breaks the BIDS specification.

            if strict_search:
                print('Warning: The input directory is neither a BIDS root directory (no subject directories were found)\n'
                      '         nor a BIDS sub-directory (unable to extract a \'sub-<label>\' from the BIDS input\n'
                      '         path. The BIDS standard dictates that data files should be in a subject directory).')

            else:
                # try to search for datasets further down the path

                print('Warning: The input directory is neither a BIDS root directory (no subject directories were found)\n'
                      '         nor a BIDS sub-directory (unable to extract a \'sub-<label>\' from the BIDS input path. The BIDS\n'
                      '         standard dictates that data files should be in a subject directory).\n\n'
                      '         Now recursively searching the directory for BIDS data files (that according to the BIDS\n'
                      '         standard should include a subject-label in their filename)...')

                # search for files
                data_files = _search_directory_for_datasets(bids_search_directory, extensions, name_search_patterns=subset_patterns, modalities=None)

                # loop over the results and split out the different subjects with their corresponding subjects labels
                for file in data_files:
                    file_subject_label = _extract_subject_label_from_path(file)

                    if file_subject_label is None:
                        print('Warning: Unable to determine the subject label for dataset \'' + file + '\'. Excluded from listing.')
                    else:

                        # apply the subject filter if there is one
                        if not subjects_filter is None and subjects_filter:
                            if not file_subject_label.lower() in subjects_filter:
                                continue

                        # add the subject and datafiles
                        if not file_subject_label in datasets.keys():
                            datasets['sub-' + file_subject_label] = []
                        datasets['sub-' + file_subject_label].append(file)

        else:
            # subject label was found

            # add subject as a subject directory (with the subject folder for further searching of subsets)
            # apply the subject filter if there is one
            if not subjects_filter is None and subjects_filter:
                if subject_label.lower() in subjects_filter:
                    subject_dirs['sub-' + subject_label] = bids_search_directory
            else:
                subject_dirs['sub-' + subject_label] = bids_search_directory


    # loop over the subject directories
    for dir_subject_label, subject_path in subject_dirs.items():

        # search file data files in the subject directory (given more or less constrains)
        if strict_search:
            data_files = _search_directory_for_datasets(subject_path, extensions, name_search_patterns=subset_patterns, modalities=('ieeg', 'eeg'))
        else:
            data_files = _search_directory_for_datasets(subject_path, extensions, name_search_patterns=subset_patterns, modalities=None)

        # loop over the datasets that were found in the subject dir
        for file in data_files:
            file_subject_label = _extract_subject_label_from_path(file)

            # check if the file's subject label corresponds with the subject-label from the directory
            if file_subject_label is None:
                if strict_search:
                    print('Warning: Unable to determine the subject-label for dataset \'' + file + '\'. Excluded file from listing.')
                    continue
                else:
                    print('Warning: Unable to determine the subject-label for dataset \'' + file + '\'.\n'
                          '         Assuming the subject-label from the directory as the subject-label for the file')
                    file_subject_label = dir_subject_label[4:]

            # check if file subject label is equal to the sub-folder subject label
            if not file_subject_label == dir_subject_label[4:]:
                if strict_search:
                    print('Warning: The subject-label in the file (\'sub-' + file_subject_label + '\') and the subject-label\n'
                          '         in the directory (\'' + dir_subject_label + '\') do not match.\n'
                          '         Excluded file from listing.')
                    continue

                else:
                    print('Warning: The subject-label in the file (\'sub-' + file_subject_label + '\') and the subject-label\n'
                          '         in the directory (\'' + dir_subject_label + '\') do not match.\n'
                          '         Assuming the subject-label from the file is correct')

                # it is possible that the subject-label from the file has replaced subject-label from the directory by now, therefore
                # we re-check whether the subject-label (from the file) against the included subjects list here
                if not subjects_filter is None and subjects_filter and not file_subject_label.lower() in subjects_filter:
                    continue

            # add the subject and data file
            if not ('sub-' + file_subject_label) in datasets.keys():
                datasets['sub-' + file_subject_label] = []
            datasets['sub-' + file_subject_label].append(file)

        # empty subjects might also need to be listed
        # if so, check to make sure an empty entry exists (in case no files were added above)
        if not only_subjects_with_subsets and not dir_subject_label in datasets.keys():
            datasets[dir_subject_label] = []


    # sort the subjects and datasets
    dataset_sorted = dict()
    for subject in sorted(datasets):
        dataset_sorted[subject] = datasets[subject]
        dataset_sorted[subject].sort()

    #
    return dataset_sorted


def _extract_subject_label_from_path(path):
    """
    Extract the BIDS subject-label from a given path. This can either be a BIDS directory or a BIDS datafile.
    If the path points to a file (determined by the presence of a file-extension), then the subject-label will be
    extracted from the filename and the directory in which the file resides will be ignored.

    Args:
        path (str):                         The directory or filepath to extract the subject label from

    Returns:
        The subject label (without the 'sub-' prefix) if successful; None on failure to extract

    """

    # detect the prefix
    sub_start = path.rfind('sub-')
    if sub_start != -1:

        # check if file (every BIDS file has an extension)
        sub_end = path.find('.', sub_start)

        if sub_end != -1:
            # assume file

            # check for directory seperator (if file, there should not be a file seperator after the 'sub-' part)
            seperator_pos = max(path.find('/', sub_start), path.find('\\', sub_start))
            if seperator_pos == -1:
                # no directory-seperator, can continue to assume file

                # extract the subject-name from the file
                sub_end = path.find('_', sub_start)
                if sub_end != -1 and sub_end != sub_start + 4:
                    return path[sub_start + 4:sub_end]

        else:
            # assume directory

            sub_end = path.find('_', sub_start)
            if sub_end == -1:                                       sub_end = max(path.find('/', sub_start), path.find('\\', sub_start))
            if sub_end == -1 and len(path[sub_start:]) > 4:         sub_end = len(path)

            if sub_end != -1 and sub_end != sub_start + 4:
                return path[sub_start + 4:sub_end]

    # return failure
    return None


def _search_directory_for_datasets(search_path, dataset_extensions, name_search_patterns=('',), modalities=None):
    """
    Search the input directory recursively for (BIDS) datasets

    Args:
        search_path (str):                  The path to the directory to (recursively) search
        dataset_extensions (list/tuple):    The extensions that will be considered as datasets
        name_search_patterns (list/tuple):  A list of search strings. If a dataset name contains any of these subset
                                            patterns it will be included in the search results. The pattern search is
                                            case-insensitive
        modalities (list/tuple):            A list of BIDS modalities. If specified, the function will only search for
                                            datasets in BIDS paths that match any of the specified modalities (e.g. '*/ieeg/*')
                                            To disable the modalities requirement, set this argument to None.

    Returns:
        A list of datasets that were found in the directory

    """

    # reproduce the filter function from fnmatch but compile a case-insensitive regex
    # Note: this allows the search to be case-insensitive, but will leave the casing intact on the returned filenames (important file unix filepaths)
    # (which would be more complicated in a solution where all the search patterns and all filenames would be lowered)
    def filter_case_insensitive(names, pat):
        result = []
        pat = os.path.normcase(pat)
        regex = fnmatch.translate(pat)
        match = re.compile(regex, re.IGNORECASE).match
        if os.path is posixpath:
            for name in names:
                if match(name):
                    result.append(name)
        else:
            for name in names:
                if match(os.path.normcase(name)):
                    result.append(name)
        return result


    # loop over all folders within the search path
    subsets = []
    for root, dirs, files in os.walk(search_path):
        if modalities is None or root.lower().endswith(modalities):
            for extension in dataset_extensions:
                for search_pattern in name_search_patterns:
                    search_pattern = '*' + search_pattern + '*' if search_pattern else '*'
                    subsets.extend([os.path.join(root, f) for f in filter_case_insensitive(files, search_pattern + extension)])
                    subsets.extend([os.path.join(root, f) for f in filter_case_insensitive(dirs, search_pattern + extension)])

    # in the case of multiple search patterns, filenames can satisfy multiple patterns and therefore occur multiple
    # times in the results. Check and remove duplicates here
    if len(name_search_patterns) > 1:
        subsets = [*set(subsets)]

    # bring subsets with multiple formats/extensions down to one format (prioritized to occurrence in the extension var)
    for subset in subsets:
        subset_name = subset[:subset.rindex(".")]
        for subset_other in reversed(subsets):
            if not subset == subset_other:
                subset_other_name = subset_other[:subset_other.rindex(".")]
                if subset_name == subset_other_name:
                    subsets.remove(subset_other)

    # return the results
    return subsets
