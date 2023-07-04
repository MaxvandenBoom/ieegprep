"""
Functions to load BIDS sidecar metadata


=====================================================
Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from math import isnan
import json
import logging
import pandas as pd
from ieegprep.utils.misc import is_number


def load_channel_info(filepath):
    """

    Retrieve the channel metadata from a _channels.tsv file

    Args:
        filepath (str):             The path to the _channels.tsv file to load

    Returns:
        csv (dataframe):            A pandas dataframe containing the channels information

    Raises:
        FileNotFoundError:          If the file could not be found
        LookupError:                If one of the required columns could not be found

    """

    try:
        csv = pd.read_csv(filepath, sep='\t', header=0, encoding='unicode_escape', na_filter=False, dtype=str)
    except FileNotFoundError:
        logging.error('Could not find the file \'' + filepath + '\'')
        raise FileNotFoundError('Could not find file')

    # check the existence of required columns
    if 'name' not in csv.columns:
        logging.error('Could not find the \'name\' column in \'' + filepath + '\'')
        raise LookupError('Could not find column')
    if 'type' not in csv.columns:
        logging.error('Could not find the \'type\' column in \'' + filepath + '\'')
        raise LookupError('Could not find column')
    if 'status' not in csv.columns:
        logging.error('Could not find the \'status\' column in \'' + filepath + '\'')
        raise LookupError('Could not find column')

    #
    return csv


def load_event_info(filepath, addition_required_columns=None):
    """
    Retrieve the events from a _events.tsv file

    Args:
        filepath (str):                         The path to the _events.tsv file to load
        addition_required_columns(list/tuple):  One or multiple additional columns that need to be present in the _events.tsv

    Returns:
        csv (dataframe):                        A pandas dataframe containing the events information

    Raises:
        FileNotFoundError:                      If the file could not be found
        LookupError:                            If the mandatory 'onset' column or any of the required additional
                                                columns could not be found
    """

    try:
        csv = pd.read_csv(filepath, sep='\t', header=0, encoding='unicode_escape', na_filter=False, dtype=str)
    except FileNotFoundError:
        logging.error('Could not find the file \'' + filepath + '\'')
        raise FileNotFoundError('Could not find file')

    # check the existence of required columns
    if 'onset' not in csv.columns:
        logging.error('Could not find the \'onset\' column in \'' + filepath + '\'')
        raise LookupError('Could not find column')
    if addition_required_columns is not None:
        for column in addition_required_columns:
            if column not in csv.columns:
                logging.error('Could not find the \'' + column + '\' column in \'' + filepath + '\'')
                raise LookupError('Could not find column')

    #
    return csv


def load_stim_event_info(filepath, additional_required_columns=None):
    """
    Retrieve the electrical stimulation events from a _events.tsv file

    Args:
        filepath (str):                           The path to the _events.tsv file to load
        additional_required_columns(list/tuple):  One or multiple additional columns that need to be present in the _events.tsv

    Returns:
        trial_onsets (list)                       A list with the onsets of the stimulus events
        trial_pairs (list)                        A list with the stim-pair names of each stimulus events
        trials_bad_onsets                         If a status column exists in the events file, this list holds the onsets
                                                  of the trials that were marked as 'bad' and not included

    Raises:
        RuntimeError:                             If the file could not be found, or if the mandatory 'onset', 'trial_type',
                                                  'electrical_stimulation_site' column or any of the required additional
                                                   columns could not be found

    Note:   This function expects the column 'trial_type' and 'electrical_stimulation_site' to exist in the _events.tsv file
            according to the BIDS iEEG electrical stimulation specification.
    Note 2: If a column status exists in the _events.tsv file, then these trials marked as 'bad' will be excluded

    """

    # complete list of required columns
    required_columns = ['trial_type', 'electrical_stimulation_site']
    if not additional_required_columns is None:
        for column in additional_required_columns:
            required_columns.append(column)

    # retrieve the stimulation events (onsets and pairs) from the events.tsv file
    try:
        events_tsv = load_event_info(filepath, required_columns)
    except (FileNotFoundError, LookupError):
        logging.error('Could not load the stimulation event metadata (\'' + filepath + '_events.tsv\'), exiting...')
        raise RuntimeError('Could not load the stimulation event metadata')

    # acquire the onset and electrode-pair for each stimulation
    trial_onsets = []
    trial_pairs = []
    trials_bad_onsets = []
    trials_have_status = 'status' in events_tsv.columns
    for index, row in events_tsv.iterrows():
        if row['trial_type'].lower() == 'electrical_stimulation':
            if not is_number(row['onset']) or isnan(float(row['onset'])) or float(row['onset']) < 0:
                logging.warning('Invalid onset \'' + row['onset'] + '\' in events, should be a numeric value >= 0. Discarding trial...')
                continue

            if trials_have_status:
                if not row['status'].lower() == 'good':
                    trials_bad_onsets.append(row['onset'])
                    continue

            pair = row['electrical_stimulation_site'].split('-')
            if not len(pair) == 2 or len(pair[0]) == 0 or len(pair[1]) == 0:
                logging.error('Electrical stimulation site \'' + row['electrical_stimulation_site'] + '\' invalid, should be two values separated by a dash (e.g. CH01-CH02), exiting...')
                raise RuntimeError('Electrical stimulation site invalid')

            trial_onsets.append(float(row['onset']))
            trial_pairs.append(pair)

    return trial_onsets, trial_pairs, trials_bad_onsets


def load_ieeg_sidecar(filepath):
    """
    Read a JSON sidecar file

    Args:
        filepath (str):             The path to the JSON sidecar file to load

    Returns:
        ieeg_json (dict):           A dictionary containing the sidecar information

    Raises:
        IOError:                    If the file could not be found or accessed
        RuntimeError:               If the JSON file could not be parsed
    """

    # try to read the JSON configuration file
    try:
        with open(filepath) as json_file:
            ieeg_json = json.load(json_file)
    except IOError:
        logging.error('Could not access the IEEG JSON sidecar file at \'' + filepath + '\'')
        raise IOError('Could not access the IEEG JSON sidecar file')
    except json.decoder.JSONDecodeError as e:
        logging.error('Could not interpret the IEEG JSON sidecar file at \'' + filepath + '\', make sure the JSON syntax is valid: \'' + str(e) + '\'')
        raise RuntimeError('Could not interpret the IEEG JSON sidecar file')

    #
    return ieeg_json

