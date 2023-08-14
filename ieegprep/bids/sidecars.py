"""
Functions to load BIDS sidecar metadata


=====================================================
Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
from math import isnan
import json
import logging
from ieegprep.utils.misc import is_number


def load_channel_info(filepath):
    """

    Retrieve the channel metadata from a _channels.tsv file

    Args:
        filepath (str):             The path to the _channels.tsv file to load

    Returns:
        metadata (extended dict):   An extended dictionary containing the channels information

    Raises:
        FileNotFoundError:          If the file could not be found
        LookupError:                If the mandatory 'name', 'type' columns or one of the required columns could not be found

    """

    try:
        return BidsIeegChannelTsv.from_file(filepath, interpret_numeric=True)
    except FileNotFoundError:
        logging.error('Could not find the file \'' + filepath + '\'')
        raise FileNotFoundError('Could not find file')


def load_event_info(filepath, addition_required_columns=None):
    """
    Retrieve the events from a _events.tsv file

    Args:
        filepath (str):                         The path to the _events.tsv file to load
        addition_required_columns(list/tuple):  One or multiple additional columns that need to be present in the _events.tsv

    Returns:
        metadata (extended dict):               An extended dictionary containing the events information

    Raises:
        FileNotFoundError:                      If the file could not be found
        LookupError:                            If the mandatory 'onset' column or any of the required additional
                                                columns could not be found
    """

    try:
        return BidsIeegEventsTsv.from_file(filepath, interpret_numeric=True)
    except FileNotFoundError:
        logging.error('Could not find the file \'' + filepath + '\'')
        raise FileNotFoundError('Could not find file')


def load_elec_stim_events(filepath, additional_required_columns=None, exclude_bad_events=True,
                          concat_bidirectional_stimpairs=True, only_stimpairs_between_channels=None):
    """
    Retrieve the electrical stimulation events from a _events.tsv file

    Args:
        filepath (str):                              The path to the _events.tsv file to load
        additional_required_columns(list/tuple):     One or multiple additional columns that need to be present in the _events.tsv
        exclude_bad_events (bool):                   Whether to exclude events marked as 'bad' (only applied if a status column is present)
        concat_bidirectional_stimpairs (bool):       Whether to concatenate events that concern the same two stimulation
                                                     electrodes. If true, stimulation events between - for example - Ch01 and
                                                     Ch02 will be concatenated with stimulation events between Ch02 and Ch01.
        only_stimpairs_between_channels(tuple/list): Only include stimulated pairs when both stimulated electrodes are included
                                                     in this (list/tuple) argument. Set to None to include all pairs.

    Returns:
        trial_onsets (list)                          A list with the onsets of the stimulus events
        trial_pairs (list)                           A list with the stim-pair names of each stimulus events
        stimpair_onsets (dict)                       A dictionary that holds the onsets for each distinct stimulated electrode
                                                     pair. Each stimulated pair is an entry (e.g. Ch01-Ch02), and each entry
                                                     contains the corresponding onsets times as a list.
        bad_trial_onsets                             If a status column exists in the events file, this list holds the onsets
                                                     of the trials that were marked as 'bad' and not included

    Raises:
        RuntimeError:                                If the file could not be found, or if the mandatory 'onset', 'trial_type',
                                                     'electrical_stimulation_site' column or any of the required additional
                                                     columns could not be found

    Note:   This function expects the column 'trial_type' and 'electrical_stimulation_site' to exist in the _events.tsv file
            according to the BIDS iEEG electrical stimulation specification. Events of which the 'trial_type' are labelled
            as 'electrical_stimulation' are regarded as electrical stimulation events. The 'electrical_stimulation_site' of
            each stimulation event should indicate the stimulated electrodes separated by a dash, as such: Ch01-Ch02.

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
    bad_trial_onsets = []
    trials_have_status = 'status' in events_tsv.columns
    for index, row in events_tsv.iterrows():
        if row['trial_type'].lower() == 'electrical_stimulation':
            if not is_number(row['onset']) or isnan(float(row['onset'])) or float(row['onset']) < 0:
                logging.warning('Invalid onset \'' + row['onset'] + '\' in events, should be a numeric value >= 0. Discarding trial...')
                continue

            if exclude_bad_events and trials_have_status:
                if not row['status'].lower() == 'good':
                    bad_trial_onsets.append(row['onset'])
                    continue

            pair = row['electrical_stimulation_site'].split('-')
            if not len(pair) == 2 or len(pair[0]) == 0 or len(pair[1]) == 0:
                logging.error('Electrical stimulation site \'' + row['electrical_stimulation_site'] + '\' invalid, should be two values separated by a dash (e.g. CH01-CH02), exiting...')
                raise RuntimeError('Electrical stimulation site invalid')

            trial_onsets.append(float(row['onset']))
            trial_pairs.append(pair)


    # dictionary to hold the stimulated electrode pairs and their onsets
    stimpairs_onsets = dict()

    #
    if only_stimpairs_between_channels is None:
        # include all stim-pairs

        for trial_index in range(len(trial_pairs)):
            trial_pair = trial_pairs[trial_index]
            if concat_bidirectional_stimpairs and (trial_pair[1] + '-' + trial_pair[0]) in stimpairs_onsets.keys():
                stimpairs_onsets[trial_pair[1] + '-' + trial_pair[0]].append(trial_onsets[trial_index])
            else:
                if (trial_pair[0] + '-' + trial_pair[1]) in stimpairs_onsets.keys():
                    stimpairs_onsets[trial_pair[0] + '-' + trial_pair[1]].append(trial_onsets[trial_index])
                else:
                    stimpairs_onsets[trial_pair[0] + '-' + trial_pair[1]] = [trial_onsets[trial_index]]

    else:

        # loop over all the combinations of included channels
        # Note:     only the combinations of stim-pairs that actually have events/trials end up in the output
        for iChannel0 in range(len(only_stimpairs_between_channels)):
            for iChannel1 in range(len(only_stimpairs_between_channels)):

                # retrieve the indices of all the trials that concern this stim-pair
                indices = []
                if concat_bidirectional_stimpairs:
                    # allow concatenation of bidirectional pairs, pair order does not matter
                    if not iChannel1 < iChannel0:
                        # unique pairs while ignoring pair order
                        indices = [i for i, x in enumerate(trial_pairs) if
                                   (x[0] == only_stimpairs_between_channels[iChannel0] and x[1] == only_stimpairs_between_channels[iChannel1]) or (x[0] == only_stimpairs_between_channels[iChannel1] and x[1] == only_stimpairs_between_channels[iChannel0])]

                else:
                    # do not concatenate bidirectional pairs, pair order matters
                    indices = [i for i, x in enumerate(trial_pairs) if
                               x[0] == only_stimpairs_between_channels[iChannel0] and x[1] == only_stimpairs_between_channels[iChannel1]]

                # add the pair if there are trials for it
                if len(indices) > 0:
                    stimpairs_onsets[only_stimpairs_between_channels[iChannel0] + '-' + only_stimpairs_between_channels[iChannel1]] = [trial_onsets[i] for i in indices]

    # success, return the results
    return trial_onsets, trial_pairs, stimpairs_onsets, bad_trial_onsets


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


#
# tsv helper classes
#

class BidsTsv(dict):
    """
    An extension on the build-in Python dictionary to hold tsv information with some convenience functions added
    """

    def read_from_file(self, filepath, required_columns=None, interpret_numeric=True):
        """
        Read the metadata from a tsv file

        Args:
            filepath (str):                           The path to the _events.tsv file to load
            required_columns (list/tuple):            One or multiple columns that need to be present in the tsv file
            interpret_numeric (bool):                 Whether to try to interpret column values as numeric

        Returns:
            metadata (dictionary):                    A extended dictionary containing the information

        Raises:
            FileNotFoundError:                        If the file could not be found
            LookupError:                              If any of the required columns could not be found
        """
        # check file existence
        if not os.path.exists(filepath):
            logging.error('Tsv file \'' + filepath + '\' could not be found')
            raise FileNotFoundError('No such file or directory: \'' + filepath + '\'')

        # BIDS tsv files should not be large, therefore read the entire file in one go.
        # This has the advantage of being able to allocate memory for the values for each of the columns
        input = open(filepath, mode="r", encoding="utf-8")
        rawText = input.read()
        input.close()

        # split the lines, remove empty lines, and ensure there are columns in the first line
        lines = [line for line in rawText.split("\n") if line]
        if len(lines) < 1:
            logging.error('Empty tsv file, make sure at least the required column names are present in the first line of the file')
            raise LookupError('Empty tsv file')

        # extract the columns and check if all required columns are there
        columns = lines[0].split("\t")
        if required_columns is not None:
            for required_column in required_columns:
                if required_column not in columns:
                    logging.error('Could not find the required column \'' + required_column + '\' in \'' + filepath + '\'')
                    raise LookupError('Could not find required column in tsv file')

        # initialize the lists according to the columns
        if not interpret_numeric:
            for column in columns:
                self[column] = [None] * (len(lines) - 1)

        else:

            # column is assumed numeric until proven false
            column_is_numeric = [True] * len(columns)

            # keep a copy (speed over small amount of memory) for each column, so the values both as strings or numbers
            # Note: because each value needs to be tested to be numeric and we don't want to do this twice (test-convert and
            #       later convert), we test and convert only once. And afterwards choose which version to set for the dictionary
            column_values_as_numeric = [([None] * (len(lines) - 1)) for x in range(len(columns))]
            column_values_as_string = [([None] * (len(lines) - 1)) for x in range(len(columns))]


        # loop over each line (skipping the header)
        # TODO: could also loop over columns, rather than rows.
        for row_index, line in enumerate(lines[1:]):

            # separate the values by delimiter and check if the number of values matches the header
            row_values = line.split('\t')
            if len(row_values) != len(columns):
                logging.error('Row ' + str(row_index + 1) + ' in \'' + filepath + '\' does not contain the same number of values as the header dictates (' + str(len(columns)) + ')')
                raise LookupError('Number of values in row does not match header in tsv file')

            # store the values in the row in their appropriate dictionaries (matching the order of the columns)
            if not interpret_numeric:
                for column_index, column_name in enumerate(columns):
                    self[column_name][row_index] = row_values[column_index]

            else:
                for column_index, column_name in enumerate(columns):

                    # always store as string (a column can turn out to be non-numeric at any point)
                    column_values_as_string[column_index][row_index] = row_values[column_index]

                    # check if the column can still be numeric
                    if column_is_numeric[column_index]:

                        # check value to be numeric (or nan, NaN , n/a)
                        if row_values[column_index].lower() == 'n/a':
                            column_values_as_numeric[column_index][row_index] = float('nan')
                        else:
                            try:
                                column_values_as_numeric[column_index][row_index] = float(row_values[column_index])
                            except:

                                # flag as non-numeric and clear the
                                column_is_numeric[column_index] = False
                                column_values_as_numeric[column_index] = None

        # if columns and values are interpreted to be numeric than set the correct version (string or numeric list) to the dictionary
        if interpret_numeric:
            for column_index, column_name in enumerate(columns):
                if column_is_numeric[column_index]:
                    self[column_name] = column_values_as_numeric[column_index]
                    column_values_as_string[column_index] = None
                else:
                    self[column_name] = column_values_as_string[column_index]
                    column_values_as_numeric[column_index] = None

        # success
        return self

    def get_columns(self):
        return self.keys()
    columns = property(get_columns)

    def iterrows(self):
        if len(self.keys()) == 0:
            return
            yield

        num_rows = len(self[list(self.keys())[0]])
        if num_rows == 0:
            return
            yield

        for i in range(num_rows):
            yield i, self.row_by_index(i)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if len(self.keys()) == 0:
            raise StopIteration

        num_rows = len(self[list(self.keys())[0]])
        if num_rows == 0:
            raise StopIteration

        if self.iter_index < num_rows:
            self.iter_index += 1
            return self.row_by_index(self.iter_index - 1)
        else:
            raise StopIteration


    def row_by_index(self, row_index):
        row = dict()
        for column_name in self.keys():
            row[column_name] = self[column_name][row_index]
        return row

    def row_by_index_as_list(self, row_index):
        row = [None] * len(self.keys())
        for column_index, column_name in enumerate(self.keys()):
            row[column_index] = self[column_name][row_index]
        return row

    def _find_row_index_by_value(self, find_column_name, find_value):
        for column_name in self.keys():
            if column_name == find_column_name:
                for row_index, row_value in enumerate(self[column_name]):
                    if row_value == find_value:
                        return row_index
        return None

    def row_by_value(self, find_column_name, find_value):
        index = self._find_row_index_by_value(find_column_name, find_value)
        return self.row_by_index(index) if not index is None else None

    def row_by_value_as_list(self, find_column_name, find_value):
        index = self._find_row_index_by_value(find_column_name, find_value)
        return self.row_by_index_as_list(index) if not index is None else None


class BidsIeegChannelTsv(BidsTsv):
    """
    An extension on the build-in Python dictionary to hold channel metadata with some convenience functions added
    """

    def __init__(self):
        self['name'] = []
        self['type'] = []

    @classmethod
    def from_file(cls, filepath, additional_required_columns=None, interpret_numeric=True):
        required_columns = ['name', 'type']
        if not additional_required_columns is None:
            for column in additional_required_columns:
                required_columns.append(column)

        # read tsv and return
        return cls().read_from_file(filepath, required_columns=required_columns, interpret_numeric=interpret_numeric)


    def row_by_name(self, channel_name):
        row = self.row_by_value('name', channel_name)
        if row is not None:
            return row
        else:
            raise LookupError('Could not find row by channel name')

    def row_by_name_as_list(self, channel_name):
        row = self.row_by_value_as_list('name', channel_name)
        if row is not None:
            return row
        else:
            raise LookupError('Could not find row by channel name')


class BidsIeegEventsTsv(BidsTsv):
    """
    An extension on the build-in Python dictionary to hold event metadata with some convenience functions added
    """

    def __init__(self):
        self['onset'] = []

    @classmethod
    def from_file(cls, filepath, additional_required_columns=None, interpret_numeric=True):
        required_columns = ['onset']
        if not additional_required_columns is None:
            for column in additional_required_columns:
                required_columns.append(column)
        return cls().read_from_file(filepath, required_columns=required_columns, interpret_numeric=interpret_numeric)
