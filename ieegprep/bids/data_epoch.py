"""
Functions to load and epoch BIDS data
=====================================================


Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import gc
import logging
import warnings
import numpy as np
from scipy import signal

from ieegprep.fileio.IeegDataReader import IeegDataReader, VALID_FORMAT_EXTENSIONS
from ieegprep.utils.misc import allocate_array
from ieegprep.utils.console import multi_line_list, print_progressbar
LOGGING_CAPTION_INDENT_LENGTH   = 50      # TODO: also in erdetect.core.config


def load_data_epochs(data_path, retrieve_channels, onsets,
                     trial_epoch=(-1, 3), baseline_norm=None, baseline_epoch=(-1, -0.1),
                     out_of_bound_handling='error',
                     high_pass=False, early_reref=None, line_noise_removal=None, late_reref=None, preproc_priority='mem'):
    """
    Load and epoch the data into a matrix based on channels, the trial onsets and the epoch range (relative to the onsets)

    Args:
        data_path (str):                    Path to the data file or folder
        retrieve_channels (list or tuple):  The channels (by name) of which the data should be retrieved, the output will
                                            be sorted accordingly according to this input argument.
        onsets (1d list or tuple):          The onsets of the trials around which to epoch the data
        trial_epoch (tuple):                The time-span that will be considered as the signal belonging to a single trial.
                                            Expressed as a tuple with the start- and end-point in seconds relative to
                                            the onset of the trial (e.g. the standard tuple of '-1, 3' will extract
                                            the signal in the period from 1s before the trial onset to 3s after trial onset).
        baseline_norm (None or str):        Baseline normalization setting [None, 'Mean' or 'Median']. If other than None,
                                            normalizes each trial epoch by subtracting the mean or median of part of the
                                            trial (the epoch of the trial indicated in baseline_epoch)
        baseline_epoch (tuple):             The time-span on which the baseline is calculated, expressed as a tuple with the
                                            start- and end-point in seconds relative to the trial onset (e.g. the
                                            standard tuple of '-1, -.1' will use the period from 1s before trial onset
                                            to 100ms before trial onset to calculate the baseline on); this argument
                                            is only used when baseline_norm is set to mean or median
        out_of_bound_handling (str):        Configure the handling of out-of-bound trial epochs;
                                                'error': (default) Throw an error and return when any epoch is out of bound;
                                                'first_last_only': Allows only the first trial epoch to start before the
                                                                   data-set and the last trial epoch to end beyond the
                                                                   length of the data-set, the trial epochs will be padded
                                                                   with NaN values. Note that the first and last trial are
                                                                   determined by the first and last entry in the 'onsets'
                                                                   parameter, which is not sorted by this function;
                                                'allow':           Allow trial epochs to be out-of-bound, NaNs values will
                                                                   be used for part of, or the entire, the trial epoch
        high_pass (bool):                   Preprocess with high-pass filtering (true) or without filtering (false)
        early_reref (None or RerefStruct):  Preprocess with early (before line-noise removal) re-referencing. Generate a
                                            RerefStruct instance using one of the factory methods (e.g. generate_car) and
                                            pass it here to allow for re-referencing. Pass None to skip early re-referencing.
        line_noise_removal (None or int):   Whether to preprocess with line-noise removal. If an integer (e.g. 50 or 60) is
                                            passed, then a notch filter will be applied around that number. Passing None
                                            will disable line-noise removal.
        late_reref (None or RerefStruct):   Preprocess with late (after line-noise removal) re-referencing. Generate a
                                            RerefStruct instance using one of the factory methods (e.g. generate_car) and
                                            pass it here to allow for re-referencing. Pass None to skip late re-referencing.
        preproc_priority (str):             Set the preprocessing priority, can be set to either 'mem' (default) or 'speed'

    Returns:
        sampling_rate (int or double):      the sampling rate at which the data was acquired
        data (ndarray):                     A three-dimensional array with data epochs per channel (format: channel x
                                            trials/epochs x time); or None when an error occurs

    Note: this function's input arguments are in seconds relative to the trial onsets because the sample rate will
          only be known till after we read the data
    """

    #
    # check input
    #
    try:
        data_reader, baseline_method, out_of_bound_method = __prepare_input(data_path,
                                                                            trial_epoch, baseline_norm, baseline_epoch,
                                                                            out_of_bound_handling, preproc_priority)
        # TODO: check preprocessing input

    except Exception as e:
        logging.error('Error preparing input: ' + str(e))
        raise RuntimeError('Error preparing input')

    #
    # read and process the data
    #
    try:

        # check whether preprocessing is needed (full channel loading)
        if high_pass or early_reref is not None or line_noise_removal is not None or late_reref is not None:
            # require preprocessing

            # Load data epoch averages by iterating over the channels
            # Note:   with preprocessing per channel manipulations are needed before epoch-ing (,metric calculation) and averaging
            sampling_rate, data = __load_data_epochs__by_channels__withPrep(False, data_reader, retrieve_channels, onsets,
                                                                            trial_epoch=trial_epoch,
                                                                            baseline_method=baseline_method, baseline_epoch=baseline_epoch,
                                                                            out_of_bound_method=out_of_bound_method,
                                                                            metric_callbacks=None,
                                                                            high_pass=high_pass,
                                                                            early_reref=early_reref,
                                                                            line_noise_removal=line_noise_removal,
                                                                            late_reref=late_reref,
                                                                            priority=preproc_priority)

        else:
            # no preprocessing required

            if data_reader.data_format in ('bv', 'edf'):
                # EDF or BrainVision format
                # TODO: test speeds and depend on preload (not on datatype)

                # load the data by iterating over the channels and picking out the epochs, for entire preloaded datasets
                # this is a reasonable options since the data is already in memory

                sampling_rate, data = __load_data_epochs__by_channels( data_reader, retrieve_channels, onsets,
                                                                       trial_epoch=trial_epoch,
                                                                       baseline_method=baseline_method, baseline_epoch=baseline_epoch,
                                                                       out_of_bound_method=out_of_bound_method)

            elif data_reader.data_format == 'mef3':
                # MEF3 format
                # TODO: test speeds and depend on preload (not on datatype)

                # load the data by iterating over the trials, for non-preloaded data this is the most memory efficient (and likely fastest)
                sampling_rate, data = __load_data_epochs__by_trial(data_reader, retrieve_channels, onsets,
                                                                   trial_epoch=trial_epoch,
                                                                   baseline_method=baseline_method, baseline_epoch=baseline_epoch,
                                                                   out_of_bound_method=out_of_bound_method)

    except Exception as e:
        logging.error('Error on loading and epoching data: ' + str(e))
        raise RuntimeError('Error on loading and epoching data')

    # close (and unload) the data reader
    data_reader.close()

    #
    return sampling_rate, data


def load_data_epochs_averages(data_path, retrieve_channels, conditions_onsets,
                              trial_epoch=(-1, 3), baseline_norm=None, baseline_epoch=(-1, -0.1),
                              out_of_bound_handling='error', metric_callbacks=None,
                              high_pass=False, early_reref=None, line_noise_removal=None, late_reref=None, preproc_priority='mem'):


    """
    Load, epoch and return the average for each channel and condition (i.e. the signal in time averaged
    over all trials that belong to the same condition).

    Note: Because this function only has to return the average signal for each channel and condition, it is much more
          memory efficient (this is particularly important when the amount of memory is limited by a Docker or VM)
    Note 2: For the same reason, metric callbacks are implemented here, so while loading, but before averaging, metrics
            can be calculated on subsets of data with minimum memory usage. If memory is not an issue,
            load_data_epochs function can be used to retrieve the full dataset first and then perform calculations.

    Args:
        data_path (str):                      Path to the data file or folder
        retrieve_channels (list or tuple):    The channels (by name) of which the data should be retrieved, the output
                                              will be sorted accordingly
        conditions_onsets (2d list or tuple): A two-dimensional list to indicate the conditions, and the onsets
                                              of the trials that belong to each condition.
                                              (format: conditions x condition onsets)
        trial_epoch (tuple):                  The time-span that will be considered as the signal belonging to a single
                                              trial. Expressed as a tuple with the start- and end-point in seconds
                                              relative to onset of the trial (e.g. the standard tuple of '-1, 3' will
                                              extract the signal in the period from 1s before the trial onset to 3s
                                              after the trial onset).
        baseline_norm (None or str):          Baseline normalization setting [None, 'Mean' or 'Median']. If other
                                              than None, normalizes each trial epoch by subtracting the mean or median
                                              of part of the trial (the epoch of the trial indicated in baseline_epoch)
        baseline_epoch (tuple):               The time-span on which the baseline is calculated, expressed as a tuple with
                                              the start- and end-point in seconds relative to the trial onset (e.g. the
                                              standard tuple of '-1, -.1' will use the period from 1s before the trial
                                              onset to 100ms before the trial onset to calculate the baseline on);
                                              this argument is only used when baseline_norm is set to mean or median
        out_of_bound_handling (str):          Configure the handling of out-of-bound trial epochs;
                                                 'error': (default) Throw an error and return when any epoch is out of bound;
                                                 'first_last_only': Allows only the first trial epoch to start before the
                                                                    data-set and the last trial epoch to end beyond the
                                                                    length of the data-set, the trial epochs will be padded
                                                                    with NaN values. Note that the first and last trial are
                                                                    determined by the first and last entry in the 'onsets'
                                                                    parameter, which is not sorted by this function;
                                                 'allow':           Allow trial epochs to be out-of-bound, NaNs values will
                                                                    be used for part of, or the entire, the trial epoch
        metric_callbacks (func or tuple):     Function or tuple of functions that are called to calculate metrics based
                                              on subsets of the un-averaged data. The function(s) are called per
                                              with the following input arguments:
                                                 sampling_rate -    The sampling rate of the data
                                                 data -             A subset of the data in a 2d array: trials x samples
                                                 baseline -         The corresponding baseline values for the trials
                                              If callbacks are defined, a third variable is returned that holds the
                                              return values of the metric callbacks in the format: channel x condition x metric
        high_pass (bool):                     Preprocess with high-pass filtering (true) or without filtering (false)
        early_reref (None or RerefStruct):    Preprocess with early (before line-noise removal) re-referencing. Generate a
                                              RerefStruct instance using one of the factory methods (e.g. generate_car) and
                                              pass it here to allow for re-referencing. Pass None to skip early re-referencing.
        line_noise_removal (None or int):     Whether to preprocess with line-noise removal. If an integer (e.g. 50 or 60) is
                                              passed, then a notch filter will be applied around that number. Passing None
                                              will disable line-noise removal.
        late_reref (None or RerefStruct):     Preprocess with late (after line-noise removal) re-referencing. Generate a
                                              RerefStruct instance using one of the factory methods (e.g. generate_car) and
                                              pass it here to allow for re-referencing. Pass None to skip late re-referencing.
        preproc_priority (str):               Set the preprocessing priority, can be set to either 'mem' (default) or 'speed'

    Returns:
        sampling_rate (int or double):        The sampling rate at which the data was acquired
        data (ndarray):                       A three-dimensional array with signal averages per channel and condition
                                              (format: channel x condition x samples); or None when an error occurs
        metrics (ndarray):                    If metric callbacks are specified, will return a three-dimensional array
                                              with the metric callback results (format: channel x condition x metric),
                                              else wise None

    Note: this function input arguments are in seconds relative to trial onsets because the sample rate will
          only be known till after we read the data
    """

    #
    # check input
    #
    try:
        data_reader, baseline_method, out_of_bound_method = __prepare_input(data_path,
                                                                            trial_epoch, baseline_norm, baseline_epoch,
                                                                            out_of_bound_handling, preproc_priority)
        # TODO: check preprocessing input

    except Exception as e:
        logging.error('Error preparing input: ' + str(e))
        raise RuntimeError('Error preparing input')


    #
    # read and process the data
    #
    try:

        # check whether preprocessing is needed (full channel loading)
        if high_pass or early_reref is not None or line_noise_removal is not None or late_reref is not None:
            # require preprocessing

            # Load data epoch averages by iterating over the channels
            # Note:   with preprocessing per channel manipulations are needed before epoch-ing (,metric calculation) and averaging
            sampling_rate, data, metric_values = __load_data_epochs__by_channels__withPrep(True, data_reader, retrieve_channels, conditions_onsets,
                                                                                           trial_epoch=trial_epoch,
                                                                                           baseline_method=baseline_method, baseline_epoch=baseline_epoch,
                                                                                           out_of_bound_method=out_of_bound_method,
                                                                                           metric_callbacks=metric_callbacks,
                                                                                           high_pass=high_pass,
                                                                                           early_reref=early_reref,
                                                                                           line_noise_removal=line_noise_removal,
                                                                                           late_reref=late_reref,
                                                                                           priority=preproc_priority)

        else:
            # no preprocessing required

            if data_reader.data_format in ('bv', 'edf'):
                # EDF or BrainVision format
                # TODO: test speeds and depend on preload (not on datatype)

                # Load data epoch averages by first iterating over conditions, then over the channels and then retrieve
                # and average (metric) over the epoch-trials within the channel-condition combination
                #
                # Note:     This method is good if the entire dataset is preloaded in memory. So there is no minimum of loading of data possible.
                sampling_rate, data, metric_values = __load_data_epoch_averages__by_channel_condition_trial(data_reader, retrieve_channels, conditions_onsets,
                                                                                                            trial_epoch=trial_epoch,
                                                                                                            baseline_method=baseline_method, baseline_epoch=baseline_epoch,
                                                                                                            out_of_bound_method=out_of_bound_method, metric_callbacks=metric_callbacks)

            elif data_reader.data_format == 'mef3':
                # MEF3 format
                # TODO: test speeds and depend on preload (not on datatype)

                # load the data by first iterating over conditions, second over trials within that condition and then
                # retrieve the epoch-data for all channels and take average (and metric) for each channel.
                #
                # For MEF3 this is the fastest solution while using a small amount of memory (because only the required data is loaded)
                sampling_rate, data, metric_values = __load_data_epoch_averages__by_condition_trial(data_reader, retrieve_channels, conditions_onsets,
                                                                                                    trial_epoch=trial_epoch,
                                                                                                    baseline_method=baseline_method, baseline_epoch=baseline_epoch,
                                                                                                    out_of_bound_method=out_of_bound_method, metric_callbacks=metric_callbacks)

    except Exception as e:
        logging.error('Error on loading, epoching and averaging data: ' + str(e))
        raise RuntimeError('Error on loading, epoching and averaging data')

    # close (and unload) the data reader
    data_reader.close()

    # return success
    return sampling_rate, data, metric_values



#
# private functions
#

def __prepare_input(data_path, trial_epoch, baseline_norm, baseline_epoch, out_of_bound_handling, preproc_priority):
    """
    Check and prepare the input for loading data

    preproc_priority (str):              Set the preprocessing priority, can be set to either 'mem' (default) or 'speed'
    """

    # data-set format
    data_extension = data_path[data_path.rindex("."):]
    if not any(data_extension in x for x in VALID_FORMAT_EXTENSIONS):
        logging.error('Unknown data format (' + data_extension + ')')
        raise ValueError('Unknown data format (' + data_extension + ')')

    # check trial epoch input
    if trial_epoch[1] < trial_epoch[0]:
        logging.error('Invalid \'trial_epoch\' parameter, the given end-point (at ' + str(trial_epoch[1]) + ') lies before the start-point (at ' + str(trial_epoch[0]) + ')')
        raise ValueError('Invalid \'trial_epoch\' parameter')

    # create and initialize an IEEG data-reader instance to manage the data.
    try:
        data_reader = IeegDataReader(data_path, preproc_priority == 'speed')
    except ValueError:
        raise ValueError('Error upon constructing a data reader')
    except RuntimeError:
        raise RuntimeError('Error upon initializing a data reader')

    # baseline normalization
    baseline_method = 0
    if baseline_norm is not None and len(baseline_norm) > 0:
        if baseline_norm.lower() == 'mean' or baseline_norm.lower() == 'average':
            baseline_method = 1
        elif baseline_norm.lower() == 'median':
            baseline_method = 2
        elif baseline_norm.lower() == 'none':
            baseline_method = 0
        else:
            logging.error('Unknown normalization argument (' + baseline_norm + '), this can only be one of the following options: None, \'mean\' or \'median\'')
            raise ValueError('Unknown normalization argument')

        #
        if baseline_epoch[1] < baseline_epoch[0]:
            logging.error('Invalid \'baseline_epoch\' parameter, the given end-point (at ' + str(baseline_epoch[1]) + ') lies before the start-point (at ' + str(baseline_epoch[0]) + ')')
            raise ValueError('Invalid \'baseline_epoch\' parameter')

        # TODO: check mef3 baseline in trial, might not be a restriction for all epoching routines
        if data_reader.data_format == 'mef3':
            if baseline_epoch[0] < trial_epoch[0]:
                logging.error('Invalid \'baseline_epoch\' parameter, the given baseline start-point (at ' + str(baseline_epoch[0]) + ') lies before the trial start-point (at ' + str(trial_epoch[0]) + ')')
                raise ValueError('Invalid \'baseline_epoch\' parameter')
            if baseline_epoch[1] > trial_epoch[1]:
                logging.error('Invalid \'baseline_epoch\' parameter, the given baseline end-point (at ' + str(baseline_epoch[1]) + ') lies after the trial end-point (at ' + str(trial_epoch[1]) + ')')
                raise ValueError('Invalid \'baseline_epoch\' parameter')

    # out-of-bound handling
    if out_of_bound_handling.lower() == 'first_last_only':
        out_of_bound_method = 1
    elif out_of_bound_handling.lower() == 'allow':
        out_of_bound_method = 2
    elif out_of_bound_handling.lower() == 'error':
        out_of_bound_method = 0
    else:
        logging.error('Unknown out-of-bound handling argument (' + out_of_bound_handling + '), this can only be one of the following options: \'error\', \'first_last_only\' or \'allow\'')
        raise ValueError('Unknown out-of-bound handling argument')

    return data_reader, baseline_method, out_of_bound_method


def __epoch_data__from_channel_data__by_trials(ref_data, channel_idx, channel_data, sampling_rate,
                                               onsets, trial_epoch,
                                               baseline_method, baseline_epoch, out_of_bound_method):
    """
    Epoch the trial-data for a single channel by looping over the trial-onsets
    """

    # calculate the size of the time dimension (in samples)
    trial_num_samples = int(round(abs(trial_epoch[1] - trial_epoch[0]) * sampling_rate))
    baseline_num_samples = int(round(abs(baseline_epoch[1] - baseline_epoch[0]) * sampling_rate))

    # loop through the trials
    for trial_idx in range(len(onsets)):

        # calculate the sample indices
        trial_sample_start = int(round((onsets[trial_idx] + trial_epoch[0]) * sampling_rate))
        trial_sample_end = trial_sample_start + trial_num_samples
        baseline_start_sample = int(round((onsets[trial_idx] + baseline_epoch[0]) * sampling_rate))
        baseline_end_sample = baseline_start_sample + baseline_num_samples
        local_start = 0
        local_end = trial_num_samples

        # check whether the trial epoch is within bounds
        if trial_sample_end < 0:
            if (out_of_bound_method == 1 and trial_idx == 0) or out_of_bound_method == 2:
                if channel_idx == 0:
                    logging.warning('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the end of the trial-epoch lies before the start of the data-set.')
                continue
            else:
                logging.error('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the end of the trial-epoch lies before the start of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                raise RuntimeError('Cannot extract trial')
        if trial_sample_start < 0:
            if (out_of_bound_method == 1 and trial_idx == 0) or out_of_bound_method == 2:
                if channel_idx == 0:
                    logging.warning('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the start of the trial-epoch lies before the start of the data-set.')
                local_start = trial_sample_start * -1
                trial_sample_start = 0
            else:
                logging.error('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the start of the trial-epoch lies before the start of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                raise RuntimeError('Cannot extract trial')
        if trial_sample_start > channel_data.size:
            if (out_of_bound_method == 1 and trial_idx == len(onsets) - 1) or out_of_bound_method == 2:
                if channel_idx == 0:
                    logging.warning('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the start of the trial-epoch lies after the end of the data-set.')
                continue
            else:
                logging.error('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the start of the trial-epoch lies after the end of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                raise RuntimeError('Cannot extract trial')
        if trial_sample_end > channel_data.size:
            if (out_of_bound_method == 1 and trial_idx == len(onsets) - 1) or out_of_bound_method == 2:
                if channel_idx == 0:
                    logging.warning('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the end of the trial-epoch lies after the end of the data-set.')
                local_end = trial_num_samples - (trial_sample_end - channel_data.size)
                trial_sample_end = channel_data.size
            else:
                logging.error('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the end of the trial-epoch lies after the end of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                raise RuntimeError('Cannot extract trial')

        # check whether the baseline is within bounds
        if baseline_method > 0:
            if baseline_start_sample < 0 or baseline_end_sample > channel_data.size:
                logging.error('Cannot extract the baseline for the trial with onset ' + str(onsets[trial_idx]) + ', the range for the baseline lies outside of the data')
                raise RuntimeError('Cannot extract the baseline')

        # extract the trial data and perform baseline normalization on the trial if needed
        if baseline_method == 0:

            # Note: since we are not manipulating the data (which in the other cases converts a view to data), always
            #       make a copy. Even if the channel input is a data array (and not a view), it might be possible that
            #       epochs overlap; in addition, avoiding views ensures there are no remaining references to the source
            #       numpy-array, allowing it to be cleared from memory
            ref_data[channel_idx, trial_idx, local_start:local_end] = channel_data[trial_sample_start:trial_sample_end].copy()

        elif baseline_method == 1:
            baseline_mean = np.nanmean(channel_data[baseline_start_sample:baseline_end_sample])
            ref_data[channel_idx, trial_idx, local_start:local_end] = channel_data[trial_sample_start:trial_sample_end] - baseline_mean
        elif baseline_method == 2:
            baseline_median = np.nanmedian(channel_data[baseline_start_sample:baseline_end_sample])
            ref_data[channel_idx, trial_idx, local_start:local_end] = channel_data[trial_sample_start:trial_sample_end] - baseline_median

    # return success
    return ref_data


def __load_data_epochs__by_channels(data_reader, retrieve_channels,
                                    onsets, trial_epoch,
                                    baseline_method, baseline_epoch, out_of_bound_method):
    """
    Load data epochs to a matrix (format: channel x trials/epochs x time) by iterating over and loading data per channel
    and retrieving the trial-epochs

    Note:   Since this method retrieves the data of a single channel before extracting the epochs, it is reasonably memory
            efficient. It is well suited when the entire dataset is pre-loaded in memory anyway. However,
            when the entire set is not in memory, epoching can be performed even more memory efficient using
            the '__load_data_epochs__by_trial' method (which should be equally fast or even faster)
            # TODO: test speeds

    Args:
        data_reader (IeegDataReader):       An instance of the IeegDataReader to retrieve metadata and channel data
        retrieve_channels (list or tuple):  The channels (by name) of which the data should be retrieved, the output
                                            will be sorted accordingly
    """

    # calculate the size of the time dimension (in samples)
    trial_num_samples = int(round(abs(trial_epoch[1] - trial_epoch[0]) * data_reader.sampling_rate))
    baseline_num_samples = int(round(abs(baseline_epoch[1] - baseline_epoch[0]) * data_reader.sampling_rate))

    # initialize a data buffer (channel x trials/epochs x time)
    try:
        data = allocate_array((len(retrieve_channels), len(onsets), trial_num_samples))
    except MemoryError:
        raise MemoryError('Not enough memory create a data output matrix')

    # loop through the included channels
    for channel_idx in range(len(retrieve_channels)):

        try:

            # retrieve the channel data
            channel_data = data_reader.retrieve_channel_data(retrieve_channels[channel_idx], False)

            # epoch the channel data
            __epoch_data__from_channel_data__by_trials(data,
                                                      channel_idx, channel_data, data_reader.sampling_rate,
                                                      onsets, trial_epoch,
                                                      baseline_method, baseline_epoch, out_of_bound_method)
        except RuntimeError:
            raise RuntimeError('Error upon loading and epoching data')

        #
        del channel_data

    # return the sample rate and the epoched data
    return data_reader.sampling_rate, data


def __load_data_epochs__by_trial(data_reader, retrieve_channels,
                                 onsets, trial_epoch,
                                 baseline_method, baseline_epoch, out_of_bound_method):
    """
    Load data epochs to a matrix (format: channel x trials/epochs x time) by looping over and loading data per
    trial (for all channels) and retrieving the trial data by iterating over each of the channels

    Note:   Especially when the entire set is not held in memory this is the most memory efficient because only the
            minimum amount of data is loaded into memory, which could also be faster because less data is read from the disk.
            When the entire dataset is first loaded in memory then there no benefit
            # TODO: test speeds

    Args:
        data_reader (IeegDataReader):       An instance of the IeegDataReader to retrieve metadata and channel data
        retrieve_channels (list or tuple):  The channels (by name) of which the data should be retrieved, the output
                                            will be sorted accordingly
    """

    # calculate the size of the time dimension (in samples)
    trial_num_samples = int(round(abs(trial_epoch[1] - trial_epoch[0]) * data_reader.sampling_rate))
    baseline_num_samples = int(round(abs(baseline_epoch[1] - baseline_epoch[0]) * data_reader.sampling_rate))

    # initialize a data buffer (channel x trials/epochs x time)
    try:
        data = allocate_array((len(retrieve_channels), len(onsets), trial_num_samples))
    except MemoryError:
        raise MemoryError('Not enough memory create a data output matrix')

    # create progress bar
    print_progressbar(0, len(onsets), prefix='Progress:', suffix='Complete', length=50)

    # loop through the trials
    for trial_idx in range(len(onsets)):

        #
        trial_sample_start = int(round((onsets[trial_idx] + trial_epoch[0]) * data_reader.sampling_rate))
        trial_sample_end = trial_sample_start + trial_num_samples
        baseline_start_sample = int(round((onsets[trial_idx] + baseline_epoch[0]) * data_reader.sampling_rate)) - trial_sample_start
        baseline_end_sample = baseline_start_sample + baseline_num_samples
        local_start = 0
        local_end = trial_num_samples

        # check whether the trial epoch is within bounds
        if trial_sample_end < 0:
            if (out_of_bound_method == 1 and trial_idx == 0) or out_of_bound_method == 2:
                logging.warning('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the end of the trial-epoch lies before the start of the data-set.')
                continue
            else:
                logging.error('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the end of the trial-epoch lies before the start of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                raise RuntimeError('Cannot extract trial')
        if trial_sample_start < 0:
            if (out_of_bound_method == 1 and trial_idx == 0) or out_of_bound_method == 2:
                logging.warning('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the start of the trial-epoch lies before the start of the data-set.')
                local_start = trial_sample_start * -1
                trial_sample_start = 0
            else:
                logging.error('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the start of the trial-epoch lies before the start of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                raise RuntimeError('Cannot extract trial')
        if trial_sample_start > data_reader.num_samples:
            if (out_of_bound_method == 1 and trial_idx == len(onsets) - 1) or out_of_bound_method == 2:
                logging.warning('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the start of the trial-epoch lies after the end of the data-set.')
                continue
            else:
                logging.error('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the start of the trial-epoch lies after the end of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                raise RuntimeError('Cannot extract trial')
        if trial_sample_end > data_reader.num_samples:
            if (out_of_bound_method == 1 and trial_idx == len(onsets) - 1) or out_of_bound_method == 2:
                logging.warning('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the end of the trial-epoch lies after the end of the data-set.')
                local_end = trial_num_samples - (trial_sample_end - data_reader.num_samples)
                trial_sample_end = data_reader.num_samples
            else:
                logging.error('Cannot extract the trial with onset ' + str(onsets[trial_idx]) + ', the end of the trial-epoch lies after the end of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                raise RuntimeError('Cannot extract trial')

        # check whether the baseline is within bounds
        if baseline_method > 0:
            if baseline_start_sample < 0:
                logging.error('Cannot extract the baseline for the trial with onset ' + str(onsets[trial_idx]) + ', the start of the baseline-epoch lies before the start of the trial-epoch')
                raise RuntimeError('Cannot extract baseline')
            if baseline_end_sample > trial_num_samples:
                logging.error('Cannot extract the baseline for the trial with onset ' + str(onsets[trial_idx]) + ', the end of the baseline-epoch lies outside of the trial-epoch')
                raise RuntimeError('Cannot extract baseline')
            if baseline_start_sample < local_start or baseline_end_sample > local_end:
                logging.error('Cannot extract the baseline for the trial with onset ' + str(onsets[trial_idx]) + ', the range for the baseline lies outside of the trial-epoch because that part of the trial-epoch was out-of-bounds')
                raise RuntimeError('Cannot extract baseline')

        # load the trial data
        try:
            trial_data = data_reader.retrieve_sample_range_data(trial_sample_start, trial_sample_end,
                                                                channels=retrieve_channels, ensure_own_data=False)
        except (RuntimeError, LookupError):
            raise RuntimeError('Could not load data')

        # loop through the channels
        for channel_idx in range(len(retrieve_channels)):

            # extract the trial data and perform baseline normalization on the trial if needed
            if baseline_method == 0:
                # Note: since we are not manipulating the data (which in the other cases converts a view to data), ensure
                #       the epoch has its own data (not a view). Avoiding views prevents trouble with overlapping
                #       epochs; in addition, avoiding views ensures there are no remaining references to the source
                #       numpy-array, allowing it to be cleared from memory
                if trial_data[channel_idx].base is None:
                    data[channel_idx, trial_idx, local_start:local_end] = trial_data[channel_idx]
                else:
                    data[channel_idx, trial_idx, local_start:local_end] = trial_data[channel_idx].copy()
            elif baseline_method == 1:
                baseline_mean = np.nanmean(trial_data[channel_idx][baseline_start_sample:baseline_end_sample])
                data[channel_idx, trial_idx, local_start:local_end] = trial_data[channel_idx] - baseline_mean
            elif baseline_method == 2:
                baseline_median = np.nanmedian(trial_data[channel_idx][baseline_start_sample:baseline_end_sample])
                data[channel_idx, trial_idx, local_start:local_end] = trial_data[channel_idx] - baseline_median

        # clear temp data
        del trial_data

        # update progress bar
        print_progressbar(trial_idx + 1, len(onsets), prefix='Progress:', suffix='Complete', length=50)

    # return the sample rate and the epoched data
    return data_reader.sampling_rate, data


def __load_data_epoch_averages__by_condition_trial(data_reader, retrieve_channels,
                                                   conditions_onsets, trial_epoch,
                                                   baseline_method, baseline_epoch, out_of_bound_method, metric_callbacks):
    """
    Load data epoch averages to a matrix (format: channel x condition x time) by looping over conditions, looping over
    the trials within a condition and then load the data per condition-trial (for all channels) and perform
    averaging (and metric calculation) by iterating over each of the channels

    Note:   When the entire dataset is not preloaded this is the fastest solution while using a small amount of
            memory (because only the required data is loaded). The '__load_data_epoch_averages__by_channel_condition_trial' is
            even more memory efficient but slower for MEF3. For preloaded data there is not much difference because the
            whole set is load to memory first, so just numpy-views are returned and used.
            # TODO: test speeds
    Note2:  only an option when at no point the full channel data is needed (e.g. cannot be used when high-pass filtering is required)
    """


    # calculate the size of the time dimension (in samples)
    trial_num_samples = int(round(abs(trial_epoch[1] - trial_epoch[0]) * data_reader.sampling_rate))
    baseline_num_samples = int(round(abs(baseline_epoch[1] - baseline_epoch[0]) * data_reader.sampling_rate))

    # initialize a data buffer (channel x conditions x samples)
    try:
        data = allocate_array((len(retrieve_channels), len(conditions_onsets), trial_num_samples))
    except MemoryError:
        raise MemoryError('Not enough memory create a data output matrix')

    # initialize a metric buffer (channel x conditions x metric)
    try:
        metric_values = None
        if metric_callbacks is not None:
            if callable(metric_callbacks):
                metric_values = allocate_array((len(retrieve_channels), len(conditions_onsets)))
            elif type(metric_callbacks) is tuple and len(metric_callbacks) > 0:
                metric_values = allocate_array((len(retrieve_channels), len(conditions_onsets), len(metric_callbacks)))
    except MemoryError:
        raise MemoryError('Not enough memory create metric output matrix')

    # create progress bar
    print_progressbar(0, len(conditions_onsets), prefix='Progress:', suffix='Complete', length=50)

    # loop through the conditions
    for condition_idx in range(len(conditions_onsets)):

        # initialize a buffer to put all the data for this condition in (channels x trials x samples)
        try:
            condition_data = allocate_array((len(retrieve_channels), len(conditions_onsets[condition_idx]), trial_num_samples))
        except MemoryError:
            raise MemoryError('Not enough memory create an condition-data matrix')

        # if baseline normalization is needed and the pre-average callback function is defined, then we first need
        # to accumulate the full (i.e. channels x trials x samples) un-normalized subset to provide to the function.
        # Therefore, we initialize an array to store the baseline values for each channel x trial, so we can normalize
        # after the callback
        baseline_data = None
        if not baseline_method == 0 and metric_callbacks is not None:
            try:
                baseline_data = allocate_array((len(retrieve_channels), len(conditions_onsets[condition_idx]), baseline_num_samples))
            except MemoryError:
                raise MemoryError('Not enough memory create temporary baseline-data matrix')

        # loop through the trials in the condition
        for trial_idx in range(len(conditions_onsets[condition_idx])):

            # calculate the sample indices
            trial_sample_start = int(round((conditions_onsets[condition_idx][trial_idx] + trial_epoch[0]) * data_reader.sampling_rate))
            trial_sample_end = trial_sample_start + trial_num_samples
            baseline_start_sample = int(round((conditions_onsets[condition_idx][trial_idx] + baseline_epoch[0]) * data_reader.sampling_rate)) - trial_sample_start
            baseline_end_sample = baseline_start_sample + baseline_num_samples
            local_start = 0
            local_end = trial_num_samples

            # check whether the trial epoch is within bounds
            if trial_sample_end < 0:
                if (out_of_bound_method == 1 and condition_idx == 0 and trial_idx == 0) or out_of_bound_method == 2:
                    logging.warning('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the end of the trial-epoch lies before the start of the data-set.')
                    continue
                else:
                    logging.error('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the end of the trial-epoch lies before the start of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                    raise RuntimeError('Cannot extract trial')
            if trial_sample_start < 0:
                if (out_of_bound_method == 1 and condition_idx == 0 and trial_idx == 0) or out_of_bound_method == 2:
                    logging.warning('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the start of the trial-epoch lies before the start of the data-set.')
                    local_start = trial_sample_start * -1
                    trial_sample_start = 0
                else:
                    logging.error('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the start of the trial-epoch lies before the start of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                    raise RuntimeError('Cannot extract trial')
            if trial_sample_start > data_reader.num_samples:
                if (out_of_bound_method == 1 and condition_idx == len(conditions_onsets) - 1 and trial_idx == len(conditions_onsets[condition_idx]) - 1) or out_of_bound_method == 2:
                    logging.warning('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the start of the trial-epoch lies after the end of the data-set.')
                    continue
                else:
                    logging.error('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the start of the trial-epoch lies after the end of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                    raise RuntimeError('Cannot extract trial')
            if trial_sample_end > data_reader.num_samples:
                if (out_of_bound_method == 1 and condition_idx == len(conditions_onsets) - 1 and trial_idx == len(conditions_onsets[condition_idx]) - 1) or out_of_bound_method == 2:
                    logging.warning('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the end of the trial-epoch lies after the end of the data-set.')
                    local_end = trial_num_samples - (trial_sample_end - data_reader.num_samples)
                    trial_sample_end = data_reader.num_samples
                else:
                    logging.error('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the end of the trial-epoch lies after the end of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                    raise RuntimeError('Cannot extract trial')

            # check whether the baseline is within bounds
            if baseline_method > 0:
                if baseline_start_sample < 0:
                    logging.error('Cannot extract the baseline for the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the start of the baseline-epoch lies before the start of the trial-epoch')
                    raise RuntimeError('Cannot extract baseline')
                if baseline_end_sample > trial_num_samples:
                    logging.error('Cannot extract the baseline for the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the end of the baseline-epoch lies outside of the trial-epoch')
                    raise RuntimeError('Cannot extract baseline')
                if baseline_start_sample < local_start or baseline_end_sample > local_end:
                    logging.error('Cannot extract the baseline for the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the range for the baseline lies outside of the trial-epoch because that part of the trial-epoch was out-of-bounds')
                    raise RuntimeError('Cannot extract baseline')

            # load the trial data
            try:
                trial_data = data_reader.retrieve_sample_range_data(trial_sample_start, trial_sample_end,
                                                                    channels=retrieve_channels, ensure_own_data=False)
            except (RuntimeError, LookupError):
                raise RuntimeError('Could not load data')

            # loop through the channels
            for channel_idx in range(len(retrieve_channels)):

                # extract the trial data and perform baseline normalization on the trial if needed
                #
                # except when there is a function callback. When a callback is then we need to first accumulate the
                # full (i.e. channels x trials x epoch) un-normalized subset to provide to the function, and store
                # the baseline values in a separate array, so they can be applied later
                #
                if baseline_method == 0 or metric_callbacks is not None:

                    # Note: not relevant whether this is a numpy-view or not, since we will average over the trials
                    #       later. Assume metric_callback does not manipulate the data it is given
                    condition_data[channel_idx, trial_idx, local_start:local_end] = trial_data[channel_idx]

                if baseline_method == 1:

                    if metric_callbacks is None:
                        # no callback, normalize and store the trial data with baseline applied
                        condition_data[channel_idx, trial_idx, local_start:local_end] = trial_data[channel_idx] - np.nanmean(trial_data[channel_idx][baseline_start_sample:baseline_end_sample])

                    else:
                        # callback, store the baseline values for later use
                        # Note: not relevant whether this is a numpy-view or not, since we will average over the trials
                        #       later. Assume metric_callback does not manipulate the data it is given
                        baseline_data[channel_idx, trial_idx, :] = trial_data[channel_idx][baseline_start_sample:baseline_end_sample]

                elif baseline_method == 2:

                    if metric_callbacks is None:
                        # no callback, normalize and store the trial data with baseline applied
                        condition_data[channel_idx, trial_idx, local_start:local_end] = trial_data[channel_idx] - np.nanmedian(trial_data[channel_idx][baseline_start_sample:baseline_end_sample])

                    else:
                        # callback, store the baseline values for later use
                        # Note: not relevant whether this is a numpy-view or not, since we will average over the trials
                        #       later. Assume metric_callback does not manipulate the data it is given
                        baseline_data[channel_idx, trial_idx, :] = trial_data[channel_idx][baseline_start_sample:baseline_end_sample]

        # check if a pre-averaging callback function is defined
        metric = None
        if metric_callbacks is not None:

            # per channel, pass the trials x epoch un-normalized subset to the callback function
            # and retrieve the result
            for channel_idx in range(len(retrieve_channels)):

                if callable(metric_callbacks):

                    # pass the trials x time un-normalized subset to the callback function(s) and store the result
                    metric_value = metric_callbacks(data_reader.sampling_rate,
                                                    condition_data[channel_idx, :, :],
                                                    None if baseline_data is None else baseline_data[channel_idx, :, :])

                    if metric_value is not None:
                        if not np.isscalar(metric_value):
                            logging.error('Return metric is not scalar')
                            raise RuntimeError('Return metric is not scalar')
                        metric_values[channel_idx, condition_idx] = metric_value

                elif type(metric_callbacks) is tuple and len(metric_callbacks) > 0:
                    for iCallback in range(len(metric_callbacks)):
                        if callable(metric_callbacks[iCallback]):

                            # pass the trials x time un-normalized subset to the callback function(s) and store the result
                            metric_value = metric_callbacks[iCallback](data_reader.sampling_rate,
                                                                       condition_data[channel_idx, :, :],
                                                                       None if baseline_data is None else baseline_data[channel_idx, :, :])
                            if metric_value is not None:
                                if not np.isscalar(metric_value):
                                    logging.error('Return metric is not scalar')
                                    raise RuntimeError('Return metric is not scalar')
                                metric_values[channel_idx, condition_idx, iCallback] = metric_value

            # the callback has been made, perform -if needed- the (postponed) normalization with the baseline values
            if baseline_method == 1:
                condition_data -= np.nanmean(baseline_data, axis=2)[:, :, None]
            elif baseline_method == 2:
                condition_data -= np.nanmedian(baseline_data, axis=2)[:, :, None]

        # average the trials for each channel (within this condition) and store the results
        data[:, condition_idx, :] = np.nanmean(condition_data, axis=1)

        # clear reference to data
        del condition_data, trial_data

        # update progress bar
        print_progressbar(condition_idx + 1, len(conditions_onsets), prefix='Progress:', suffix='Complete', length=50)

    # return the sample rate, the average epoch and the metric values (None if no metrics)
    return data_reader.sampling_rate, data, metric_values


def __subload_data_epoch_averages__from_channel__by_condition_trials(ref_data, ref_metric_values,
                                                                     data_reader, channel_idx, channel_name, channel_data,
                                                                     conditions_onsets, trial_epoch,
                                                                     baseline_method, baseline_epoch,
                                                                     out_of_bound_method, metric_callbacks,
                                                                     exclude_epochs=None,
                                                                     var_epoch=None, ref_var=None,
                                                                     CAR_per_condition=None):
    """
    For a specific channel, load data epoch averages to an already existing/initialized
    matrix (format: channel x condition x time) by looping over conditions and then within that channel-condition
    combination loop over each of the trials to load the specific channel-condition-trial data.

    Args:
        ...
        ref_data (ref ndarray):             Reference to the numpy matrix that holds all the epoch averages. Reference is also returned on success
        ref_metric_values (ref ndarray):    Reference to the ... that holds all the metric values. Reference is also returned on success
        exclude_epochs (ndarray):           List of ranges (tuples) to exclude. Each tuple should contain two values that
                                            indicate the start-time and end-time of what should be excluded (nanned)
        CAR_per_condition:                  If set, common averages re-referencing will be applied...
    Returns:
        variances (ndarray):                Array the variance for each condition

    """

    if channel_data is None:
        channel_num_samples = data_reader.num_samples
    else:
        channel_num_samples = channel_data.size

    # calculate the size of the time dimension (in samples)
    trial_num_samples = int(round(abs(trial_epoch[1] - trial_epoch[0]) * data_reader.sampling_rate))

    if baseline_method > 0:
        baseline_num_samples = int(round(abs(baseline_epoch[1] - baseline_epoch[0]) * data_reader.sampling_rate))

        # if the epochs should be baselined and common average re-referencing should be applied, make sure that the baseline
        # window is within the trial window (because the common averages that are passed for re-referencing cover only
        # the trial epoch)
        if CAR_per_condition is not None:

            if trial_num_samples != CAR_per_condition.shape[1]:
                logging.error('The number of samples for the trial epoch does not match the number of samples in the CAR matrix\n')
                raise RuntimeError('Number of sample in CAR mismatch')

            if baseline_epoch[0] < trial_epoch[0]:
                logging.error('Invalid \'baseline_epoch\' parameter, the given baseline start-point (at ' + str(baseline_epoch[0]) + ') lies before the trial start-point (at ' + str(trial_epoch[0]) + ')\n'
                              'When baselining is enabled and common average re-referencing needs to be applied, the baseline window should fall within the trial window\n')
                raise ValueError('Invalid \'baseline_epoch\' parameter')
            if baseline_epoch[1] > trial_epoch[1]:
                logging.error('Invalid \'baseline_epoch\' parameter, the given baseline end-point (at ' + str(baseline_epoch[1]) + ') lies after the trial end-point (at ' + str(trial_epoch[1]) + ')\n'
                'When baselining is enabled and common average re-referencing needs to be applied, the baseline window should fall within the trial window\n')
                raise ValueError('Invalid \'baseline_epoch\' parameter')

    # check if the variances should be retrieved
    if var_epoch is not None:
        if var_epoch[1] < var_epoch[0]:
            logging.error('Invalid \'var_epoch\' parameter, the given end-point (at ' + str(var_epoch[1]) + ') lies before the start-point (at ' + str(var_epoch[0]) + ')')
            raise ValueError('Invalid \'var_epoch\' parameter')

        # ensure that the variance epoch is within the trial epoch
        if var_epoch[0] < trial_epoch[0]:
            logging.error('Invalid \'var_epoch\' parameter, the given variance start-point (at ' + str(var_epoch[0]) + ') lies before the trial start-point (at ' + str(trial_epoch[0]) + ')')
            raise ValueError('Invalid \'var_epoch\' parameter')
        if var_epoch[1] > trial_epoch[1]:
            logging.error('Invalid \'var_epoch\' parameter, the given variance end-point (at ' + str(var_epoch[1]) + ') lies after the trial end-point (at ' + str(trial_epoch[1]) + ')')
            raise ValueError('Invalid \'var_epoch\' parameter')

    # prepare excludes if needed
    exclude_epochs_starts = None
    exclude_epochs_ends = None
    if exclude_epochs is not None:
        exclude_epochs_starts = np.full(len(exclude_epochs), 0, dtype=int)
        exclude_epochs_ends = np.full(len(exclude_epochs), 0, dtype=int)
        for exclude_epoch_index in range(len(exclude_epochs)):
            exclude_epochs_starts[exclude_epoch_index] = int(round(exclude_epochs[exclude_epoch_index][0] * data_reader.sampling_rate))
            exclude_epochs_ends[exclude_epoch_index] = int(round(exclude_epochs[exclude_epoch_index][1] * data_reader.sampling_rate))

    # loop through the conditions
    for condition_idx in range(len(conditions_onsets)):

        # initialize a buffer to put all the channel's epoch data for this condition in (trials x samples)
        # and
        try:
            condition_epoch_data = allocate_array((len(conditions_onsets[condition_idx]), trial_num_samples))
            if var_epoch is not None:
                condition_trial_variances = allocate_array(len(conditions_onsets[condition_idx]))
        except MemoryError:
            raise MemoryError('Not enough memory to create a temporary data matrix')


        # if baseline normalization is needed and the pre-average callback function is defined, then we first
        # need to accumulate the full (i.e. channels x trials x epoch) un-normalized subset to provide to the
        # function. Therefore, we initialize an array to store the baseline values for each channel x trial, so
        # we can normalize after the callback
        baseline_data = None
        if baseline_method > 0 and metric_callbacks is not None:
            try:
                baseline_data = allocate_array((len(conditions_onsets[condition_idx]), baseline_num_samples))
            except MemoryError:
                raise MemoryError('Not enough memory to create a temporary condition-channel baseline data matrix')

        # loop through the trials in the condition
        for trial_idx in range(len(conditions_onsets[condition_idx])):

            # calculate the sample indices
            trial_sample_start = int(round((conditions_onsets[condition_idx][trial_idx] + trial_epoch[0]) * data_reader.sampling_rate))
            trial_sample_end = trial_sample_start + trial_num_samples
            if baseline_method > 0:
                baseline_start_sample = int(round((conditions_onsets[condition_idx][trial_idx] + baseline_epoch[0]) * data_reader.sampling_rate))
                baseline_end_sample = baseline_start_sample + baseline_num_samples
            local_start = 0
            local_end = trial_num_samples

            # check whether the trial epoch is within bounds
            if trial_sample_end < 0:
                if (out_of_bound_method == 1 and condition_idx == 0 and trial_idx == 0) or out_of_bound_method == 2:
                    if channel_idx == 0:
                        logging.warning('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the end of the trial-epoch lies before the start of the data-set.')
                    continue
                else:
                    logging.error('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the end of the trial-epoch lies before the start of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                    raise RuntimeError('Cannot extract trial')
            if trial_sample_start < 0:
                if (out_of_bound_method == 1 and condition_idx == 0 and trial_idx == 0) or out_of_bound_method == 2:
                    if channel_idx == 0:
                        logging.warning('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the start of the trial-epoch lies before the start of the data-set.')
                    local_start = trial_sample_start * -1
                    trial_sample_start = 0
                else:
                    logging.error('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the start of the trial-epoch lies before the start of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                    raise RuntimeError('Cannot extract trial')
            if trial_sample_start > channel_num_samples:
                if (out_of_bound_method == 1 and condition_idx == len(conditions_onsets) - 1 and trial_idx == len(conditions_onsets[condition_idx]) - 1) or out_of_bound_method == 2:
                    if channel_idx == 0:
                        logging.warning('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the start of the trial-epoch lies after the end of the data-set.')
                    continue
                else:
                    logging.error('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the start of the trial-epoch lies after the end of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                    raise RuntimeError('Cannot extract trial')
            if trial_sample_end > channel_num_samples:
                if (out_of_bound_method == 1 and condition_idx == len(conditions_onsets) - 1 and trial_idx == len(conditions_onsets[condition_idx]) - 1) or out_of_bound_method == 2:
                    if channel_idx == 0:
                        logging.warning('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the end of the trial-epoch lies after the end of the data-set.')
                    local_end = trial_num_samples - (trial_sample_end - channel_num_samples)
                    trial_sample_end = channel_num_samples
                else:
                    logging.error('Cannot extract the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the end of the trial-epoch lies after the end of the data-set. Use a different out_of_bound_handling argument to allow out-of-bound trial epochs')
                    raise RuntimeError('Cannot extract trial')

            # check whether the baseline is within bounds
            if baseline_method > 0:
                if baseline_start_sample < 0 or baseline_end_sample > channel_num_samples:
                    logging.error('Cannot extract the baseline for the trial with onset ' + str(conditions_onsets[condition_idx][trial_idx]) + ', the range for the baseline lies outside of the data')
                    raise RuntimeError('Cannot extract baseline')

            # extract the trial data
            if channel_data is None:
                # retrieve using reader

                try:
                    trial_trial_data = data_reader.retrieve_sample_range_data(trial_sample_start, trial_sample_end, channels=channel_name, ensure_own_data=False)[0]

                    # retrieve baseline data if baselining is needed and we are not performing CAR (with CAR, the
                    # baseline should be inside of the trial epoch, therefor we just copy it from there later)
                    if baseline_method > 0 and CAR_per_condition is None:
                        # TODO: if baseline is within trial_trial_data, no need to read from disk like below, instead just copy from trial_trial_data
                        trial_baseline_data = data_reader.retrieve_sample_range_data(baseline_start_sample, baseline_end_sample, channels=channel_name, ensure_own_data=False)[0]

                except (RuntimeError, LookupError):
                    raise RuntimeError('Could not load data')

            else:
                # retrieve from passed channel-data

                trial_trial_data = channel_data[trial_sample_start:trial_sample_end]

                # retrieve baseline data if baselining is needed and we are not performing CAR (with CAR, the
                # baseline should be inside of the trial epoch, therefor we just copy it from there later)
                if baseline_method > 0 and CAR_per_condition is None:
                    trial_baseline_data = channel_data[baseline_start_sample:baseline_end_sample]


            #
            # (optionally) CAR_per_condition
            #

            if CAR_per_condition is not None:
                trial_trial_data = np.array(trial_trial_data - CAR_per_condition[condition_idx, :])

                # since the baseline window should be within the trial window, we can copy the values that are already re-referenced
                if baseline_method > 0:
                    trial_baseline_data = np.array(trial_trial_data[baseline_start_sample - trial_sample_start:baseline_end_sample - trial_sample_start])

            #
            # (optionally) exclude epochs
            #

            if exclude_epochs is not None:

                # function to check and exclude (nan) values in a data range
                def apply_excludes_to_range(ref_range_data, range_sample_start, range_sample_end):

                    # check if trial start or end is within an exclude epoch
                    exclude_starts_in_range = np.logical_and(exclude_epochs_starts >= range_sample_start, exclude_epochs_starts <= range_sample_end)
                    exclude_ends_in_range = np.logical_and(exclude_epochs_ends >= range_sample_start, exclude_epochs_ends <= range_sample_end)
                    exclude_surround_range = np.logical_and(exclude_epochs_starts < range_sample_start, exclude_epochs_ends > range_sample_end)
                    excludes_indices = np.logical_or(np.logical_or(exclude_starts_in_range, exclude_ends_in_range), exclude_surround_range).nonzero()[0]

                    # apply the exclusion epochs that were found
                    for exclude_index in excludes_indices:

                        start_nan_index = 0
                        if exclude_starts_in_range[exclude_index]:
                            start_nan_index = exclude_epochs_starts[exclude_index] - trial_sample_start

                        end_nan_index = len(trial_trial_data)
                        if exclude_ends_in_range[exclude_index]:
                            end_nan_index = exclude_epochs_ends[exclude_index] - trial_sample_start

                        ref_range_data[start_nan_index:end_nan_index] = np.nan

                #
                if not trial_trial_data.flags['OWNDATA']:
                    trial_trial_data = trial_trial_data.copy()
                apply_excludes_to_range(trial_trial_data, trial_sample_start, trial_sample_end)

                if baseline_method > 0:
                    if not trial_baseline_data.flags['OWNDATA']:
                        trial_baseline_data = trial_baseline_data.copy()
                    apply_excludes_to_range(trial_baseline_data, baseline_start_sample, baseline_end_sample)

            # determine the variance for the trial
            if var_epoch is not None:

                var_epoch_sample_offset_start = int(round((var_epoch[0] - trial_epoch[0]) * data_reader.sampling_rate))
                var_epoch_sample_offset_end   = int(round((var_epoch[1] - trial_epoch[0]) * data_reader.sampling_rate))

                # TODO: minimum number of samples to determine var?
                var_values = trial_trial_data[var_epoch_sample_offset_start:var_epoch_sample_offset_end]
                if (~np.isnan(var_values)).sum() > 1:
                    condition_trial_variances[trial_idx] = np.nanvar(var_values)


            # perform baseline normalization on the trial if needed
            #
            # except when there is a function callback. When a callback is then we need to first accumulate the
            # full (i.e. channels x trials x epoch) un-normalized subset to provide to the function, and store
            # the baseline values in a separate array, so they can be applied later
            #
            if baseline_method == 0 or metric_callbacks is not None:

                # Note: not relevant whether this is a numpy-view or not, since we will average over the trials
                #       later. Assume metric_callback does not manipulate the data it is given
                condition_epoch_data[trial_idx, local_start:local_end] = trial_trial_data

            if baseline_method == 1:

                if metric_callbacks is None:
                    # no callback, normalize and store the trial data with baseline applied

                    if exclude_epochs is None or not np.isnan(trial_baseline_data).all():
                        condition_epoch_data[trial_idx, local_start:local_end] = trial_trial_data - np.nanmean(trial_baseline_data)
                    else:
                        condition_epoch_data[trial_idx, local_start:local_end] = trial_trial_data

                else:
                    # callback, store the baseline values for later use
                    # Note: not relevant whether this is a numpy-view or not, since we will average over the trials
                    #       later. Assume metric_callback does not manipulate the data it is given
                    baseline_data[trial_idx, :] = trial_baseline_data

            elif baseline_method == 2:

                if metric_callbacks is None:
                    # no callback, normalize and store the trial data with baseline applied

                    if exclude_epochs is None or not np.isnan(trial_baseline_data).all():
                        condition_epoch_data[trial_idx, local_start:local_end] = trial_trial_data - np.nanmedian(trial_baseline_data)
                    else:
                        condition_epoch_data[trial_idx, local_start:local_end] = trial_trial_data
                else:
                    # callback, store the baseline values for later use
                    # Note: not relevant whether this is a numpy-view or not, since we will average over the trials
                    #       later. Assume metric_callback does not manipulate the data it is given
                    baseline_data[trial_idx, :] = trial_baseline_data

        # check if a pre-averaging callback function is defined
        if metric_callbacks is not None:

            if callable(metric_callbacks):

                # pass the trials x epoch un-normalized subset to the callback function(s) and store the result
                metric_value = metric_callbacks(data_reader.sampling_rate, condition_epoch_data, baseline_data)
                if metric_value is not None:
                    ref_metric_values[channel_idx, condition_idx] = metric_value

            elif type(metric_callbacks) is tuple and len(metric_callbacks) > 0:
                for iCallback in range(len(metric_callbacks)):
                    if callable(metric_callbacks[iCallback]):

                        # pass the trials x epoch un-normalized subset to the callback function(s) and store the result
                        metric_value = metric_callbacks[iCallback](data_reader.sampling_rate, condition_epoch_data, baseline_data)
                        if metric_value is not None:
                            ref_metric_values[channel_idx, condition_idx, iCallback] = metric_value

            # the callback has been made, check if (postponed) normalization should occur based on the baseline
            if baseline_method == 1:
                condition_epoch_data -= np.nan_to_num(np.nanmean(baseline_data, axis=1)[:, None])
            elif baseline_method == 2:
                condition_epoch_data -= np.nan_to_num(np.nanmedian(baseline_data, axis=1)[:, None])

        # average the trials for each channel (within this condition) and store the results
        # Note: catching warning is needed to suppress unnecessary "RuntimeWarning: Mean of empty slice"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ref_data[channel_idx, condition_idx, :] = np.nanmean(condition_epoch_data, axis=0)
        del condition_epoch_data

        # average the trial variances and store the results
        # TODO: var could be calculated by taking variance over a range (var_epoch) in condition_epoch_data variable. However,
        #       condition_epoch_data can have baselining applied, not sure if/what effect that has on the variance
        if var_epoch is not None:
            # Note: catching warning is needed to suppress unnecessary "RuntimeWarning: Mean of empty slice"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ref_var[channel_idx, condition_idx] = np.nanmean(condition_trial_variances)
            del condition_trial_variances

    #
    return data_reader.sampling_rate, ref_data, ref_metric_values, ref_var


def __load_data_epoch_averages__by_channel_condition_trial(data_reader, channels,
                                                           conditions_onsets, trial_epoch,
                                                           baseline_method, baseline_epoch, out_of_bound_method, metric_callbacks):
    """
    Load data epoch averages to a matrix (format: channel x condition x time) by looping over channels, then over
    conditions and then within that channel-condition combination loop over each of the trials to load the specific
    channel-condition-trial data. The averaging (and metric calculation) is performed on a temporary matrix in the
    channel loop

    Note:     This function is even more memory efficient than '__load_data_epoch_averages__by_condition_trial', but
              slower when the entire dataset is not pre-loaded. For pre-loaded datasets there is not much difference because
              the whole set is loaded to memory first, so just numpy-views are returned and used.
              # TODO: retest speeds
    """

    # calculate the size of the time dimension (in samples)
    trial_num_samples = int(round(abs(trial_epoch[1] - trial_epoch[0]) * data_reader.sampling_rate))
    baseline_num_samples = int(round(abs(baseline_epoch[1] - baseline_epoch[0]) * data_reader.sampling_rate))

    # initialize a data buffer (channel x conditions x samples)
    try:
        data = allocate_array((len(channels), len(conditions_onsets), trial_num_samples))
    except MemoryError:
        raise MemoryError('Not enough memory create a data output matrix')

    # initialize a metric buffer (channel x conditions x metric)
    try:
        metric_values = None
        if metric_callbacks is not None:
            if callable(metric_callbacks):
                metric_values = allocate_array((len(channels), len(conditions_onsets)))
            elif type(metric_callbacks) is tuple and len(metric_callbacks) > 0:
                metric_values = allocate_array((len(channels), len(conditions_onsets), len(metric_callbacks)))
    except MemoryError:
        raise MemoryError('Not enough memory create a metric output matrix')

    # create progress bar
    print_progressbar(0, len(conditions_onsets), prefix='Progress:', suffix='Complete', length=50)

    # loop through the channels
    for channel_idx in range(len(channels)):

        #
        try:
            __subload_data_epoch_averages__from_channel__by_condition_trials(data, metric_values,
                                                                             data_reader, channel_idx, channels[channel_idx], None,
                                                                             conditions_onsets, trial_epoch,
                                                                             baseline_method, baseline_epoch, out_of_bound_method,
                                                                             metric_callbacks)
        except (MemoryError, RuntimeError):
            raise RuntimeError('Error upon loading, epoching and averaging data')

        # update progress bar
        print_progressbar(channel_idx + 1, len(channels), prefix='Progress:', suffix='Complete', length=50)

    # return the sample rate, the average epoch and the metric values (None if no metrics)
    return data_reader.sampling_rate, data, metric_values


def __load_data_epochs__by_channels__withPrep(average, data_reader, retrieve_channels, onsets,
                                              trial_epoch, baseline_method, baseline_epoch,
                                              out_of_bound_method, metric_callbacks,
                                              high_pass, early_reref, line_noise_removal, late_reref, priority):
    """
    Load the data, preprocess and either epoch or (optionally) calculate metrics and epoch-average to a matrix.
    This function processes data per channel in order to minimize memory usage but still be able to apply preprocessing
    steps (optionally: high-pass filtering, early re-referencing, line-noise removal and late re-referencing).
    After preprocessing, the channel data is either epoched and returned in a matrix (format: channel x trials/epochs x time),
    or (optionally) metrics are calculated and the epoch-average is in a matrix (format: channel x condition x time).
    In addition, an optimized parameter can be set to either 'mem' or 'speed'; 'mem' will unload the channel inbetween
    processing steps, making the process slower but most memory efficient; 'speed' will keep the channel data in memory
    throughout the processing, allowing for more speed but also requiring more memory.


    Note:   Note that for preloaded datasets, preprocessing will require a copy of the channel data for manipulation, so
            there is no memory benefit in the fact that the entire dataset is loaded into memory first.
            # TODO: retest speeds

    Args:
        average (boolean):                  Whether, after preprocessing, only epochs (False) should be extracted and
                                            returned, or whether (optionally) metrics should be calculated and
                                            epoch-averages should be returned (True)
        data_reader (IeegDataReader):       An instance of the IeegDataReader to retrieve metadata and channel data
        retrieve_channels (list or tuple):  The channels (by name) of which the data should be retrieved, the output will be sorted accordingly
                                            Note: that this does not have to include the channels that are required for re-referencing

        onsets (1d/2d list or tuple):       If only the epochs need to be extracted and returned (average=False) then this
                                            argument should be a 1d list or tuple that holds the onsets of the trials around
                                            which to epoch the data.
                                            If metric and average-epochs need to be calculated and returned (Average=True) then
                                            this argument should be a 2d list or tuple that indicates the conditions, and
                                            the onsets of the trials that belong to each condition. (format: conditions x condition onsets)
    """

    # if baselining and late re-ref channel selection based on variance is enabled, make sure the baseline epoch is
    # included in the trial epoch. This way the common average (which is calculated over the trial epoch) can be applied
    # to both the baseline epoch and the trial epoch
    if baseline_method > 0 and late_reref is not None and late_reref.late_group_reselect_varPerc is not None:
        if baseline_epoch[0] < trial_epoch[0]:
            logging.error('Invalid \'baseline_epoch\' parameter, the given baseline start-point (at ' + str( baseline_epoch[0]) + ') lies before the trial start-point (at ' + str(trial_epoch[0]) + ')\n'
                          'When baselining is enabled and common average re-referencing needs to be applied, the baseline window should fall within the trial window\n')
            raise ValueError('Invalid \'baseline_epoch\' parameter')
        if baseline_epoch[1] > trial_epoch[1]:
            logging.error('Invalid \'baseline_epoch\' parameter, the given baseline end-point (at ' + str(baseline_epoch[1]) + ') lies after the trial end-point (at ' + str(trial_epoch[1]) + ')\n'
                          'When baselining is enabled and common average re-referencing needs to be applied, the baseline window should fall within the trial window\n')
            raise ValueError('Invalid \'baseline_epoch\' parameter')

    # calculate the size of the time dimension (in samples)
    trial_num_samples = int(round(abs(trial_epoch[1] - trial_epoch[0]) * data_reader.sampling_rate))
    baseline_num_samples = int(round(abs(baseline_epoch[1] - baseline_epoch[0]) * data_reader.sampling_rate))

    # initialize a data buffer (channel x trials/epochs x time)
    try:
        data = allocate_array((len(retrieve_channels), len(onsets), trial_num_samples))
    except MemoryError:
        raise MemoryError('Not enough memory create a data output matrix')

    # initialize a metric buffer (channel x conditions x metric)
    if average:
        try:
            metric_values = None
            if metric_callbacks is not None:
                if callable(metric_callbacks):
                    metric_values = allocate_array((len(retrieve_channels), len(onsets)))
                elif type(metric_callbacks) is tuple and len(metric_callbacks) > 0:
                    metric_values = allocate_array((len(retrieve_channels), len(onsets), len(metric_callbacks)))
        except MemoryError:
            raise MemoryError('Not enough memory create a metric output matrix')

    # create a list of all channels that we need, most with the purpose of reading and returning (e.g. only ECoG) but some only to be used for re-referencing (e.g. both ECoG and SEEG)
    all_channels = retrieve_channels.copy()
    try:
        if early_reref is not None:
            early_req_channels = early_reref.get_required_channels(retrieve_channels)
            early_req_groups = early_reref.get_required_groups(retrieve_channels)
            for channel in early_req_channels:
                if channel not in all_channels:
                    all_channels.append(channel)
        if late_reref is not None:
            late_req_channels = late_reref.get_required_channels(retrieve_channels)
            late_req_groups = late_reref.get_required_groups(retrieve_channels)
            for channel in late_req_channels:
                if channel not in all_channels:
                    all_channels.append(channel)
    except ValueError:
        logging.error('Could not find a channel that is to be used for early or late re-referencing in the reref struct')
        raise ValueError('Could not find a channel that is to retrieved in the reref struct')


    #
    # Initialize variable that track processing
    #

    channel_epoched = dict()                        # tracks for each channel whether it has been epoched (fully processed)
    for channel in retrieve_channels:
        channel_epoched[channel] = False

    if early_reref is not None:
        channel_early_reref_collected = dict()          # tracks for each channel if the channel-data is added to the early re-ref group averages
        for channel in early_req_channels:
            channel_early_reref_collected[channel] = False

        early_group_data = dict()                       # for each early re-ref group stores the (total/average) data
        early_group_numdata = dict()                    # for each early re-ref group stores the num of samples for each datapoint in the (total) data
        early_group_channels_collected = dict()         # tracks for each early re-ref group all the channels that need to be collected
        for group in early_req_groups:
            early_group_data[str(group)] = None
            early_group_numdata[str(group)] = None
            early_group_channels_collected[str(group)] = dict()
            for channel in early_reref.groups[group]:
                early_group_channels_collected[str(group)][channel] = False

    if late_reref is not None:
        channel_late_reref_collected = dict()          # tracks for each channel if the channel-data is added to the late re-ref group averages
        for channel in late_req_channels:
            channel_late_reref_collected[channel] = False

        late_group_data = dict()                        # for each late re-ref group stores the (total/average) data
        late_group_numdata = dict()                     # for each late re-ref group stores the num of samples for each datapoint in the (total) data
        late_group_channels_collected = dict()          # tracks for each late re-ref group all the channels that need to be collected
        for group in late_req_groups:
            late_group_data[str(group)] = None
            late_group_numdata[str(group)] = None
            late_group_channels_collected[str(group)] = dict()
            for channel in late_reref.groups[group]:
                late_group_channels_collected[str(group)][channel] = False

    channel_data = dict()
    channel_hp_applied = dict()              # tracks for each channel in the channel-data matrix if high-pass filtering is applied
    channel_early_applied = dict()           # tracks for each channel in the channel-data matrix if early re-ref is applied
    channel_lnr_applied = dict()             # tracks for each channel in the channel-data matrix if line-noise removal is applied
    for channel in all_channels:
        channel_data[channel] = None
        channel_hp_applied[channel] = False
        channel_early_applied[channel] = False
        channel_lnr_applied[channel] = False


    #
    # Prepare filters
    #

    if high_pass is not None:

        order = 2
        fs = data_reader.sampling_rate
        pass_freq = 0.50     # Hz <<<<
        stop_freq = 0.05     # Hz
        pass_ripple = 3      # dB
        stop_atten = 30      # dB

        # TODO: matlab also allows passing of stopband (buttord), however getting the same values out of python buttord is difficult
        #       settling for standard python ways for now (direct use of butter)
        """
        # normalize the passband and stopband to the Nyquist frequency
        norm_pass_freq = pass_freq / (fs / 2)  # pass band freq in radian
        norm_stop_freq = stop_freq / (fs / 2)  # stop band freq in radian
    
        filtOrder, cut_freq = signal.buttord(norm_pass_freq, norm_stop_freq, pass_ripple, stop_atten, True)
    
        # sos = signal.butter(order, normal_cutoff, btype='highpass', analog=False, output='sos', fs=fs)
        #[filtZeros, filtPoles, filtGains] = signal.butter(2, 2.745120377767732e-04, 'high', output='zpk')
        [filtZeros, filtPoles, filtGains] = signal.butter(2, 2.745120377767732e-04, 'high', output='zpk', fs=fs)
        
        [filtSos] = signal.zpk2sos(filtZeros, filtPoles, filtGains)
        gain = filtGains
        """

        # normalize the high-pass cut-off frequency using the nyquist frequency (srate / 2)
        cut_freq = pass_freq / (data_reader.sampling_rate / 2)

        # design a butterworth filter and get the filter coefficients (numerator / denominator (‘ba’)
        hp_numerator, hp_denominator = signal.butter(order, cut_freq, btype='highpass', analog=False, output='ba', fs=fs)
        # TODO: the 'ba' or 'sos' returned by butter differ from what matlab gives

        #sos = signal.butter(order, normal_cutoff, btype='highpass', analog=False, output='sos', fs=fs)
        #sos2 = [[1, -2, 1, 1, -1.998780375302085, 0.998781118591159]]  # taken from matlab
        #print(sos)


    if line_noise_removal is not None:

        # design a notch filter and get the filter coefficients (numerator / denominator (‘ba’)
        lnr_numerator, lnr_denominator = signal.iirnotch(line_noise_removal, 30.0, data_reader.sampling_rate)


    #
    # Progress bar subscript
    #

    def update_progressbar():

        # set how much each step contributes to the total progress
        prop_early_collected = 0
        prop_late_collected = 0
        if early_reref is None and late_reref is None:
            prop_channel_epoched = 1

        else:
            prop_channel_epoched = .4

            if early_reref is not None and late_reref is None:
                prop_early_collected = .6

            elif early_reref is None and late_reref is not None:
                prop_late_collected = .6

            else:
                if priority == 'mem':
                    prop_early_collected = .3
                    prop_late_collected = .3
                else:
                    prop_early_collected = .5
                    prop_late_collected = .1

        # retrieve proportions
        progression_channel_epoched = sum(channel_epoched.values()) / len(channel_epoched)
        progression_early_collected = 0
        progression_late_collected = 0
        if early_reref is not None:
            progression_early_collected = sum([sum(v.values()) for v in early_group_channels_collected.values()]) / sum([len(v) for v in early_group_channels_collected.values()])
        if late_reref is not None:
            progression_late_collected = sum([sum(v.values()) for v in late_group_channels_collected.values()]) / sum([len(v) for v in late_group_channels_collected.values()])

        # update the progress bar
        prop = prop_channel_epoched * progression_channel_epoched + prop_early_collected * progression_early_collected + prop_late_collected * progression_late_collected
        print_progressbar(prop, 1, prefix='Progress:', suffix='Complete', length=50)

    # create progress bar
    update_progressbar()


    #
    # Process
    #

    # until all channels are epoch-ed (fully processed)
    while not all(channel_epoched.values()):

        # loop over all the required channels
        # Note: potentially even over the ones that do not need to be retrieved but are still needed for re-referencing
        for channel in all_channels:
            channel_idx = None

            # check if we need this channel, which is the case when:
            #   - it still needs to be epoch-ed
            #   - when it is needed for early re-ref but not collected
            #   - when it is needed for late re-ref but not collected
            if (channel in channel_epoched.keys() and not channel_epoched[channel]) or \
               (early_reref is not None and channel in early_req_channels and not channel_early_reref_collected[channel]) or \
               (late_reref is not None and channel in late_req_channels and not channel_late_reref_collected[channel]):
                # needed, process channel

                # check if channel data is available
                # Note: during speed option the channel data is kept in memory, so no reloading is required when still in memory
                if channel_data[channel] is None:
                    #print(channel + ": load")

                    # retrieve the channel data
                    # TODO: consider data type
                    #       - PyMEF will always return float64
                    #         (original data format is 32-bit integer, but outputting float64 allows NaNs for discontinuities in the time-series and still retains 53-bit significand precision to hold the exact 32-bit value)
                    #       - Brainvision can be 16-bit integer, 32-bit integer or 32-bit float. However, the resolution in the channel acts as multiplication factor
                    #         As such, if the multiplication factor is 1 (or empty) then the output will always be in 32-bit floats (not 64-bit to save memory).
                    #         However, if a multiplication factor
                    # Note: ensure it is not a view, elsewise manipulations further on might adjust the source data
                    try:
                        channel_data[channel] = data_reader.retrieve_channel_data(channel, True)
                    except RuntimeError:
                        raise RuntimeError('Error upon retrieving data')

                #
                # High-pass filtering
                #
                if high_pass and not channel_hp_applied[channel]:
                    #print(channel + ": HP")

                    # Filter the data
                    channel_data[channel] = signal.filtfilt(hp_numerator, hp_denominator, channel_data[channel], padtype='odd', padlen=3 * (max(len(hp_numerator), len(hp_denominator)) - 1))

                    # TODO: more exact translation from matlab
                    #       sosfiltfilt (so with sos) returns different values on the same data
                    #y = signal.filtfilt(sos, gain, channel_data[channel])
                    #y = signal.filtfilt(hp_numerator, hp_denominator, channel_data[channel])
                    #y = signal.filtfilt(hp_numerator, hp_denominator, channel_data[channel], padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
                    #y2 = signal.sosfiltfilt(sos2, channel_data[channel], padtype=None)
                    #python padlength = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
                    #print(y[1:10])
                    #print(y2[1:10])

                    # set high passing as to been applied to the channel-data in memory
                    channel_hp_applied[channel] = True


                #
                # Early re-referencing
                #

                # check if early re-referencing needed
                if early_reref is not None:

                    #
                    # Early re-referencing collect
                    #

                    # check if the data of this channel (at this point) is already collected for the early re-reference groups
                    if not channel_early_reref_collected[channel]:
                        # early not collected
                        #print(channel + ": Collecting early reref values from channel")

                        # loop over the early-reref groups
                        for group in early_req_groups:

                            # check if this group requires this channel
                            if channel in early_group_channels_collected[str(group)].keys():

                                # create arrays to hold the group data if not yet initialized
                                if early_group_data[str(group)] is None:
                                    early_group_data[str(group)] = np.zeros((len(channel_data[channel]),), dtype=np.float64)
                                    if early_reref.channel_exclude_epochs is not None:
                                        early_group_numdata[str(group)] = np.zeros((len(channel_data[channel]),), dtype=np.uint16)

                                # add to group data
                                if early_reref.channel_exclude_epochs is None or channel not in early_reref.channel_exclude_epochs:

                                    # no exclusion epochs, just add the whole channel
                                    early_group_data[str(group)] += channel_data[channel]

                                    # count the number of samples added to the total if needed
                                    if early_reref.channel_exclude_epochs is not None:
                                        early_group_numdata[str(group)] += 1

                                else:
                                    # channel has exclusion epochs

                                    # create a binary numpy vector of the samples to include
                                    channel_includes = np.ones((len(channel_data[channel]),), dtype=bool)
                                    for channel_exclude_epoch in early_reref.channel_exclude_epochs[channel]:
                                        exclude_sample_start = int(round(channel_exclude_epoch[0] * data_reader.sampling_rate))
                                        exclude_sample_end = int(round(channel_exclude_epoch[1] * data_reader.sampling_rate))
                                        channel_includes[exclude_sample_start:exclude_sample_end] = 0

                                    # add the channel (taking into account on the inclusion vector)
                                    early_group_data[str(group)] += (channel_data[channel] * channel_includes)
                                    early_group_numdata[str(group)] += channel_includes
                                    pass

                                # flag channel within the group as collected
                                early_group_channels_collected[str(group)][channel] = True

                                # check whether all the channels in the group are collected
                                if all(early_group_channels_collected[str(group)].values()):

                                    # take the average over the total
                                    # (if specific epochs were excluded, each sample should be divided by its own number)
                                    if early_reref.channel_exclude_epochs is not None:
                                        early_group_data[str(group)] /= early_group_numdata[str(group)]

                                        # clear the array was used for division
                                        del early_group_numdata[str(group)]

                                    else:
                                        early_group_data[str(group)] /= len(early_group_channels_collected[str(group)])

                        # flag that for this channel the re-ref values have been collected
                        channel_early_reref_collected[channel] = True

                        # update the progress bar
                        update_progressbar()

                        # check if channel is no longer needed after this (for epoch-ing or for late re-ref)
                        # Note: this also means the channel was only loaded for early re-referencing
                        if channel not in channel_epoched.keys() and (late_reref is None or channel not in channel_late_reref_collected):
                            # channel-data is no longer needed at all

                            # remove the reference to the numpy array, this way the memory should be available for collection
                            channel_data[channel] = None
                            gc.collect()

                            # skip to next channel
                            continue

                    #
                    # Early re-referencing apply
                    #

                    # check if early re-referencing is not applied to this channel
                    if not channel_early_applied[channel]:

                        # retrieve the early re-ref group for this channel
                        group = early_reref.channel_group[channel]

                        # check if all the early re-referencing information is available yet (early average for this group)
                        if all(early_group_channels_collected[str(group)].values()):
                            # all required information is available, perform early re-referencing on the channel

                            #print(channel + ": performing early reref on channel")

                            # perform early re-ref using reref_values
                            channel_data[channel] -= early_group_data[str(group)]

                            # set early re-referencing as to have been applied to the channel-data in memory
                            channel_early_applied[channel] = True

                            # TODO: if this is the latest channel to use the early group average, see if we can safely clear the group average array
                            #       note that early re-ref data still might be needed at late re-ref


                        else:
                            # not all required information for early re-ref is available, we will have to wait
                            # an iteration (over the rest of the channels) for the information to become available

                            # check whether it is optimized for memory, if so, clear
                            if priority == 'mem':

                                #print(channel + ": clearing channel from mem")

                                # remove the reference to the numpy array, this way the memory should be available for collection
                                channel_data[channel] = None
                                gc.collect()

                                # since we need to reload the channel the next iteration, we will also have to high-pass it again
                                channel_hp_applied[channel] = False

                            # continue to the next channel
                            continue

                #
                # Line noise removal
                #
                if line_noise_removal is not None and not channel_lnr_applied[channel]:

                    #print(channel + ": LNR - " + str(line_noise_removal))

                    # Filter the data
                    channel_data[channel] = signal.filtfilt(lnr_numerator, lnr_denominator, channel_data[channel], padtype='odd', padlen=3 * (max(len(lnr_numerator), len(lnr_denominator)) - 1))

                    # set line noise removal to have been applied to the channel-data in memory
                    channel_lnr_applied[channel] = True


                #
                # Late re-referencing
                #

                # check if late re-referencing needed
                if late_reref is not None:

                    #
                    # Late re-referencing collect
                    #

                    # check if the data of this channel (at this point) is already collected for the late re-reference groups
                    if not channel_late_reref_collected[channel]:
                        # late not collected
                        #print(channel + ": Collecting late reref values from channel")

                        # loop over the late-reref groups
                        for group in late_req_groups:

                            # check if this group requires this channel
                            if channel in late_group_channels_collected[str(group)].keys():

                                # check if the channel selection for late re-referencing is based on the variance
                                if late_reref.late_group_reselect_varPerc is None:
                                    # late re-referencing does not require channel selection based on variance

                                    # create arrays to hold the group common average data if not yet initialized
                                    if late_group_data[str(group)] is None:
                                        late_group_data[str(group)] = np.zeros((len(channel_data[channel]),), dtype=np.float64)
                                        if late_reref.channel_exclude_epochs is not None:
                                            late_group_numdata[str(group)] = np.zeros((len(channel_data[channel]),), dtype=np.uint16)

                                    # check if there are exclusion epochs
                                    if late_reref.channel_exclude_epochs is None or channel not in late_reref.channel_exclude_epochs:

                                        # no exclusion epochs, just add the whole channel
                                        late_group_data[str(group)] += channel_data[channel]

                                        # count the number of samples added to the total if needed
                                        if late_reref.channel_exclude_epochs is not None:
                                            late_group_numdata[str(group)] += 1

                                    else:
                                        # channel has exclusion epochs

                                        # create a binary numpy vector of the sample to include
                                        channel_includes = np.ones((len(channel_data[channel]),), dtype=bool)
                                        for channel_exclude_epoch in late_reref.channel_exclude_epochs[channel]:
                                            exclude_sample_start = int(round(channel_exclude_epoch[0] * data_reader.sampling_rate))
                                            exclude_sample_end = int(round(channel_exclude_epoch[1] * data_reader.sampling_rate))
                                            channel_includes[exclude_sample_start:exclude_sample_end] = 0

                                        # add the channel (taking into account on the inclusion vector)
                                        late_group_data[str(group)] += (channel_data[channel] * channel_includes)
                                        late_group_numdata[str(group)] += channel_includes


                                else:
                                    # late re-referencing requires channel selection based on variance
                                    # Note: the condition averages are also retrieved together with the condition variance in one go, both are
                                    #       used later to determine to calculate common average for each condition in the reference

                                    #
                                    if channel_idx is None:
                                        try:
                                            channel_idx = retrieve_channels.index(channel)
                                        except ValueError:
                                            logging.error('Could not find epoch channel ' + channel + ' in the list of channels to retrieve')
                                            raise RuntimeError('Could not find late ref channel in retrieve list')

                                    # check if there are exclusion epochs
                                    channel_exclude_epochs = None
                                    if late_reref.channel_exclude_epochs is not None and channel in late_reref.channel_exclude_epochs:
                                        channel_exclude_epochs = late_reref.channel_exclude_epochs[channel]

                                    # create arrays to hold the group variances data if not yet initialized
                                    if late_group_data[str(group)] is None:

                                        # TODO: maybe improve
                                        # Note: deliberately make this array larger so that the index of the channels in the 'data' variable and the 'late_group_data[str(group)]' variable can match
                                        late_group_data[str(group)] = allocate_array((len(retrieve_channels), len(onsets)))
                                        #late_group_data[str(group)] = allocate_array((len(late_reref.groups[group]), len(onsets)))


                                    # Note 1: 'data' will hold the averages to be used to the common averages per channel per condition later, after the
                                    #         common averages are determined, the values in data will be cleared/overwritten with the actual output data
                                    __subload_data_epoch_averages__from_channel__by_condition_trials(data, None,
                                                                                                     data_reader, channel_idx, channel, channel_data[channel],
                                                                                                     onsets, trial_epoch,
                                                                                                     0, None,
                                                                                                     out_of_bound_method,
                                                                                                     metric_callbacks=None,
                                                                                                     exclude_epochs=channel_exclude_epochs,
                                                                                                     var_epoch=(.015, .5), ref_var=late_group_data[str(group)])

                                # flag channel within the group as collected
                                late_group_channels_collected[str(group)][channel] = True

                                # check whether all the channels in the group are collected
                                if all(late_group_channels_collected[str(group)].values()):

                                    # check if the channel selection for late re-referencing is based on the variance
                                    if late_reref.late_group_reselect_varPerc is None:
                                        # late re-referencing does not require channel selection based on variance

                                        # take the average over the total
                                        # (if specific epochs were excluded, each sample should be divided by its own number)
                                        if late_reref.channel_exclude_epochs is not None:
                                            late_group_data[str(group)] /= late_group_numdata[str(group)]

                                            # clear the array was used divide the total to
                                            del late_group_numdata[str(group)]

                                        else:
                                            late_group_data[str(group)] /= len(late_group_channels_collected[str(group)])

                                    else:
                                        # late re-referencing requires channel selection based on variance

                                        # check minimum number of channels with variances within the re-referencing group
                                        # TODO: now set to 5, discuss a default and put in config. Perhaps as warning?
                                        variance_channels_per_condition = np.sum(~np.isnan(late_group_data[str(group)]), axis=0)
                                        if np.any(variance_channels_per_condition < 5):
                                            logging.error('One or more stim-pairs/conditions have too few channel variances within the current late re-referencing group ' + str(group) + ' to perform channel selection by variance.\n'
                                                          'If re-referencing with CAR per headbox, consider using just CAR.\n')
                                            raise RuntimeError('Too few channel variances to perform channel selection')

                                        # determine the variance threshold (below which to include channels) per condition
                                        variance_threshold_per_condition = np.nanquantile(late_group_data[str(group)], late_reref.late_group_reselect_varPerc, axis=0)

                                        # create a matrix to hold the trial epoch common average for each condition
                                        group_CAR_per_condition = allocate_array((len(onsets), trial_num_samples))

                                        # loop over the conditions
                                        for condition_index in range(late_group_data[str(group)].shape[1]):

                                            # TODO: optionally mention condition name (stim-pairs)
                                            logging.info('Re-referencing group: ' + str(group) + ' - Condition index: ' + str(condition_index))
                                            logging.info('    - R2 threshold: ' + str(round(variance_threshold_per_condition[condition_index], 1)) + '  (at quantile: ' + str(late_reref.late_group_reselect_varPerc) + ')')

                                            # retrieve the indices of the channels that should be used for re-referencing based on the threshold for this condition
                                            lowest_var_channels = (late_group_data[str(group)][:, condition_index] < variance_threshold_per_condition[condition_index]).nonzero()[0]

                                            # output channels with variances
                                            var_channels_print = [retrieve_channels[var_channel] + ' (' + str(round(late_group_data[str(group)][var_channel, condition_index], 1)) + ')' for var_channel in lowest_var_channels]
                                            var_channels_print = [str_print.ljust(len(max(var_channels_print, key=len)), ' ') for str_print in var_channels_print]
                                            logging.info(multi_line_list(var_channels_print, LOGGING_CAPTION_INDENT_LENGTH, '    - Channels: ', 5, '   ', str(len(var_channels_print)) + ' of ' + str(len(late_reref.groups[group]))))

                                            # check minimum number of channels within the condition
                                            # TODO: now set to 5, discuss a default and put in config
                                            if len(lowest_var_channels) < 5:
                                                logging.error('Too few channels (' + str(len(lowest_var_channels))  + ' from a group of ' + str(len(late_reref.groups[group])) + ') left for re-referencing after applying the variance threshold (' + str(variance_threshold_per_condition[condition_index]) + ') for this stim-pair/condition.\n'
                                                              'If re-referencing with CAR per headbox, consider using just CAR.\n')
                                                raise RuntimeError('Too few channel after variance thresholding to perform channel selection')

                                            # calculate condition common average
                                            group_CAR_per_condition[condition_index, :] = np.nanmean(data[lowest_var_channels, condition_index, :], axis=0)


                                        # clear variance data and instead store the group common averages (per condition) there
                                        del late_group_data[str(group)]
                                        late_group_data[str(group)] = group_CAR_per_condition


                        # flag that for this channel the late re-ref values have been collected
                        channel_late_reref_collected[channel] = True

                        # update the progress bar
                        update_progressbar()

                        # check if channel is no longer needed after this (for epoching)
                        # Note: this also means the channel was only loaded for early or late re-referencing
                        if channel not in channel_epoched.keys():
                            # channel-data is no longer needed at all

                            # remove the reference to the numpy array, this way the memory should be available for collection
                            channel_data[channel] = None
                            gc.collect()

                            # skip to next channel
                            continue



                    #
                    # Late re-referencing apply
                    #

                    # since late re-referencing and epoching are the last steps, there is no storing of the channel data
                    # with late re-referencing applied. The channel data either stays as it arrived at this point (when
                    # optimized for speed; waiting for being able to perform the late re-ref) or is reprocessed from the start
                    # to the same state (when optimized for memory, then the late re-ref will be applied) or it is immediately
                    # late re-referenced and epoched (and the channel-data cleared)

                    # retrieve the late re-ref group for this channel
                    group = late_reref.channel_group[channel]

                    # check if all the late re-referencing information is available yet (late average for this group)
                    if all(late_group_channels_collected[str(group)].values()):
                        # all required information is available, perform late re-referencing on the channel
                        # print(channel + ": performing late reref on channel")

                        if late_reref.late_group_reselect_varPerc is None:
                            # late re-referencing does not require channel selection based on variance

                            # perform late re-ref using reref_values
                            channel_data[channel] -= late_group_data[str(group)]

                            # TODO: if this is the latest channel to use the late group average, see if we can safely clear the group average array

                    else:
                        # not all required information for late re-ref is available, we will have to wait
                        # an iteration (over the rest of the channels) for the information to become available

                        # check whether it is optimized for memory, if so, clear
                        if priority == 'mem':

                            #print(channel + ": clearing channel from mem")

                            # remove the reference to the numpy array, this way the memory should be available for collection
                            channel_data[channel] = None
                            gc.collect()

                            # since we need to reload the channel the next iteration, we will also have to high-pass, early
                            # re-ref and remove line-noise again
                            channel_hp_applied[channel] = False
                            channel_early_applied[channel] = False
                            channel_lnr_applied[channel] = False

                        # continue to the next channel
                        continue


                #
                # Epoch-ing
                #

                # epoch the channel data
                #print(channel + ": epoch")
                try:

                    # retrieve the index of the channel in the requested list (so it can be placed in the correct spot of the return matrix)
                    # Note: Channels that are needed for re-referencing but not for epoch-ing should not get this far due
                    #       to the check/continue statements in the re-referencing collects sections above
                    if channel_idx is None:
                        try:
                            channel_idx = retrieve_channels.index(channel)
                        except ValueError:
                            logging.error('Could not find epoch channel ' + channel + ' in the list of channels to retrieve')
                            raise RuntimeError('Could not find epoch channel in retrieve list')

                    # check if late re-referencing with based on variance is needed
                    CAR_per_condition = None
                    if late_reref is not None and late_reref.late_group_reselect_varPerc is not None:

                        # retrieve the late re-ref group for this channel
                        group = late_reref.channel_group[channel]

                        # clear the data for this channel
                        data[channel_idx, :, :] = np.nan

                        #
                        CAR_per_condition = late_group_data[str(group)]


                    if average:
                        # epoch and average

                        __subload_data_epoch_averages__from_channel__by_condition_trials(data, metric_values,
                                                                                         data_reader, channel_idx, channel, channel_data[channel],
                                                                                         onsets, trial_epoch,
                                                                                         baseline_method, baseline_epoch,
                                                                                         out_of_bound_method,
                                                                                         metric_callbacks,
                                                                                         CAR_per_condition=CAR_per_condition)

                    else:
                        # epoch only
                        __epoch_data__from_channel_data__by_trials(data,
                                                                   channel_idx, channel_data[channel],
                                                                   data_reader.sampling_rate,
                                                                   onsets, trial_epoch,
                                                                   baseline_method, baseline_epoch, out_of_bound_method)

                except (MemoryError, RuntimeError):
                    raise RuntimeError('Error upon loading and epoching data')

                #print(channel + ": clearing channel from mem")

                # clear channel data from the channel-data matrix
                # (all we needed from this channel is either in the re-ref average arrays or in the epoch data-matrix now)
                channel_data[channel] = None
                gc.collect()

                # mark channel as epoch-ed (fully processed)
                channel_epoched[channel] = True

                # update the progress bar
                update_progressbar()

    #
    if average:
        return data_reader.sampling_rate, data, metric_values
    else:
        return data_reader.sampling_rate, data

