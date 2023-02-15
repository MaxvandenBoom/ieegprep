"""
Universal IEEG data reader class


=====================================================
Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
from abc import abstractmethod


#
# constants
#
VALID_FORMAT_EXTENSIONS         = ('.mefd', '.edf', '.vhdr', '.vmrk', '.eeg')   # valid data format to search for (European Data Format, BrainVision and MEF3)


class IeegDataReader(object):

    initialized = False
    data_path = ''
    data_preload = False

    data_format = ''                # the type of data ('bv', 'edf' or 'mef3')
    sampling_rate = -1              # the sampling rate of the dataset (from metadata, assumes same rate over channels)
    num_samples = -1                # the total number of samples per channel (from metadata, assumes same rate over channels)
    channel_names = []              # all of the channel names

    def __new__(cls, data_path, data_preload=False):

        # return a sub-class instance depending on the format
        try:
            data_extension = data_path[data_path.rindex("."):]
            if data_extension == '.edf':
                from .EdfReader import EdfReader
                return super(IeegDataReader, cls).__new__(EdfReader)

            elif data_extension == '.vhdr' or data_extension == '.vmrk' or data_extension == '.eeg':
                from .BrainVisionReader import BrainVisionReader
                return super(IeegDataReader, cls).__new__(BrainVisionReader)

            elif data_extension == '.mefd':
                from .Mef3Reader import Mef3Reader
                return super(IeegDataReader, cls).__new__(Mef3Reader)

            else:
                logging.error('Unknown data format (' + data_extension + ')')
                raise ValueError('Unknown data format')

        except ValueError:
                logging.error('Invalid data path (' + data_path + ')')
                raise ValueError('Invalid data path')

    def __init__(self, data_path, data_preload=False):
        self.data_path = data_path
        self.data_preload = data_preload

    @abstractmethod
    def close(self): pass

    @abstractmethod
    def retrieve_channel_data(self, channel_name, ensure_own_data=True): pass

    @abstractmethod
    def retrieve_sample_range_data(self, channels, sample_start, sample_end, ensure_own_data=True): pass

    @staticmethod
    def check_sample_arguments(start_sample, end_sample, number_of_samples, number_of_channels):
        """
        Helper function to check the start_sample and end_sample arguments

        Args:
            start_sample (int or list):     The start-point in time (in samples) to start reading from (0-based). This argument
                                            can either be a single value to indicate the start value of a single range that
                                            should be retrieved over all the requested channels; or this argument can be a
                                            list of start values to retrieve multiple ranges (start- and end-points) for all
                                            the requested channels. When retrieving multiple ranges, make sure the number of
                                            values in the start_sample and the end_sample arguments match, and the size of the
                                            ranges (in samples) are equal.
            end_sample (int or list):       The sample to end the reading (0-based). This argument can either be a single
                                            value to indicate the end-sample of a single range that should be retrieved
                                            over all the requested channels; or this argument can be a list of end values
                                            to retrieve multiple ranges (start- and end-points) for all the requested
                                            channels. When retrieving multiple ranges, make sure the number of values in
                                            the start_sample and the end_sample arguments match, and the size of the ranges
                                            (in samples) are equal. Also note that because the range is 0-based, data are
                                            loaded "up-till" the range end-index, so the result does not include the value
                                            at the end-index (e.g. a requested sample range of 0-3 will return the first 3 values,
                                            being the values at [0], [1], [2]). A value of -1 (default) or -1 value in the list
                                            of values represents the latest sample of the time-series (as indicated by the header)
            number_of_samples (int):        The number of samples (for each channel) in the dataset, is used to check boundaries
            number_of_channels (int):       The number of channels, is only used to output the output matrix dimension in one go

        Returns:
            single_range:                   True if single range, False if multiple ranges
            output_dimensions (tuple):      The dimensions that the output matrix should be based on the ranges
            range_start (int or list):      Int (single range) or list (multiple ranges) containing the start sample(s) of each requested range
            range_length (int or list):     Int (single range) or List (multiple ranges) containing the length(s) of each requested range

        Raises:
            TypeError:                      Raised when input argument types are wrong
            RuntimeError:                   Raised upon a processing error
        """

        def check_input_range(in_start_sample, in_end_sample):

            # fraction
            if isinstance(in_start_sample, float) and not in_start_sample.is_integer():
                logging.error('invalid start_sample argument (' + str(in_start_sample) + '), cannot be a factional value')
                raise TypeError('invalid start_sample argument')
            if isinstance(in_end_sample, float) and not in_end_sample.is_integer():
                logging.error('invalid end_sample argument (' + str(in_end_sample) + '), cannot be a factional value')
                raise TypeError('invalid end_sample argument')

            # start
            if in_start_sample < 0:
                logging.error('invalid start_sample input argument (' + str(in_start_sample) + '), should be 0 or higher')
                raise RuntimeError('invalid start_sample argument')
            if in_start_sample > number_of_samples:
                logging.error('invalid start_sample input argument (' + str(in_start_sample) + '), exceeds the number of sample in the data (' + str(number_of_samples) + ')')
                raise RuntimeError('invalid start_sample argument')

            # end
            if in_end_sample == -1:
                out_end_sample = number_of_samples
            elif in_end_sample < 1:
                logging.error('invalid end_sample input argument (' + str(in_end_sample) + '), should either -1 or >0')
                raise RuntimeError('invalid end_sample argument')
            elif in_end_sample > number_of_samples:
                logging.error('invalid end_sample input argument (' + str(in_end_sample) + '), exceeds the number of sample in the data (' + str(number_of_samples) + ')')
                raise RuntimeError('invalid end_sample argument')
            else:
                out_end_sample = in_end_sample

            # range
            if out_end_sample < in_start_sample:
                logging.error('invalid start_sample and end_sample input arguments, end-point sample (' + str(out_end_sample) + ') should be after the start-point sample (' + str(start_sample) + ')')
                raise RuntimeError('invalid start_sample and end_sample arguments')
            if out_end_sample - in_start_sample == 0:
                logging.error('invalid start_sample (' + str(in_start_sample) + ') and end_sample (' + str(out_end_sample) + ') input arguments, the requested range length is 0')
                raise RuntimeError('invalid start_sample and end_sample arguments')

            return in_start_sample, out_end_sample, out_end_sample - in_start_sample


        # type (specific) checks
        if (isinstance(start_sample, list) or isinstance(start_sample, tuple)) and (isinstance(end_sample, float) or isinstance(end_sample, int)) or \
           (isinstance(end_sample, list) or isinstance(end_sample, tuple)) and (isinstance(start_sample, float) or isinstance(start_sample, int)):
            logging.error('Start_sample and end_sample input argument types do not match. Both arguments should either be single values or lists/tuples of values')
            raise TypeError('Start_sample and end_sample input argument types do not match')

        elif (isinstance(start_sample, float) or isinstance(start_sample, int)) and (isinstance(end_sample, float) or isinstance(end_sample, int)):
            # single value checks

            # flag as single range and check input
            single_range = True
            try:
                range_start, _, range_length = check_input_range(start_sample, end_sample)
            except RuntimeError:
                raise RuntimeError('Error on input arguments')

            # set output dimensions
            output_dimensions = [number_of_channels, range_length]

        elif (isinstance(start_sample, list) or isinstance(start_sample, tuple)) and (isinstance(end_sample, list) or isinstance(end_sample, tuple)):
            # multiple value checks

            # check the lists
            if len(start_sample) != len(end_sample):
                logging.error('Number of values in the start_sample and end_sample arguments do not match. To retrieve multiple ranges/epochs, make sure there are an equal amount of start-points and end-points in the input arguments')
                raise RuntimeError('Number of values in the start_sample and end_sample arguments do not match')
            if len(start_sample) == 0:
                logging.error('The start_sample and end_sample arguments are empty')
                raise RuntimeError('Empty start_sample and end_sample arguments')

            # flag as multiple ranges and check input ranges
            single_range = False
            range_start = []
            range_length = []
            for index in range(len(start_sample)):
                try:
                    i_range_start, _, i_range_length = check_input_range(start_sample[index], end_sample[index])
                    range_start.append(i_range_start)
                    range_length.append(i_range_length)
                    del i_range_start, i_range_length

                except RuntimeError:
                    raise RuntimeError('Error on input argument list pair: ' + str(start_sample[index]) + '-' + str(end_sample[index]))

            # check range lengths
            if range_length.count(range_length[0]) != len(range_length):
                logging.error('The length of the ranges in the start_sample and end_sample arguments do not match. To retrieve multiple ranges/epochs, make sure that each of the start- and end-point combinations result in a range of the same length')
                raise RuntimeError('The length of the ranges in the start_sample and end_sample arguments do not match')

            # set output dimensions
            output_dimensions = [number_of_channels, range_length[0], len(range_length)]

        else:
            logging.error('Invalid input argument type(s). Both arguments should either be single values or lists/tuples of values')
            raise TypeError('Invalid input argument type(s)')

        # return whether single or multiple ranges and the output matrix dimensions and the ranges
        return single_range, output_dimensions, range_start, range_length
