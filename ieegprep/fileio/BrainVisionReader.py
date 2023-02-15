"""
Functions to read BrainVision files


=====================================================
Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

Adapted from Fieldtrip (by Robert Robert Oostenveld) while replicating some additional header logic from the MNE
package (Teon Brooks, Christian Brodbeck, Eric Larson, Jona Sassenhagen, Phillip Alday, Okba Bekhelifi, Stefan Appelhoff)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import os
import logging
import numpy as np
from configparser import ConfigParser
from .IeegDataReader import IeegDataReader
from ieegprep.utils.misc import allocate_array


class BrainVisionReader(IeegDataReader):

    bv_hdr = None                   # header
    bv_data = None                  # data (only used on preload)

    def __init__(self, data_path, data_preload=False):
        super().__init__(data_path, data_preload)
        self.data_format = 'bv'

        # load header
        try:
            self.bv_hdr = self.bv_read_header(self.data_path)
        except (FileNotFoundError, IOError):
            logging.error('Could not read BrainVision header')
            raise RuntimeError('Could not read BrainVision header')

        # retrieve the sample-rate, total number of samples and channel names
        self.sampling_rate = self.bv_hdr['sampling_rate']
        self.num_samples = self.bv_hdr['number_of_samples']
        self.channel_names = self.bv_hdr['channel_names']

        # (optionally) preload data
        if self.data_preload:
            try:
                _, self.bv_data = self.bv_read_data(filepath=self.data_path, hdr=self.bv_hdr)
            except (FileNotFoundError, IOError, TypeError, RuntimeError):
                logging.error('Could not read BrainVision data')
                raise RuntimeError('Could not read BrainVision data')


    def close(self):
        del self.bv_hdr
        if self.bv_data is not None:
            del self.bv_data


    def retrieve_channel_data(self, channel_name, ensure_own_data=True):
        """
        Retrieve the channel data (mef = numpy data-array, mne = numpy data-view)

        Args:
            channel_name:                   The name of the channel for which to retrieve the data
            ensure_own_data (bool):         Should ensure the return a numpy array has it's own data (is not a view)

        Returns:
            Numpy array with data

        Raises:
            LookupError:                    Raised when the channel name cannot be found
            RuntimeError:                   Raised when unable to retrieve channel data
        """

        # try to find the channel name
        try:
            channel_index = self.bv_hdr['channel_names'].index(channel_name)
        except ValueError:
            raise LookupError('Could not find channel')

        # determine whether the data is preloaded
        if self.bv_data is None:
            # not preloaded

            # load the channel data
            try:
                _, channel_data = self.bv_read_data(filepath=self.data_path,
                                                    hdr=self.bv_hdr,
                                                    channels=(channel_name,))
            except Exception:
                logging.error('Could not read BrainVision data')
                raise RuntimeError('Could not read BrainVision data')

            return channel_data.squeeze()

        else:
            # preloaded data

            # return the channel data
            if ensure_own_data:
                return self.bv_data[channel_index, :].copy()
            else:
                return self.bv_data[channel_index, :]


    def retrieve_sample_range_data(self, channels, sample_start, sample_end, ensure_own_data=True):
        """
        Retrieve a specific range of BrainVision data for the requested channels

        Args:
            channels (str, list or tuple):  The channel(s) for which to retrieve the data.
                                            If empty, all channels will be retrieved
            sample_start (int):             The start-point in time (in samples) to start reading from (0-based)
            sample_end (int):               The sample to end the reading at (0-based)
            ensure_own_data (bool):         Should ensure the return a numpy array has it's own data (is not a view)

        Returns:
            data                            A float data matrix containing the signal data (of the requested channels)
                                            formatted as <channels> x <samples/time>

        Raises:
            LookupError:                    Raised when a channel name cannot be found
            RuntimeError:                   Raised when unable to retrieve data
        """

        # check/prepare the channels argument
        if isinstance(channels, str):
            channels = [channels]
        if isinstance(channels, tuple):
            channels = list(channels)
        if channels is None or len(channels) == 0:
            channels = self.channel_names

        if self.bv_data is None:
            # data is not preloaded

            # load the data
            try:
                _, np_sample_data = self.bv_read_data(filepath=self.data_path, hdr=self.bv_hdr,
                                                      channels=channels,
                                                      start_sample=sample_start, end_sample=sample_end)
            except Exception:
                logging.error('Could not read BrainVision data')
                raise RuntimeError('Could not read BrainVision data')

            # return the data (from 2-d numpy array to list of numpy arrays)
            return list(np_sample_data)

        else:
            # data is preloaded

            # create a list with the numpy arrays
            sample_data = [None] * len(channels)

            # loop over the requested channels
            for counter in range(len(channels)):

                # find the channel index
                try:
                    channel_index = self.bv_hdr['channel_names'].index(channels[counter])
                except ValueError:
                    raise LookupError('Could not find channel')

                # pick the slice
                if ensure_own_data:
                    sample_data[counter] = self.bv_data[channel_index, sample_start:sample_end].copy()
                else:
                    sample_data[counter] = self.bv_data[channel_index, sample_start:sample_end]

        return sample_data


    #
    # BrainVision header and data reading functions
    #

    @staticmethod
    def bv_read_header(filepath) -> dict:
        """
        Read a BrainVision header (.vhdr) file

        Args:
            filepath (str):                 The path to the BrainVision header file

        Returns:
            A dictionary with header information

        Raises:
            FileNotFoundError:              Raised when unable to find the file
            IOError:                        Raised on any error that occurs while reading or parsing the header file

        Examples:
            hdr = bv_read_header('~/dataset.vhdr')
        """

        #
        # Read/check the file
        #

        # check file existence
        if not os.path.exists(filepath):
            logging.error('Header file \'' + filepath + '\' could not be found')
            raise FileNotFoundError('No such file or directory: \'' + filepath + '\'')

        # check extension
        if os.path.splitext(filepath)[1].lower() not in ('.vhdr',):
            logging.error('The header file \'' + filepath + '\' has an invalid extension (should be .vhdr)')
            raise IOError('Header file \'' + filepath + '\' has an invalid extension')

        # read
        with open(filepath, 'rb') as file:

            # retrieve and check the header
            header_line = file.readline().decode(encoding='ascii', errors='ignore').strip()
            if not header_line.startswith('BrainVision Data Exchange Header File Version ') and not \
                   header_line.startswith('Brain Vision Data Exchange Header File Version '):
                logging.error('Invalid or missing header in \'' + filepath + '\'')
                raise IOError('Invalid or missing header in .vhdr file')

            version = header_line[-3:]
            if not version == '1.0' and not version == '2.0':
                logging.error('Header file indicates an unknown BrainVision format version \'' + filepath + '\'')
                raise IOError('Unknown BrainVision format version')

            # parse the rest of the INI file
            try:
                config = ConfigParser()
                config.read_string(file.read().decode(encoding='UTF-8', errors='ignore'))
            except:
                logging.error('Error while reading and parsing the BrainVision header file \'' + filepath + '\'')
                raise IOError('Unable to read/parse .vhdr file')


        #
        # Obtain information from the header
        #

        def getField(parser, section, field, type, optional=False):
            try:
                if type == 'str':
                    return parser.get(section, field)
                elif type == 'int':
                    return parser.getint(section, field)
                elif type == 'float':
                    return parser.getfloat(section, field)
            except:
                if optional:
                    return None
                else:
                    logging.error('Could not read/parse mandatory option \'' + field + '\' (section \'' + section + '\') from the BrainVision header file \'' + filepath + '\'')
                    raise IOError('Unable to read/parse mandatory option from .vhdr file')

        hdr = dict()
        try:
            hdr['data_file']            = getField(config, 'Common Infos', 'DataFile', 'str')
            hdr['number_of_channels']   = getField(config, 'Common Infos', 'NumberOfChannels', 'int')
            hdr['Sampling_interval']    = getField(config, 'Common Infos', 'SamplingInterval', 'float')
            hdr['sampling_rate']        = 1e6 / hdr['Sampling_interval']
            hdr['data_points']          = getField(config, 'Common Infos', 'DataPoints', 'int', True)

            # format
            hdr['data_format']          = getField(config, 'Common Infos', 'DataFormat', 'str')
            if hdr['data_format'] == 'BINARY':
                hdr['data_binary_format']          = getField(config, 'Binary Infos', 'BinaryFormat', 'str')
                if not hdr['data_binary_format'] in ('INT_16', 'INT_32', 'IEEE_FLOAT_32'):
                    logging.error('Unknown data binary format \'' + hdr['data_binary_format'] + '\' in BrainVision header file \'' + filepath + '\'')
                    raise IOError('Unknown data binary format in .vhdr file')

            elif hdr['data_format'] == 'ASCII':

                skip_lines                          = getField(config, 'ASCII Infos', 'SkipLines', 'int', True)
                hdr['data_ascii_skip_lines']        = skip_lines if skip_lines is not None else 0
                skip_columns                        = getField(config, 'ASCII Infos', 'SkipColumns', 'int', True)
                hdr['data_ascii_skip_columns']      = skip_columns if skip_columns is not None else 0
                hdr['data_ascii_decimal_symbol']    = getField(config, 'ASCII Infos', 'DecimalSymbol', 'str', True)

            else:
                logging.error('Unknown data format \'' + hdr['data_format'] + '\' in BrainVision header file \'' + filepath + '\'')
                raise IOError('Unknown data format in .vhdr file')

            # orientation
            hdr['data_orientation']     = getField(config, 'Common Infos', 'DataOrientation', 'str')
            if not hdr['data_orientation'] in ('MULTIPLEXED', 'VECTORIZED'):
                logging.error('Unknown data orientation \'' + hdr['data_orientation'] + '\' in BrainVision header file \'' + filepath + '\'')
                raise IOError('Unknown data orientation in .vhdr file')


            #
            # channel information
            #

            hdr['channel_names']             = []
            hdr['channel_other_info']        = []
            if hdr['number_of_channels'] != 0:
                for i in range(1, hdr['number_of_channels'] + 1):

                    # retrieve the channel info
                    chan_str  = ('Ch' + str(i))
                    chan_info = getField(config, 'Channel Infos', chan_str, 'str').split(',')

                    # Note: It is not clear whether the resolution and unit are related in the Brainvision format. It
                    #       could be that they need to match (where the resolution is the conversion factor to get to
                    #       the unit). However, the 'BrainVision Core Data Format' documentation also seems to mention
                    #       an example that speaks of 'Resolution / Unit' with reported values of '0.5 μV'. These imply
                    #       that resolution and unit are independent values.

                    # if no resolution is set, then set resolution to 1 (as specified in 'BrainVision Core Data Format' documentation)
                    chan_res = float(chan_info[2]) if chan_info[2] == '' else 1

                    # if no unit is available, then set unit to µV (as specified in 'BrainVision Core Data Format' documentation)
                    chan_unit = str(chan_info[3]) if len(chan_info) > 3 and not chan_info[3] == '' else 'µV'

                    # determine the unit for specific unit codes (otherwise ensure multiplication by 1)
                    # Note: setting this unit is used to later allow us to convert to the requested output (uV, mV or V)
                    if chan_unit.lower() in ('\u03BCv', '\u00B5v', '\x83\xCAv', 'uv'):
                        units_to_v_gain = 1e-6
                    elif chan_unit.lower() == 'mv':
                        units_to_v_gain = 1e-3
                    elif chan_unit.lower() == 'v':
                        units_to_v_gain = 1
                    else:
                        #logging.warning('The unit on channel ' + hdr['channel_names'][count] + ' is not unknown, assume multiplication by 1')
                        units_to_v_gain = 1

                    # store (and remove some troublesome characters)
                    hdr['channel_names'].append(chan_info[0].replace(r'\1', ','))
                    hdr['channel_other_info'].append({"ref": chan_info[1],
                                                      "resolution": chan_res,
                                                      "unit": chan_unit.replace('\xc2', ''),
                                                      "units_to_V_gain": units_to_v_gain})

        except:
            logging.error('Unable to obtain the required fields from the BrainVision header file \'' + filepath + '\'')
            raise IOError('Unable to obtain the required field from the .vhdr file')


        #
        # determine the number of samples
        #

        # determine data filepath
        base_dir = os.path.dirname(filepath)
        data_file = os.path.splitext(os.path.basename(filepath))[0] + '.eeg'
        if not data_file == hdr['data_file']:
            logging.warning('The name of the data-file, based on the header\'s filename (\'' + data_file + '\') and DataFile field (\'' + hdr['data_file'] + '\'), are not in agreement.\nUsing name based on header\'s filename')
        data_file = os.path.join(base_dir, data_file)

        # infer number of samples
        if hdr['data_format'] == 'BINARY':

            # determine the size of the data file
            try:
                with open(data_file, 'rb') as file:
                    num_bytes = file.seek(0, os.SEEK_END)
            except:
                logging.error('Error while getting the file size of the BrainVision data file \'' + data_file + '\', unable to determine number of samples')
                raise IOError('Unable to the file size of the .eeg file, unable to determine number of samples')

            # calculate the number of samples
            if hdr['data_binary_format'] == 'INT_16':
                hdr['number_of_samples'] = int(num_bytes / (hdr['number_of_channels'] * 2))
            elif hdr['data_binary_format'] == 'INT_32' or hdr['data_binary_format'] == 'IEEE_FLOAT_32':
                hdr['number_of_samples'] = int(num_bytes / (hdr['number_of_channels'] * 4))

        elif hdr['data_format'] == 'ASCII':

            if hdr['data_points'] is not None:
                hdr['number_of_samples'] = hdr['data_points']
            else:

                if hdr['data_orientation'] == 'VECTORIZED':
                    logging.warning('No DataPoints field in the header file, inferring number of samples from the (ASCII) data-file assuming VECTORIZED data orientation')

                    # determine the number of values in the first line
                    try:
                        with open(data_file, 'r') as file:
                            first_line = file.readline().decode(encoding='ascii', errors='ignore')
                    except:
                        logging.error('Error while reading BrainVision data file \'' + data_file + '\', unable to determine number of samples')
                        raise IOError('Error while reading the .eeg file, unable to determine number of samples')

                    # infer the number of samples
                    first_line_values = first_line.replace('  ', ' ').split(' ')
                    if len(first_line_values) < 2:
                        logging.error('Error while inferring the number of samples, the first line in the data file has no values')
                        raise IOError('Error while inferring the number of samples, the first line in the data file has no values')
                    hdr['number_of_samples'] = int(len(first_line_values) - 1)

                else:
                    # ASCII MULTIPLEXED
                    logging.warning('No DataPoints field in the header file and cannot infer number of samples from the (ASCII) data-files with a MULTIPLEXED orientation')

                    # count the lines
                    try:
                        num_lines = 0
                        for _ in open(data_file, 'r'):
                            num_lines += 1
                    except:
                        logging.error('Error while reading BrainVision data file \'' + data_file + '\', unable to determine number of lines/samples')
                        raise IOError('Error while reading the .eeg file, unable to determine number of lines/samples')

                    # set the number of samples as the number of lines minus the number of lines to skip, minus 1
                    hdr['number_of_samples'] = int(num_lines - hdr['data_ascii_skip_lines'] - 1)

        # TODO: more information like annotations, coordinates etc

        # return the header information
        return hdr


    @staticmethod
    def bv_read_data(filepath, hdr=None, channels=None, start_sample=0, end_sample=-1, unit='uV', use_memmap=True, chunked_read=False) -> tuple:
        """
        Read data from a BrainVision (.eeg) file

        Args:
            filepath (str):                 The path to the BrainVision data file
            hdr (dict):                     Optionally pass the header dictionary as provided by 'bv_read_header'. If this
                                            argument is not set, the header file will be parsed before retrieving the data.
                                            If set, this function will use the header that is provided (to prevent multiple
                                            header reads on multiple calls to this function)
            channels (str/list/tuple):      The names of the channels to return the signal data from. The order of channels
                                            in this input argument will determine the order of channels in the output matrix.
                                            If set to None (default), all channels will be read and ordered according to the
                                            header file (hdr['channel_names'])
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
            unit (str):                     The unit in which the data should be return ('uV' returns microVolts, 'mV' returns
                                            microVolts and 'V' returns Volts)
            use_memmap (bool):              Whether to use numpy's memmap (which wraps around mmap) while reading the data.
                                            If true, the data file is first loaded/cached into virtual memory (pagefile) to
                                            speed up (repetitive) reading. If false, the standard system read operations
                                            will be used (slightly slower for repetitive reading from the file but does not
                                            explicitly require virtual memory).
            chunked_read (bool):            Whether to read or transfer the data in chunks. If true, then reads from
                                            disk (stdIO) or transfers from virtual memory (memmap) are split up in xxMB
                                            chunks. If false, then often all data is read or transferred in one big block.

        Returns:
            hdr                             A dictionary with header information
            data                            A float data matrix containing the signal data (of the requested channels). When
                                            a single time-series is requested (one value as start_sample and one value as
                                            end_sample argument) then the matrix will be formatted as <channels> x <samples/time>,
                                            with the first dimension (rows) representing the channels (ordered based on
                                            the 'channels' input argument) and the second dimension (columns) the samples/time.
                                            If multiple ranges are requested (lists of values for the start_sample and end_sample
                                            arguments) then the return format will be <channels> x <samples/time> x <ranges>, so
                                            that the third dimension represents the requested ranges/epochs.

        Raises:
            FileNotFoundError:              Raised when unable to find the file
            IOError:                        Raised on any error that occurs while reading or parsing the data file
            TypeError:                      Raised when input argument types are wrong
            RuntimeError:                   Raised upon a processing error

        Examples:
            hdr, data = bv_read_data('~/dataset.eeg')
            hdr, data = bv_read_data('~/dataset.eeg', channels='CH01')
            hdr, data = bv_read_data('~/dataset.eeg', channels=('CH01', 'CH07'))
            hdr, data = bv_read_data('~/dataset.eeg', start_sample=1000, end_sample=2000)
            hdr, data = bv_read_data('~/dataset.eeg', start_sample=(1000, 6000), end_sample=(2000, 7000))
        """

        # check unit argument
        if unit.lower() not in ('uv', 'mv', 'v'):
            logging.error('Invalid unit ' + unit + ' to retrieve the data in. Only options are uV, mV or V')
            raise RuntimeError('Invalid unit')

        # check file existence
        if not os.path.exists(filepath):
            logging.error('Data file \'' + filepath + '\' could not be found')
            raise FileNotFoundError('No such file or directory: \'' + filepath + '\'')

        # check whether to load the header
        if hdr is None:

            # read the header information
            header_file = os.path.splitext(filepath)[0] + '.vhdr'
            try:
                hdr = BrainVisionReader.bv_read_header(header_file)
            except (FileNotFoundError, IOError):
                logging.error('Error while reading/parsing header file \'' + header_file + '\', this information is required to read the data file')
                raise IOError('Error while reading/parsing header file, this info required to read data file')

        else:
            # TODO: check fields in the header file that is passed?
            pass

        # check whether there are samples
        if hdr['number_of_samples'] == 0:
            logging.error('no samples in the data file according to the header')
            raise RuntimeError('no samples in the data')


        #
        # check/prepare channel input(s)
        #

        if isinstance(channels, str):
            channels = (channels,)

        # retrieve channel indices
        if channels is None or len(channels) == 0:
            channels = hdr['channel_names']
            channel_indices = list(range(0, len(channels)))
        else:
            channel_indices = []
            for channel in channels:
                try:
                    index = hdr['channel_names'].index(channel)
                    channel_indices.append(index)
                except ValueError:
                    logging.error('requested channel \'' + channel + '\' was not found')
                    raise IOError('Requested channel was not found')

        # list the gains of the channels in a list (for speed), consider the resolution and unit, also
        # determine whether perhaps all channels have a gain/multiplication of 1
        all_gains_one = True
        channel_gains = []
        for channel_idx in channel_indices:

            # retrieve the gain that needs to be applied to get the data in the requested unit
            unit_gain = hdr['channel_other_info'][channel_idx]['units_to_V_gain']
            if unit.lower() == 'uv':
                unit_gain *= 1e+6
            elif unit.lower() == 'mv':
                unit_gain *= 1e+3

            # calculate and store the total gain (based on resolution and unit)
            gain = hdr['channel_other_info'][channel_idx]['resolution'] * unit_gain
            channel_gains.append(gain)

            if gain != 1:
                all_gains_one = False

        channel_gains = tuple(channel_gains)

        #
        # check/prepare sample range input(s)
        #

        single_range, output_dimensions, ranges_start, ranges_length = IeegDataReader.check_sample_arguments(start_sample, end_sample,
                                                                                                             hdr['number_of_samples'],
                                                                                                             len(channel_indices))

        #
        # Read/check the data file
        #
        data_file = os.path.splitext(filepath)[0] + '.eeg'

        # check file existence
        if not os.path.exists(data_file):
            logging.error('Data file \'' + data_file + '\' could not be found')
            raise FileNotFoundError('No such file or directory: \'' + data_file + '\'')

        # determine the sample size in bytes
        if hdr['data_format'] == 'BINARY':
            if hdr['data_binary_format'] == 'INT_16':
                sample_size = 2
                sample_type = '<i2'
            elif hdr['data_binary_format'] == 'INT_32':
                sample_size = 4
                sample_type = '<i4'
            elif hdr['data_binary_format'] == 'IEEE_FLOAT_32':
                sample_size = 4
                sample_type = '<f4'

        # check if the system is 32-bit and memmap is enabled
        # (mapping on 32-bit systems is limited to 2GB files, so disable just in case)
        if use_memmap and not (sys.maxsize > 2 ** 32):
            logging.warning('This is a 32-bit system. Data will be read with standard IO operations (rather than memmap)')
            use_memmap = False

        #
        if hdr['data_format'] == 'BINARY' and hdr['data_binary_format'] in ('INT_16', 'INT_32', 'IEEE_FLOAT_32'):

            # init data matrix (with a specific output data-type)
            if all_gains_one:
                # all channel resolutions are 1, so no multiplication with a fractional number here.
                # since manipulations of the output data are expected to involve fractional precision, the output matrix
                # will be of the float datatype. Whether this is float32 or float64 depends on the data-type in the data file

                if hdr['data_binary_format'] == 'INT_16':
                    # int16 will be cast to float32 (using less memory than a float64). Float32 has 24-bit significand
                    # precision and can therefore easily hold the exact values in the range of int16
                    data = np.empty(output_dimensions, dtype=np.float32)
                    #allocate_array(output_dimensions, fill_value=np.nan, dtype=np.float32)

                elif hdr['data_binary_format'] == 'INT_32':
                    # int32 will be cast to float64. Float64 has 53-bit significand precision and can therefore hold the
                    # exact values in the range of int32 (float32 would not be able to)
                    data = np.empty(output_dimensions, dtype=np.float64)
                    #allocate_array(output_dimensions, fill_value=np.nan, dtype=np.float64)

                elif hdr['data_binary_format'] == 'IEEE_FLOAT_32':
                    # Keep in original float32 data type. Casting to float64 is possible but at this point would only add
                    # precision that is not really in the data while doubling the memory footprint.
                    data = np.empty(output_dimensions, dtype=np.float32)
                    #allocate_array(output_dimensions, fill_value=np.nan, dtype=np.float32)

            else:
                # at least one resolution is not 1, so a multiplication with - possibly - a fractional number might occur.
                # To facilitate as much precision as possible we will work with float64 numbers (at the expense of more memory)
                data = np.empty(output_dimensions, dtype=np.float64)
                #allocate_array(output_dimensions, fill_value=np.nan, dtype=np.float64)


            # local variable for number of channels
            num_channels_in_file = len(hdr['channel_names'])

            #
            if hdr['data_orientation'] == 'MULTIPLEXED':

                # Conclusions
                # - when reading samples, only part is cached
                # - when reading all or a single channel, the entire file is cached
                #
                # - after caching:
                #   - chunked reading always faster than not chunked reading, and smaller chunk sizes are faster
                #   - memmap is faster than fromfile

                # prevent per sample reading (due to reading non-chunked, standard IO and picking channels) because
                # it is notoriously slow, switch to chunked instead
                if not chunked_read and not use_memmap and not (channel_indices == list(range(0, num_channels_in_file))):
                    logging.warning('The current read settings (non-chunked, standard IO and picking channels will result in per sample reading, this is highly discouraged and exists for testing purposes only. Switching to chunked reading instead')
                    chunked_read = True

                #
                if chunked_read:
                    # chunked reading
                    #print('multiplexed - chunked - memmap = ' + str(use_memmap))

                    if use_memmap:
                        amem = np.memmap(data_file, dtype=sample_type, mode='r', offset=0, order='C')
                    else:
                        try:
                            fid_data = open(data_file, "rb")
                        except IOError:
                            logging.error('Error while opening data file \'' + data_file + '\'')
                            raise IOError('Error while opening data file')

                    # determine the size (in samples) chucks assuming ~10mb per read
                    # rounded down to the number of channels
                    chunk_size = ((int(10e6) // sample_size) // num_channels_in_file) * num_channels_in_file
                    assert (chunk_size != 0)

                    def read_binary_chunked(rb_chunk_size, rb_range_index, rb_start, rb_length):

                        # determine where to start reading from
                        if not use_memmap:
                            fid_data.seek(rb_start * num_channels_in_file * sample_size, os.SEEK_SET)

                        #
                        chunk_start = rb_start * num_channels_in_file
                        data_sample_pos = 0
                        while data_sample_pos < rb_length:

                            # determine length (in samples) of the current chunk
                            current_chunk_length = min((rb_length - data_sample_pos) * num_channels_in_file, rb_chunk_size)

                            if use_memmap:
                                chunk_data = amem[chunk_start:chunk_start + current_chunk_length].reshape(num_channels_in_file, -1, order='F')
                                chunk_start += current_chunk_length
                            else:
                                chunk_data = np.fromfile(fid_data, dtype=sample_type, count=current_chunk_length).reshape(num_channels_in_file, -1, order='F')
                            num_samples = chunk_data.shape[1]

                            # pick the channels and (optionally) apply the resolution
                            # Note 1. will automatically cast data-types (int->float) later if needed
                            # Note 2. deliberate more ifs, because function calls are expensive in python (performance wise) and these are iterated through a lot
                            for channel_count, channel_index in enumerate(channel_indices):
                                if channel_gains[channel_count] == 1:
                                    if rb_range_index is None:
                                        data[channel_count, data_sample_pos:data_sample_pos + num_samples] = chunk_data[channel_index, :]
                                    else:
                                        data[channel_count, data_sample_pos:data_sample_pos + num_samples, rb_range_index] = chunk_data[channel_index, :]
                                else:
                                    if rb_range_index is None:
                                        data[channel_count, data_sample_pos:data_sample_pos + num_samples] = chunk_data[channel_index, :].astype(np.float64) * channel_gains[channel_count]
                                    else:
                                        data[channel_count, data_sample_pos:data_sample_pos + num_samples, rb_range_index] = chunk_data[channel_index, :].astype(np.float64) * channel_gains[channel_count]

                            # move to the next chunk
                            data_sample_pos += num_samples

                    # read range(s)
                    if single_range:
                        read_binary_chunked(chunk_size, None, ranges_start, ranges_length)
                    else:
                        for range_index in range(len(ranges_start)):
                            read_binary_chunked(chunk_size, range_index, ranges_start[range_index], ranges_length[range_index])

                    # close file if needed
                    if not use_memmap and fid_data is not None:
                        fid_data.close()

                else:
                    # non-chunked reading

                    # check if all channels & in the same order as in the file (which means that we can just retrieve a blocks of samples and transpose the matrix)
                    if channel_indices == list(range(0, num_channels_in_file)):
                        #print('multiplexed - non-chunked - memmap = ' + str(use_memmap) + ' - allchan & sameorder')

                        # open
                        if use_memmap:
                            amem = np.memmap(data_file, dtype=sample_type, mode='r', offset=0, order='C')
                        else:
                            try:
                                fid_data = open(data_file, "rb")
                            except IOError:
                                logging.error('Error while opening data file \'' + data_file + '\'')
                                raise IOError('Error while opening data file')

                        #
                        def read_binary_multiplexed(rb_start, rb_length):
                            # will automatically cast data-types (int->float) later if needed
                            chan_start = rb_start * num_channels_in_file
                            if use_memmap:
                                return amem[chan_start:chan_start + rb_length * num_channels_in_file].reshape(num_channels_in_file, rb_length, order='F')
                            else:
                                fid_data.seek(chan_start * sample_size, os.SEEK_SET)
                                return np.fromfile(fid_data, dtype=sample_type, count=rb_length * num_channels_in_file).reshape(num_channels_in_file, -1, order='F')

                        # loop over the ranges
                        if single_range:
                            data[:, :] = read_binary_multiplexed(ranges_start, ranges_length)
                        else:
                            for range_index in range(len(ranges_start)):
                                data[:, :, range_index] = read_binary_multiplexed(ranges_start[range_index], ranges_length[range_index])

                        # apply the resolution per channel
                        for channel_count, channel_index in enumerate(channel_indices):
                            if not channel_gains[channel_count] == 1:
                                if single_range:
                                    data[channel_count, :] *= channel_gains[channel_count]
                                else:
                                    data[channel_count, :, :] *= channel_gains[channel_count]

                    else:
                        # not all channels or not in the same order as file
                        #print('multiplexed - non-chunked - memmap = ' + str(use_memmap) + ' - not allchan or not sameorder')

                        if use_memmap:

                            # open memory map and define the read function
                            amem = np.memmap(data_file, dtype=sample_type, mode='r', offset=0, shape=(num_channels_in_file, hdr['number_of_samples']), order='F')

                            def read_binary_multiplexed(rb_chan_index, rb_start, rb_length):
                                if channel_gains[channel_count] == 1:
                                    return amem[channel_index, rb_start:rb_start + rb_length]  # will automatically cast data-types (int->float) later if needed
                                else:
                                    return amem[channel_index, rb_start:rb_start + rb_length].astype(np.float64) * channel_gains[channel_count]

                            # loop over the channels/ranges
                            for channel_count, channel_index in enumerate(channel_indices):
                                if single_range:
                                    data[channel_count, :] = read_binary_multiplexed(channel_index, ranges_start, ranges_length)
                                else:
                                    for range_index in range(len(ranges_start)):
                                        data[channel_count, :, range_index] = read_binary_multiplexed(channel_index, ranges_start[range_index], ranges_length[range_index])


                        else:
                            # no other choice but to loop over samples and over channels (since channels are not always consecutive)

                            logging.warning('Should not be used, rather read in chunks')

                            try:
                                fid_data = open(data_file, "rb")
                            except IOError:
                                logging.error('Error while opening data file \'' + data_file + '\'')
                                raise IOError('Error while opening data file')

                            #
                            def read_binary_multiplexed(rb_range_index, rb_start, rb_length):
                                data_sample_pos = 0
                                if rb_range_index is None:

                                    for sample_index in range(rb_start, rb_start + rb_length):
                                        for channel_count, channel_index in enumerate(channel_indices):
                                            fid_data.seek((sample_index * num_channels_in_file + channel_index) * sample_size, os.SEEK_SET)
                                            if channel_gains[channel_count] == 1:
                                                data[channel_count, data_sample_pos] = np.fromfile(fid_data, dtype=sample_type, count=1)
                                            else:
                                                data[channel_count, data_sample_pos] = np.fromfile(fid_data, dtype=sample_type, count=1).astype(np.float64) * channel_gains[channel_count]
                                        data_sample_pos += 1

                                else:

                                    for sample_index in range(rb_start, rb_start + rb_length):
                                        for channel_count, channel_index in enumerate(channel_indices):
                                            fid_data.seek((sample_index * num_channels_in_file + channel_index) * sample_size, os.SEEK_SET)
                                            if channel_gains[channel_count] == 1:
                                                data[channel_count, data_sample_pos, range_index] = np.fromfile(fid_data, dtype=sample_type, count=1)
                                            else:
                                                data[channel_count, data_sample_pos, range_index] = np.fromfile(fid_data, dtype=sample_type, count=1).astype(np.float64) * channel_gains[channel_count]
                                        data_sample_pos += 1


                            # loop over the ranges
                            if single_range:
                                read_binary_multiplexed(None, ranges_start, ranges_length)
                            else:
                                for range_index in range(len(ranges_start)):
                                    read_binary_multiplexed(range_index, ranges_start[range_index], ranges_length[range_index])


            elif hdr['data_orientation'] == 'VECTORIZED':

                # Conclusions
                # - when reading samples or a single channel, only part is cached
                # - when reading all, the entire file is cached
                #
                # - after caching:
                #   - on memmap chunked read is faster, on fromfile non-chunked read is faster
                #   - in general memmap is faster than fromfile

                if use_memmap:
                    amem = np.memmap(data_file, dtype=sample_type, mode='r', offset=0, order='C')
                else:
                    try:
                        fid_data = open(data_file, "rb")
                    except IOError:
                        logging.error('Error while opening data file \'' + data_file + '\'')
                        raise IOError('Error while opening data file')

                if chunked_read:
                    # chunked reading
                    #print('vectorized - chunked - memmap = ' + str(use_memmap))

                    # determine the size (in samples) chucks assuming ~10mb per read
                    # rounded down to the number of channels (also here because than we ensure that we assign in chunks as a multiple of the channel dimension)
                    chunk_size = ((int(10e6) // sample_size))
                    assert(chunk_size != 0)

                    # read function
                    def read_binary_chunked(rb_chunk_size, rb_range_index, rb_start, rb_length):

                        # Note: reading each channel in chunks (since there is no guarantee that the requested channels are consecutive)
                        #       the chunk can be shortened depending on the number of samples in the channel or the requested range

                        # channels loop
                        for channel_count, channel_index in enumerate(channel_indices):

                            # (determine where to) start reading from
                            if use_memmap:
                                channel_sample_start = channel_index * hdr['number_of_samples'] + rb_start
                            else:
                                fid_data.seek((channel_index * hdr['number_of_samples'] + rb_start) * sample_size, os.SEEK_SET)

                            # chunk loop
                            data_sample_pos = 0
                            while data_sample_pos < rb_length:

                                # determine length (in samples) of the current chunk
                                current_chunk_length = min((rb_length - data_sample_pos), rb_chunk_size)

                                # read the chunk
                                if use_memmap:
                                    chunk_start = channel_sample_start + data_sample_pos

                                # transfer chunk data
                                # Note 1. will automatically cast data-types (int->float) later if needed
                                # Note 2. deliberate more ifs, because function calls are expensive in python (performance wise) and these are iterated through a lot
                                if channel_gains[channel_count] == 1:
                                    if rb_range_index is None:
                                        if use_memmap:
                                            data[channel_count, data_sample_pos:data_sample_pos + current_chunk_length] = amem[chunk_start:chunk_start + current_chunk_length]
                                        else:
                                            data[channel_count, data_sample_pos:data_sample_pos + current_chunk_length] = np.fromfile(fid_data, dtype=sample_type, count=current_chunk_length)
                                    else:
                                        if use_memmap:
                                            data[channel_count, data_sample_pos:data_sample_pos + current_chunk_length, rb_range_index] = amem[chunk_start:chunk_start + current_chunk_length]
                                        else:
                                            data[channel_count, data_sample_pos:data_sample_pos + current_chunk_length, rb_range_index] = np.fromfile(fid_data, dtype=sample_type, count=current_chunk_length)
                                else:
                                    if rb_range_index is None:
                                        if use_memmap:
                                            data[channel_count, data_sample_pos:data_sample_pos + current_chunk_length] = amem[chunk_start:chunk_start + current_chunk_length].astype(np.float64) * channel_gains[channel_count]
                                        else:
                                            data[channel_count, data_sample_pos:data_sample_pos + current_chunk_length] = np.fromfile(fid_data, dtype=sample_type, count=current_chunk_length).astype(np.float64) * channel_gains[channel_count]
                                    else:
                                        if use_memmap:
                                            data[channel_count, data_sample_pos:data_sample_pos + current_chunk_length, rb_range_index] = amem[chunk_start:chunk_start + current_chunk_length].astype(np.float64) * channel_gains[channel_count]
                                        else:
                                            data[channel_count, data_sample_pos:data_sample_pos + current_chunk_length, rb_range_index] = np.fromfile(fid_data, dtype=sample_type, count=current_chunk_length).astype(np.float64) * channel_gains[channel_count]

                                # move to the next chunk
                                data_sample_pos += current_chunk_length

                    # read range(s)
                    if single_range:
                        read_binary_chunked(chunk_size, None, ranges_start, ranges_length)
                    else:
                        for range_index in range(len(ranges_start)):
                            read_binary_chunked(chunk_size, range_index, ranges_start[range_index], ranges_length[range_index])

                else:
                    # non-chunked reading
                    #print('vectorized - non-chunked - memmap = ' + str(use_memmap))

                    # read function
                    # TODO: short function, move into loop below? Performance test to decide
                    if use_memmap:
                        def read_binary_vectorized(rb_chan_count, rb_start, rb_length):
                            if channel_gains[rb_chan_count] == 1:
                                return amem[rb_start:rb_start + rb_length]
                            else:
                                return amem[rb_start:rb_start + rb_length].astype(np.float64) * channel_gains[channel_count]
                    else:
                        def read_binary_vectorized(rb_chan_count, rb_start, rb_length):
                            fid_data.seek(rb_start * sample_size, os.SEEK_SET)
                            if channel_gains[rb_chan_count] == 1:
                                return np.fromfile(fid_data, dtype=sample_type, count=rb_length)
                            else:
                                return np.fromfile(fid_data, dtype=sample_type, count=rb_length).astype(np.float64) * channel_gains[channel_count]

                    # loop over the channels/ranges
                    for channel_count, channel_index in enumerate(channel_indices):
                        channel_sample_start = channel_index * hdr['number_of_samples']

                        # will automatically cast data-types (int->float) if needed
                        if single_range:
                            data[channel_count, :] = read_binary_vectorized(channel_count, channel_sample_start + ranges_start, ranges_length)
                        else:
                            for range_index in range(len(ranges_start)):
                                data[channel_count, :, range_index] = read_binary_vectorized(channel_count, channel_sample_start + ranges_start[range_index], ranges_length[range_index])

            # close file if needed
            if not use_memmap and fid_data is not None:
                fid_data.close()


        elif hdr['data_format'] == 'ASCII' and hdr['data_orientation'] == 'MULTIPLEXED':
            logging.error('Reading ASCII format with MULTIPLEXED orientation is not supported, test data is required')
            raise IOError('Reading ASCII format with MULTIPLEXED orientation is not supported, test data is required')

        elif hdr['data_format'] == 'ASCII' and hdr['data_orientation'] == 'VECTORIZED':
            logging.error('Reading ASCII format with VECTORIZED orientation is not supported, test data is required')
            raise IOError('Reading ASCII format with VECTORIZED orientation is not supported, test data is required')

        else:
            logging.error('Unknown data format')
            raise IOError('Unknown data format')

        # return the header information and the data
        return hdr, data
