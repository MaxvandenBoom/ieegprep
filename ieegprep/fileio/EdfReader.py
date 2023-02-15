"""
Functions to read European Data Format (EDF) files


=====================================================
Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

Adapted from Fieldtrip (by Robert Robert Oostenveld) while replicating some additional header logic from the MNE
package (Teon Brooks, Martin Billinger, Nicolas Barascud, Stefan Appelhoff, Joan Massich, Clemens Brunner, Jeroen Van Der Donckt)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import os
import math
import logging
import numpy as np
from .IeegDataReader import IeegDataReader


class EdfReader(IeegDataReader):

    edf_hdr = None                   # EDF header
    edf_data = None                  # EDF data (only used on preload)

    def __init__(self, data_path, data_preload=False):
        super().__init__(data_path, data_preload)
        self.data_format = 'edf'

        # load header
        try:
            self.edf_hdr = self.edf_read_header(self.data_path)
        except (FileNotFoundError, IOError):
            logging.error('Could not read EDF header')
            raise RuntimeError('Could not read EDF header')

        # retrieve the sample-rate, total number of samples and channel names
        self.sampling_rate = self.edf_hdr['sampling_frequency']
        self.num_samples = self.edf_hdr['number_of_samples']
        self.channel_names, _ = EdfReader.edf_retrieve_all_channel_names(self.edf_hdr)

        # (optionally) preload data
        if self.data_preload:
            try:
                _, self.edf_data = self.edf_read_data(filepath=self.data_path, hdr=self.edf_hdr, unit='uV')
            except (FileNotFoundError, IOError, TypeError, RuntimeError):
                logging.error('Could not read EDF data')
                raise RuntimeError('Could not read EDF data')


    def close(self):
        del self.edf_hdr
        if self.edf_data is not None:
            del self.edf_data

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
            channel_index = self.edf_hdr['channel_names'].index(channel_name)
        except ValueError:
            raise LookupError('Could not find channel')

        # determine whether the data is preloaded
        if self.edf_data is None:
            # not preloaded

            # load the channel data
            try:
                _, channel_data = self.edf_read_data(filepath=self.data_path, hdr=self.edf_hdr,
                                                     channels=(channel_name,), unit='uV')
            except Exception:
                logging.error('Could not read EDF data')
                raise RuntimeError('Could not read EDF data')

            return channel_data.squeeze()

        else:
            # preloaded data

            # return the channel data
            if ensure_own_data:
                return self.edf_data[channel_index, :].copy()
            else:
                return self.edf_data[channel_index, :]


    def retrieve_sample_range_data(self, channels, sample_start, sample_end, ensure_own_data=True):
        """
        Retrieve a specific range of EDF data for the requested channels

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

        if self.edf_data is None:
            # data is not preloaded

            # load the data
            try:
                _, np_sample_data = self.edf_read_data(filepath=self.data_path, hdr=self.edf_hdr,
                                                       channels=channels,
                                                       start_sample=sample_start, end_sample=sample_end)
            except Exception:
                logging.error('Could not read EDF data')
                raise RuntimeError('Could not read EDF data')

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
                    channel_index = self.edf_hdr['channel_names'].index(channels[counter])
                except ValueError:
                    raise LookupError('Could not find channel')

                # pick the slice
                if ensure_own_data:
                    sample_data[counter] = self.edf_data[channel_index, sample_start:sample_end].copy()
                else:
                    sample_data[counter] = self.edf_data[channel_index, sample_start:sample_end]

        return sample_data



    #
    # EDF header and data reading functions
    #


    @staticmethod
    def edf_read_header(filepath):
        """
        Read the header of a EDF (.edf) file

        Args:
            filepath (str):                 The path to a EDF file

        Returns:
            A dictionary with header information

        Raises:
            FileNotFoundError:              Raised when unable to find the file
            IOError:                        Raised on any error that occurs while reading or parsing the header

        Examples:
            hdr = edf_read_header('~/dataset.edf')
        """

        #
        # Read/check the file
        #

        # check file existence
        if not os.path.exists(filepath):
            logging.error('EDF file \'' + filepath + '\' could not be found')
            raise FileNotFoundError('No such file or directory: \'' + filepath + '\'')

        # check extension
        if os.path.splitext(filepath)[1].lower() not in ('.edf',):
            logging.error('The file \'' + filepath + '\' has an invalid extension (should be .edf)')
            raise IOError('File \'' + filepath + '\' has an invalid extension')


        try:
            with open(filepath, 'rb') as file:

                hdr = dict()

                #
                # general information
                #

                # read the first part of the header
                header_1 = file.read(256).decode('ascii')

                #
                hdr['version'] = header_1[0:8].rstrip()
                hdr['patient_id'] = header_1[8:8 + 80].rstrip()
                hdr['recording_id'] = header_1[88:88 + 80].rstrip()

                #
                hdr['t0_time'] = list((int(header_1[176:178]), int(header_1[179:181]), int(header_1[182:184])))
                t0_date = list((int(header_1[174:176]), int(header_1[171:173]), int(header_1[168:170])))
                t0_date[0] = t0_date[0] + 2000 if t0_date[0] < 85 else t0_date[0] + 1900
                hdr['t0_date'] = t0_date

                #
                hdr['header_length'] = int(header_1[184:191])
                hdr['num_records'] = int(header_1[236:243])             # number of data records (-1 if unknown)
                hdr['record_duration'] = float(header_1[244:251])       # duration of a data record, in seconds
                hdr['number_of_channels']  = int(header_1[252:255])     # number of channels/signals in data record


                #
                # channels info
                #

                hdr['channel_names'] = []
                hdr['channel_other_info'] = []
                for count in range(0, hdr['number_of_channels']):

                    # add the channel name, and a dict for each channel (to hold all the other information)
                    dict_channel_other_info = dict()
                    channel_info = file.read(16).decode('ascii').rstrip().split(' ')
                    if len(channel_info) > 1:
                        hdr['channel_names'].append(channel_info[1])
                        dict_channel_other_info['type'] = channel_info[0]
                    else:
                        hdr['channel_names'].append(channel_info[0])
                        dict_channel_other_info['type'] = ''
                    hdr['channel_other_info'].append(dict_channel_other_info)

                # transducer
                for count in range(0, hdr['number_of_channels']):
                    hdr['channel_other_info'][count]['transducer'] = file.read(80).decode('ascii').rstrip()

                # physical dim
                for count in range(0, hdr['number_of_channels']):
                    phys_dim = file.read(8).decode('ascii').rstrip()
                    hdr['channel_other_info'][count]['phys_dim'] = phys_dim

                    # determine the unit for specific phys_dim codes (otherwise ensure multiplication by 1)
                    # Note: setting this unit is used to later allow us to convert to the requested output (uV, mV or V)
                    if phys_dim.lower() in ('\u03BCv', '\u00B5v', '\x83\xCAv', 'uv'):
                        hdr['channel_other_info'][count]['units_to_V_gain'] = 1e-6
                    elif phys_dim.lower() == 'mv':
                        hdr['channel_other_info'][count]['units_to_V_gain'] = 1e-3
                    elif phys_dim.lower() == 'v':
                        hdr['channel_other_info'][count]['units_to_V_gain'] = 1
                    else:
                        #logging.warning('The phys_dim on channel ' + hdr['channel_names'][count] + ' is unknown, assume multiplication by 1')
                        hdr['channel_other_info'][count]['units_to_V_gain'] = 1

                # physical min, max, range
                for count in range(0, hdr['number_of_channels']):
                    hdr['channel_other_info'][count]['phys_min'] = np.float64(file.read(8).decode('ascii').rstrip())

                bad_phys_channel_names = []
                for count in range(0, hdr['number_of_channels']):
                    hdr['channel_other_info'][count]['phys_max'] = np.float64(file.read(8).decode('ascii').rstrip())
                    if hdr['channel_other_info'][count]['phys_min'] >= hdr['channel_other_info'][count]['phys_max']:
                        logging.warning('The physical minimum on channel ' + hdr['channel_names'][count] + ' is larger than maximum. Recheck the scaling and polarity')

                    # calculate and check the range
                    hdr['channel_other_info'][count]['phys_range'] = hdr['channel_other_info'][count]['phys_max'] - hdr['channel_other_info'][count]['phys_min']
                    if ~np.isfinite(hdr['channel_other_info'][count]['phys_range']) or hdr['channel_other_info'][count]['phys_range'] == 0:
                        bad_phys_channel_names.append(hdr['channel_names'][count])
                        hdr['channel_other_info'][count]['phys_range'] = 1
                if bad_phys_channel_names:
                    logging.warning('No valid phys scaling factor in channel: ' + ', '.join(bad_phys_channel_names) + ', scaling set to 1 for now')


                # digital min, max, range
                for count in range(0, hdr['number_of_channels']):
                    hdr['channel_other_info'][count]['dig_min'] = np.float64(file.read(8).decode('ascii').rstrip())

                bad_dig_channel_names = []
                for count in range(0, hdr['number_of_channels']):
                    hdr['channel_other_info'][count]['dig_max'] = np.float64(file.read(8).decode('ascii').rstrip())
                    if hdr['channel_other_info'][count]['dig_min'] >= hdr['channel_other_info'][count]['dig_max']:
                        logging.warning('The digital minimum on channel ' + hdr['channel_names'][count] + ' is larger than maximum. Recheck the scaling and polarity')

                    # calculate and check the range
                    hdr['channel_other_info'][count]['dig_range'] = hdr['channel_other_info'][count]['dig_max'] - hdr['channel_other_info'][count]['dig_min']
                    if ~np.isfinite(hdr['channel_other_info'][count]['dig_range']) or hdr['channel_other_info'][count]['dig_range'] == 0:
                        bad_dig_channel_names.append(hdr['channel_names'][count])
                        hdr['channel_other_info'][count]['dig_range'] = 1
                if bad_dig_channel_names:
                    logging.warning('No valid digital scaling factor in channel: ' + ', '.join(bad_dig_channel_names) + ', scaling set to 1 for now')


                # determine scaling and offset
                for count in range(0, hdr['number_of_channels']):
                    hdr['channel_other_info'][count]['cal'] = hdr['channel_other_info'][count]['phys_range'] / hdr['channel_other_info'][count]['dig_range']
                    hdr['channel_other_info'][count]['offset'] = hdr['channel_other_info'][count]['phys_min'] - hdr['channel_other_info'][count]['dig_min'] * hdr['channel_other_info'][count]['cal']

                # prefilters (HighPass, LowPass or Notch)
                for count in range(0, hdr['number_of_channels']):
                    hdr['channel_other_info'][count]['pre_filter'] = file.read(80).decode('ascii').rstrip()
                    # TODO: could split further ("HP:0.1Hz LP:75Hz N:50Hz")

                channel_SPR_equal_tofirst = []
                recordblock_length_in_samples = 0      # the length of a record block in samples (= total of all number of samples-per-record over all channels)
                for count in range(0, hdr['number_of_channels']):
                    hdr['channel_other_info'][count]['samples_per_record'] = int(file.read(8).decode('ascii').rstrip())

                    # count to the total of all number of samples-per-record over all channels
                    recordblock_length_in_samples += hdr['channel_other_info'][count]['samples_per_record']

                    # calculate the sampling rate per channel (depends on the duration of a record)
                    hdr['channel_other_info'][count]['sample_frequency'] = hdr['channel_other_info'][count]['samples_per_record'] / hdr['record_duration']

                    # check if the channel's samples_per_record are equal to the first channel
                    channel_SPR_equal_tofirst.append(hdr['channel_other_info'][0]['samples_per_record'] == hdr['channel_other_info'][count]['samples_per_record'])

                # store the record-block length in the header
                hdr['recordblock_length_in_samples'] = recordblock_length_in_samples


                #
                # checks on header fields based on file size
                #

                # check if the header length matches
                file.read(32 * hdr['number_of_channels'])      # skip
                if file.tell() != hdr['header_length']:
                    logging.error('The position at the end of the header and length of header according to the header do not match')
                    raise IOError('End of the header does not correspond with expected length of the header')

                # calculate and check number of records in the data part of the file
                #   1. determine the total number of bytes in the file
                #   2. number of data bytes = (num_bytes - hdr['header_length'])
                #   3. total number of samples = (num_data_bytes // 2)         # (each sample is 2 bytes for a EDF file)
                #   4. number of records = total number of samples // recordblock length (sum of the samples-per-record over all channels)
                num_bytes = file.seek(0, os.SEEK_END)
                num_records = ((num_bytes - hdr['header_length']) // 2) // recordblock_length_in_samples
                if hdr['num_records'] != num_records:
                    logging.warning('The number of records based on the file size does not match the number of records according to the header. Inferring from file-size')

                    # update num records in header
                    hdr['num_records'] = num_records

                #
                if all(channel_SPR_equal_tofirst):
                    # EDF format (without last annotation channel)

                    hdr['sampling_frequency'] = hdr['channel_other_info'][0]['sample_frequency']
                    hdr['number_of_samples'] = hdr['num_records'] * hdr['channel_other_info'][0]['samples_per_record']


                elif channel_SPR_equal_tofirst[0:-1]:
                    # EDF+ format (last channel are annotations)

                    hdr['sampling_frequency'] = hdr['channel_other_info'][0]['sample_frequency']
                    hdr['number_of_samples'] = hdr['num_records'] * hdr['channel_other_info'][0]['samples_per_record']

                else:
                    logging.warning('Multiple channels have different sampling rates, unable determine a global sampling rate for the header')

            # success, return header dict
            return hdr

        except:
            logging.error('Error while reading the EDF file \'' + filepath + '\'')
            raise IOError('Error reading .edf file')


    @staticmethod
    def edf_retrieve_all_channel_names(hdr):
        """
        Retrieve all of the channel names, while leaving out the annotation channel

        Args:
            hdr (dict):                     The header dictionary as provided by 'edf_read_header'.

        Returns:
            channel_names (list):           A list with all the channel names
            channel_indices (list):         A list with all corresponding channel indices

        Raises:
            RuntimeError:                   Raised upon a processing error
        """

        # determine whether the samples_per_record per channel differ from the first channel
        channel_SPR_equal_tofirst = []
        for count in range(0, hdr['number_of_channels']):
            channel_SPR_equal_tofirst.append(hdr['channel_other_info'][0]['samples_per_record'] == hdr['channel_other_info'][count]['samples_per_record'])

        # try to include all channels
        if all(channel_SPR_equal_tofirst):
            # EDF format (without last annotation channel)

            # all channels
            channel_names = hdr['channel_names']
            channel_indices = list(range(0, len(channel_names)))

        elif channel_SPR_equal_tofirst[0:-1]:
            # EDF+ format (last channel are annotations)

            # all channels except for the last
            channel_names = hdr['channel_names'][0:-1]
            channel_indices = list(range(0, len(channel_names)))

        else:
            logging.error('Multiple channels have different sampling rates, please provide which channels to retrieve (with the same sampling rate')
            raise RuntimeError('Multiple channels have different sampling rates')

        return channel_names, channel_indices


    @staticmethod
    def edf_read_data(filepath, hdr=None, channels=None, start_sample=0, end_sample=-1, unit='uV', use_memmap=True, chunked_read=True):
        """
        Read data from a European Data Format (.edf) file

        Args:
            filepath (str):                 The path to the EDF file
            hdr (dict):                     Optionally pass the header dictionary as provided by 'edf_read_header'. If this
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
            IOError:                        Raised on any error that occurs while reading or parsing the file
            TypeError:                      Raised when input argument types are wrong
            RuntimeError:                   Raised upon a processing error

        Examples:
            hdr, data = edf_read_data('~/dataset.edf')
            hdr, data = edf_read_data('~/dataset.edf', units='uV')
            hdr, data = edf_read_data('~/dataset.edf', channels='CH01')
            hdr, data = edf_read_data('~/dataset.edf', channels=('CH01', 'CH07'))
            hdr, data = edf_read_data('~/dataset.edf', start_sample=1000, end_sample=2000)
            hdr, data = edf_read_data('~/dataset.edf', start_sample=(1000, 6000), end_sample=(2000, 7000))
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
            try:
                hdr = EdfReader.edf_read_header(filepath)
            except (FileNotFoundError, IOError):
                logging.error('Error while reading/parsing header from file \'' + filepath + '\'')
                raise IOError('Error while reading/parsing header')

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

        # retrieve channel
        if channels is None or len(channels) == 0:
            # no/empty channels argument

            try:
                _, channel_indices = EdfReader.edf_retrieve_all_channel_names(hdr)
            except RuntimeError:
                raise RuntimeError('Multiple channels have different sampling rates')

        else:
            # select specific channels

            # retrieve the channel indices
            channel_indices = []
            for channel in channels:
                try:
                    index = hdr['channel_names'].index(channel)
                    channel_indices.append(index)
                except ValueError:
                    logging.error('requested channel \'' + channel + '\' was not found')
                    raise IOError('Requested channel was not found')

            # check if all requested channels have the same sampling rate
            for index in channel_indices:
                if hdr['channel_other_info'][channel_indices[0]]['samples_per_record'] != hdr['channel_other_info'][index]['samples_per_record']:
                    logging.error('One or more requested channels differ in sampling rate, can only combine channels with the same sampling rate')
                    raise RuntimeError('Multiple channels have different sampling rates')

        # determine the offsets of each channel in a record
        channel_offsets = []
        record_sample_count = 0
        for count in range(0, hdr['number_of_channels']):
            channel_offsets.append(record_sample_count)
            record_sample_count += hdr['channel_other_info'][count]['samples_per_record']
        del record_sample_count


        #
        # check/prepare sample range input(s)
        #
        single_range, output_dimensions, ranges_start, ranges_length = IeegDataReader.check_sample_arguments(start_sample, end_sample,
                                                                                                             hdr['number_of_samples'],
                                                                                                             len(channel_indices))


        #
        # Read the data file
        #

        # check if the system is 32-bit and memmap is enabled
        # (mapping on 32-bit systems is limited to 2GB files, so disable just in case)
        if use_memmap and not (sys.maxsize > 2 ** 32):
            logging.warning('This is a 32-bit system. Data will be read with standard IO operations (rather than memmap)')
            use_memmap = False

        # helper functions to speed up performance, operation is not performed if it has no effect
        def data_add(in_data, gain):
            return in_data if gain == 0 else in_data + gain

        def data_multiply(in_data, factor):
            return in_data if factor == 1 else in_data * factor

        # determine the unit gains for the requested channels
        channel_gains = []
        for channel_count, channel_index in enumerate(channel_indices):

            unit_gain = hdr['channel_other_info'][channel_index]['units_to_V_gain']
            if unit.lower() == 'uv':
                unit_gain *= 1e+6
            elif unit.lower() == 'mv':
                unit_gain *= 1e+3

            channel_gains.append(unit_gain)
        del channel_count, channel_index

        # quick ref for SPR
        samples_per_record = hdr['channel_other_info'][channel_indices[0]]['samples_per_record']

        # determine the sample length
        range_length = output_dimensions[1]

        # int16 data could be cast to float32 (using less memory than a float64) but EDF will very often
        # have an offset, cal or unit gain value. So default to float64 to accommodate more precision
        data = np.empty(output_dimensions, dtype=np.float64)

        # determine whether to read chunked
        #if use_memmap:
        #    chunked_read = True
        #else:
            # Note: the consideration here depends on how the number of channels to be retrieved compares to the
            #       number of channels (in a record), see tipping point 4/149 and 10/149 channel benchmark.
            #       More testing is needed. However, often just a single channel is loaded, in this case non-chunked
            #       is faster; else wise chunked read.
        #    chunked_read = len(channel_indices) != 1

        #
        if chunked_read:
            # chunked reading
            #print('chunked - memmap = ' + str(use_memmap))

            if use_memmap:
                amem = np.memmap(filepath, dtype="<i2", mode='r', offset=hdr['header_length'], order='C')
            else:
                try:
                    fid_data = open(filepath, "rb")
                except IOError:
                    logging.error('Error while opening data file \'' + filepath + '\'')
                    raise IOError('Error while opening data file')

            # determine how many record there are in each chunk (samplesize is 2 bytes for EDF), assuming chunks of ~10mb
            # rounded down to the number of channels
            record_chunk_size = ((int(10e6) // 2) // hdr['recordblock_length_in_samples'])
            if record_chunk_size < 1:
                record_chunk_size = 1

            # define read function for reading binary in chunks
            def read_binary_chunked(rb_range_index, rb_start, rb_length):

                # determine
                begin_record = math.floor(rb_start / samples_per_record)

                # determine the record offset in samples
                record_offset_in_samples = begin_record * hdr['recordblock_length_in_samples']

                # edf = <dim0=records> x <dim1=channels> x <dim2=samples>
                data_sample_pos = 0
                while data_sample_pos < rb_length:
                    samples_left = rb_length - data_sample_pos

                    # determine the number of records need to be read for the current chunk
                    #
                    # Note:  In the situation where: (1) the first read occurs and the requested number of samples (rb_length)
                    #        results in requiring only one chunked read (less records needed than the maximum); and (2) the
                    #        requested start_sample is not at the start of a record (start % samples_per_record); then
                    #        the chunk might require an extra record because the offset shift.
                    # Note2: The chunks after the first read will always start at the beginning of the record (no offset) so
                    #        an extra record will never be needed
                    if data_sample_pos == 0:
                        current_chunk_num_records = min(math.ceil((rb_start % samples_per_record + samples_left) / samples_per_record), record_chunk_size)
                    else:
                        current_chunk_num_records = min(math.ceil(samples_left / samples_per_record), record_chunk_size)

                    # read the data and reshape on records
                    if use_memmap:
                        chunk_data = amem[record_offset_in_samples:record_offset_in_samples + current_chunk_num_records * hdr['recordblock_length_in_samples']].reshape(current_chunk_num_records, -1, order='C')
                    else:
                        fid_data.seek(hdr['header_length'] + record_offset_in_samples * 2, os.SEEK_SET)
                        chunk_data = np.fromfile(fid_data, dtype="<i2", count=current_chunk_num_records * hdr['recordblock_length_in_samples']).reshape(current_chunk_num_records, -1, order='C')

                    # determine where in the chunk where to start and where to end
                    if data_sample_pos == 0:
                        current_chunk_begin_sample = rb_start - begin_record * samples_per_record
                    else:
                        current_chunk_begin_sample = 0
                    current_chunk_end_sample = min(current_chunk_begin_sample + samples_left, samples_per_record * current_chunk_num_records)
                    num_samples = current_chunk_end_sample - current_chunk_begin_sample

                    # loop over the channels
                    for rb_channel_count, rb_channel_index in enumerate(channel_indices):
                        channel_data = chunk_data[:, channel_offsets[rb_channel_index]:channel_offsets[rb_channel_index] + samples_per_record].flatten()

                        if rb_range_index is None:
                            data[rb_channel_count, data_sample_pos:data_sample_pos + num_samples] = data_multiply(data_add(data_multiply(channel_data[current_chunk_begin_sample:current_chunk_end_sample], hdr['channel_other_info'][rb_channel_index]['cal']), hdr['channel_other_info'][rb_channel_index]['offset']), channel_gains[rb_channel_count])
                            #data[rb_channel_count, data_sample_pos:data_sample_pos + num_samples] = channel_data[record_begin_sample:record_end_sample]

                        else:
                            data[rb_channel_count, data_sample_pos:data_sample_pos + num_samples, rb_range_index] = data_multiply(data_add(data_multiply(channel_data[current_chunk_begin_sample:current_chunk_end_sample], hdr['channel_other_info'][rb_channel_index]['cal']), hdr['channel_other_info'][rb_channel_index]['offset']), channel_gains[rb_channel_count])
                            #data[rb_channel_count, data_sample_pos:data_sample_pos + num_samples, rb_range_index] = channel_data[record_begin_sample:record_end_sample]

                    # move to the next chunk
                    data_sample_pos += num_samples
                    record_offset_in_samples += current_chunk_num_records * hdr['recordblock_length_in_samples']

            # read range(s)
            if single_range:
                read_binary_chunked(None, ranges_start, ranges_length)
            else:
                for range_index in range(len(ranges_start)):
                    read_binary_chunked(range_index, ranges_start[range_index], ranges_length[range_index])

            # close file if needed
            if not use_memmap and fid_data is not None:
                fid_data.close()


        else:
            # non-chunked reading
            #print('non-chunked - memmap = ' + str(use_memmap))

            if use_memmap:

                # open memory map
                amem = np.memmap(filepath, dtype="<i2", mode='r', offset=hdr['header_length'], shape=(hdr['num_records'], hdr['recordblock_length_in_samples']), order='C')

                # read function
                def read_range(rb_range_index, rb_start, rb_end):

                    # determine which records would hold this range of samples
                    begin_record = math.floor(rb_start / samples_per_record)
                    end_record = math.floor(rb_end / samples_per_record)

                    sample_offset = rb_start - begin_record * samples_per_record

                    # loop over the channels
                    for rb_channel_count, rb_channel_index in enumerate(channel_indices):

                        # pick channel data and flatten
                        channel_data = np.array(amem[begin_record:end_record + 1, channel_offsets[rb_channel_index]:channel_offsets[rb_channel_index] + hdr['channel_other_info'][rb_channel_index]['samples_per_record']]).flatten()

                        # retrieve the data (with cal, offset and gain applied)
                        if rb_range_index is None:
                            data[rb_channel_count, :] = data_multiply(data_add(data_multiply(channel_data[sample_offset:sample_offset + range_length], hdr['channel_other_info'][rb_channel_index]['cal']), hdr['channel_other_info'][rb_channel_index]['offset']), channel_gains[rb_channel_count])
                            #data[rb_channel_count, :] = channel_data[sample_offset:sample_offset + range_length]
                        else:
                            data[rb_channel_count, :, rb_range_index] = data_multiply(data_add(data_multiply(channel_data[sample_offset:sample_offset + range_length], hdr['channel_other_info'][rb_channel_index]['cal']), hdr['channel_other_info'][rb_channel_index]['offset']), channel_gains[rb_channel_count])
                            #data[rb_channel_count, :, rb_range_index] = channel_data[sample_offset:sample_offset + range_length]


            else:
                # frombuffer (instead of memmap)

                # open file
                try:
                    fid_data = open(filepath, "rb")
                except IOError:
                    logging.error('Error while opening data file \'' + filepath + '\'')
                    raise IOError('Error while opening data file')

                # read function
                def read_range(rb_range_index, rb_start, rb_end):

                    # determine which records would hold this range of samples
                    begin_record = math.floor(rb_start / samples_per_record)
                    end_record = math.floor(rb_end / samples_per_record)

                    sample_offset = rb_start - begin_record * samples_per_record

                    # first loop over records, then channels (memmap only),
                    data_index = 0
                    for record_index in range(begin_record, end_record + 1):

                        #
                        if record_index == begin_record:
                            record_begin_sample = sample_offset
                        else:
                            record_begin_sample = 0

                        if record_index == end_record:
                            record_end_sample = range_length - data_index + record_begin_sample
                        else:
                            record_end_sample = samples_per_record

                        #
                        data_length = record_end_sample - record_begin_sample
                        record_offset = hdr['header_length'] + record_index * hdr['recordblock_length_in_samples'] * 2

                        for rb_channel_count, rb_channel_index in enumerate(channel_indices):

                            channel_begin_sample = record_begin_sample + channel_offsets[rb_channel_index]
                            channel_num_sample = record_end_sample - record_begin_sample

                            fid_data.seek(record_offset + channel_begin_sample * 2, os.SEEK_SET)
                            if rb_range_index is None:
                                data[rb_channel_count, data_index:data_index + data_length] = data_multiply(data_add(data_multiply(np.frombuffer(fid_data.read(channel_num_sample * 2), dtype='<i2'), hdr['channel_other_info'][rb_channel_index]['cal']), hdr['channel_other_info'][rb_channel_index]['offset']), channel_gains[rb_channel_count])
                                #data[rb_channel_count, data_index:data_index + data_length] = np.frombuffer(fid_data.read(channel_num_sample * 2), dtype='<i2')
                            else:
                                data[rb_channel_count, data_index:data_index + data_length, rb_range_index] = data_multiply(data_add(data_multiply(np.frombuffer(fid_data.read(channel_num_sample * 2), dtype='<i2'), hdr['channel_other_info'][rb_channel_index]['cal']), hdr['channel_other_info'][rb_channel_index]['offset']), channel_gains[rb_channel_count])
                                #data[rb_channel_count, data_index:data_index + data_length, rb_range_index] = fid_data.read(channel_num_sample * 2), dtype='<i2')

                        #
                        data_index += data_length


            # read the range(s)
            if single_range:
                read_range(None, ranges_start, ranges_start + ranges_length)
            else:
                for range_index in range(len(ranges_start)):
                    read_range(range_index, ranges_start[range_index], ranges_start[range_index] + ranges_length[range_index])

            # close file if needed
            if not use_memmap and fid_data is not None:
                fid_data.close()


                """
                # test code for first loop over channels, then records

                # determine which records would hold this range of samples
                begin_record = math.floor(read_start_sample / samples_per_record)
                end_record = math.floor(read_end_sample / samples_per_record)

                sample_offset = read_start_sample - begin_record * samples_per_record

                # test code for first loop over channels, then records (memmap & frombuffer), single range only (because writes directly to data)
                fid_data = open(filepath, "rb")  # slower (but still an option if memmap is not supported or slower)
                for rb_channel_count, rb_channel_index in enumerate(channel_indices):

                    data_index = 0
                    for record_index in range(begin_record, end_record + 1):

                        if record_index == begin_record:
                            record_begin_sample = sample_offset
                        else:
                            record_begin_sample = 0

                        if record_index == end_record:
                            record_end_sample = range_length - data_index + record_begin_sample
                        else:
                            record_end_sample = samples_per_record

                        data_length = record_end_sample - record_begin_sample

                        record_begin_sample += channel_offsets[rb_channel_index]
                        record_end_sample += channel_offsets[rb_channel_index]

                        fid_data.seek(hdr['header_length'] + (record_index * hdr['recordblock_length_in_samples'] + record_begin_sample) * 2, os.SEEK_SET)
                        data[rb_channel_count, data_index:data_index + data_length] = np.frombuffer(fid_data.read(data_length * 2), dtype='<i2')
                        #data[rb_channel_count, data_index:data_index + data_length] = amem[record_index, record_begin_sample:record_end_sample]
                        data_index += data_length

                """

        # return the header information and the data
        return hdr, data
