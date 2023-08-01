"""
Wrapper around PyMef to read Mef3 files


=====================================================
Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

PyMef by Jan Cimbalnick, Daniel Crepeau et al.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import logging
from .IeegDataReader import IeegDataReader
from pymef.mef_file.pymef3_file import read_mef_session_metadata, read_mef_ts_data, clean_mef_session_metadata


class Mef3Reader(IeegDataReader):

    mef_session = None
    mef_data = None

    def __init__(self, data_path, preload_data=False, password=None):
        super().__init__(data_path, preload_data)
        self.data_format = 'mef3'

        # check path format
        if not data_path.endswith('/'):
            data_path += '/'
        if '.mefd' != data_path[-6:-1]:
            logging.error('MEF3 session path must end with .mefd suffix')
            raise RuntimeError('MEF3 session path must end with .mefd suffix')

        # check directory existence
        if not os.path.exists(data_path):
            logging.error('MEF3 directory \'' + data_path + '\' could not be found')
            raise RuntimeError('MEF3 directory not found')

        # read the session metadata
        try:
            self.mef_session = read_mef_session_metadata(data_path, password,
                                                         map_indices_flag=False,
                                                         copy_metadata_to_dict=False)
        except RuntimeError:
            logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
            raise RuntimeError('PyMef could not read data')

        # TODO: check if sampling_rate and num_samples is equal for each channel

        # retrieve the sample-rate, total number of samples and channel names
        self.sampling_rate = self.mef_session['time_series_metadata']['section_2']['sampling_frequency'].item(0)
        self.num_samples = self.mef_session['time_series_metadata']['section_2']['number_of_samples'].item(0)
        self.channel_names = []
        for ts_channel_name, ts_channel_metadata in self.mef_session['time_series_channels'].items():
            self.channel_names.append(ts_channel_name)

        # (optionally) preload data
        if self.preload_data:

            # initialize empty array
            self.mef_data = []

            # loop over the channels
            for ts_channel_name, ts_channel_metadata in self.mef_session['time_series_channels'].items():

                # load the channel data
                try:
                    channel_data = read_mef_ts_data(self.mef_session['time_series_channels'][ts_channel_name]['channel_specific_metadata'],
                                                    None, None)
                except Exception:
                    logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
                    raise RuntimeError('Could not read data')

                # return and apply a conversion factor if needed
                channel_conversion_factor = ts_channel_metadata['section_2']['units_conversion_factor'].item(0)
                if channel_conversion_factor != 0 and channel_conversion_factor != 1:
                    channel_data *= channel_conversion_factor

                # TODO: check MEF3 format, if after units_conversion_factor the units should be in uV, or after conversion
                #       just match whatever is in the Units Description

                # add channel
                self.mef_data.append(channel_data)


    def close(self):
        if self.mef_session is not None:
            clean_mef_session_metadata(self.mef_session['session_specific_metadata'])
            del self.mef_session
        if self.mef_data is not None:
            del self.mef_data



    #
    #
    #

    @staticmethod
    def __retrieve_channel_metadata(mef_session, channel_name):
        """
        Find and retrieve the MEF3 channel metadata and channel index by channel name
        """

        channel_metadata = None
        channel_counter = 0
        for ts_channel_name, ts_channel_metadata in mef_session['time_series_channels'].items():
            if ts_channel_name == channel_name:
                channel_metadata = ts_channel_metadata
                break
            channel_counter += 1

        if channel_metadata is None:
            logging.error('Could not find metadata for channel ' + channel_name + ', assuming there is no such channel in the dataset')
            return None, None

        return channel_metadata, channel_counter


    def retrieve_channel_data(self, channel_name, ensure_own_data=True):
        """
        Retrieve the MEF3 channel data by channel name

        Args:
            channel_name:                   The name of the channel for which to retrieve the data
            ensure_own_data (bool):         Should ensure the return a numpy array has it's own data (is not a view)

        Returns:
            Numpy array with data

        Raises:
            LookupError:                    Raised when the channel name cannot be found
            RuntimeError:                   Raised when unable to retrieve channel data
        """

        # find the channel metadata by channel name
        channel_metadata, channel_index = Mef3Reader.__retrieve_channel_metadata(self.mef_session, channel_name)
        if channel_metadata is None:
            raise LookupError('Could not find channel')

        # determine whether the data is preloaded
        if self.mef_data is None:
            # data is not preloaded

            # load the channel data
            try:
                channel_data = read_mef_ts_data(self.mef_session['time_series_channels'][channel_name]['channel_specific_metadata'],
                                                    None, None)
            except Exception:
                logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
                raise RuntimeError('Could not read data')

            # return and apply a conversion factor if needed
            channel_conversion_factor = channel_metadata['section_2']['units_conversion_factor'].item(0)

            if channel_conversion_factor != 0 and channel_conversion_factor != 1:
                channel_data *= channel_conversion_factor
            return channel_data

        else:
            # data is preloaded

            # return the channel data
            if ensure_own_data:
                return self.mef_data[channel_index].copy()
            else:
                return self.mef_data[channel_index]


    def retrieve_sample_range_data(self, sample_start, sample_end, channels=None, ensure_own_data=True):
        """
        Retrieve a specific range of MEF3 data for the requested channels

        Args:
            sample_start (int):             The start-point in time (in samples) to start reading from (0-based)
            sample_end (int):               The sample to end the reading at (0-based)
            channels (str, list or tuple):  The channel(s) for which to retrieve the data.
                                            If empty, all channels will be retrieved
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

        # create a list with the numpy arrays
        sample_data = [None] * len(channels)

        if self.mef_data is None:
            # data is not preloaded

            # loop over the channels
            for channel_counter in range(len(channels)):

                # load the trial data
                try:
                    sample_data[channel_counter] = read_mef_ts_data(self.mef_session['time_series_channels'][channels[channel_counter]]['channel_specific_metadata'],
                                                                    sample_start, sample_end)
                    if sample_data[channel_counter] is None or (len(sample_data[channel_counter]) > 0 and sample_data[channel_counter][0] is None):
                        raise RuntimeError('Could not read data')

                    # find the channel metadata by channel name
                    channel_metadata, _ = Mef3Reader.__retrieve_channel_metadata(self.mef_session, channels[channel_counter])
                    if channel_metadata is None:
                        raise LookupError('Could not find channel')

                    # apply a conversion factor if needed
                    channel_conversion_factor = channel_metadata['section_2']['units_conversion_factor'].item(0)
                    if channel_conversion_factor != 0 and channel_conversion_factor != 1:
                        sample_data[channel_counter] *= channel_conversion_factor

                except Exception:
                    logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
                    raise RuntimeError('Could not read data')

        else:
            # data is preloaded

            # loop over the channels to retrieve
            for counter in range(len(channels)):

                # retrieve the index by channel name
                try:
                    channel_index = self.channel_names.index(channels[counter])
                except ValueError:
                    raise LookupError('Could not find channel')

                # pick the slice
                if ensure_own_data:
                    sample_data[counter] = self.mef_data[channel_index][sample_start:sample_end].copy()
                else:
                    sample_data[counter] = self.mef_data[channel_index][sample_start:sample_end]

        #
        return sample_data
