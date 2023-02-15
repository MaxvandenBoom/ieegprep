"""
Wrapper around PyMef to read Mef3 files


=====================================================
Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

PyMef by Jan Cimbalnick, Daniel Crepeau et al.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
from .IeegDataReader import IeegDataReader
from pymef.mef_session import MefSession


class Mef3Reader(IeegDataReader):

    mef_session = None
    mef_data = None

    def __init__(self, data_path, data_preload=False):
        super().__init__(data_path, data_preload)
        self.data_format = 'mef3'

        # read the session metadata
        try:
            self.mef_session = MefSession(self.data_path, '', read_metadata=True)
        except RuntimeError:
            logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
            raise RuntimeError('PyMef could not read data')

        # TODO: check if sampling_rate and num_samples is equal for each channel

        # retrieve the sample-rate, total number of samples and channel names
        self.sampling_rate = self.mef_session.session_md['time_series_metadata']['section_2']['sampling_frequency'].item(0)
        self.num_samples = self.mef_session.session_md['time_series_metadata']['section_2']['number_of_samples'].item(0)
        self.channel_names = []
        for ts_channel_name, ts_channel_metadata in self.mef_session.session_md["time_series_channels"].items():
            self.channel_names.append(ts_channel_name)

        # (optionally) preload data
        if self.data_preload:

            # initialize empty array
            self.mef_data = []

            # loop over the channels
            for ts_channel_name, ts_channel_metadata in self.mef_session.session_md["time_series_channels"].items():

                # load the channel data
                try:
                    channel_data = self.mef_session.read_ts_channels_sample([ts_channel_name], (None, None))
                except Exception:
                    logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
                    raise RuntimeError('Could not read data')

                # return and apply a conversion factor if needed
                channel_conversion_factor = ts_channel_metadata['section_2']['units_conversion_factor'].item(0)
                if channel_conversion_factor != 0 and channel_conversion_factor != 1:
                    channel_data[0] *= channel_conversion_factor

                # TODO: check MEF3 format, if after units_conversion_factor the units should be in uV, or after conversion
                #       just match whatever is in the Units Description

                # add channel
                self.mef_data.append(channel_data[0])


    def close(self):
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
        for ts_channel_name, ts_channel_metadata in mef_session.session_md["time_series_channels"].items():
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
                channel_data = self.mef_session.read_ts_channels_sample([channel_name], (None, None))
            except Exception:
                logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
                raise RuntimeError('Could not read data')

            # return and apply a conversion factor if needed
            channel_conversion_factor = channel_metadata['section_2']['units_conversion_factor'].item(0)

            if channel_conversion_factor != 0 and channel_conversion_factor != 1:
                channel_data[0] *= channel_conversion_factor
            return channel_data[0]

        else:
            # data is preloaded

            # return the channel data
            if ensure_own_data:
                return self.mef_data[channel_index].copy()
            else:
                return self.mef_data[channel_index]


    def retrieve_sample_range_data(self, channels, sample_start, sample_end, ensure_own_data=True):
        """
        Retrieve a specific range of MEF3 data for the requested channels

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

        if self.mef_data is None:
            # data is not preloaded

            # load the trial data
            try:
                sample_data = self.mef_session.read_ts_channels_sample(channels, [sample_start, sample_end])
                if sample_data is None or (len(sample_data) > 0 and sample_data[0] is None):
                    raise RuntimeError('Could not read data')

                # loop over the channels to retrieve
                for channel_counter in range(len(channels)):

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

            # create a list with the numpy arrays
            sample_data = [None] * len(channels)

            # loop over the channels to retrieve
            for counter in range(len(channels)):

                # retrieve the index of the channel
                try:
                    channel_index = IeegDataReader.__retrieve_channel_metadata_mef(self.mef_session, self.channel_names.index(channels[counter]))
                except ValueError:
                    raise LookupError('Could not find channel')

                # pick the slice
                if ensure_own_data:
                    sample_data[counter] = self.mef_data[channel_index][sample_start:sample_end].copy()
                else:
                    sample_data[counter] = self.mef_data[channel_index][sample_start:sample_end]

        #
        return sample_data
