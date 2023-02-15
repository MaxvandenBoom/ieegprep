"""
Unit tests to validate the BrainVision reader

This class tests reading (chunked or non-chunked, standard IO and Memmap):
   - a single channel (all samples)                       --> test01_fileio_bv_validate_data_single_channel
   - the full set (all channels, all samples)             --> test02_fileio_bv_validate_data_fullset
   - 100 000 samples (all channels)                       --> test03_fileio_bv_validate_data_100k_samples
   - multiple (two) ranges of 1000 samples (all channels) --> test04_fileio_bv_validate_data_multi_range

Each of these tests uses the BrainVision reader to read data and validates the read data against:
    - a specific Matlab data file (.mat) that has been produced by the 'matlab/generate_bv_verification_sets.m' script,
      which in turn uses Fieldtrip to read the data
    - the same subset of data from the full dataset read using MNE


=====================================================
Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import unittest
from ieegprep.utils.console import ConsoleColors


class TestFileIOBVData(unittest.TestCase):
    """
    Validate the BrainVision reader output values against the Matlab (fieldtrip) and MNE reader output values
    """

    #
    # Configuration
    #

    data_orientation = 'multiplexed'
    #data_orientation = 'vectorized'

    matlab_data_path = 'D:\\BIDS_erdetect\\bv_'
    bv_data_path = 'D:\\BIDS_erdetect\\sub-BV\\ses-1\\ieeg\\sub-BV_ses-1_ieeg.vhdr'
    #matlab_data_path = os.path.expanduser('~/Documents/ERDetect_perf/bv_')
    #bv_data_path = os.path.expanduser('~/Documents/ERDetect_perf/sub-BV/ses-1/ieeg/sub-BV_ses-1_ieeg.vhdr')

    # set part of the filename based on the configuration
    orient_path = 'mplex_' if data_orientation.lower() == 'multiplexed' else 'vectd_'


    def test01_fileio_bv_validate_data_single_channel(self):
        # validate single channel

        #
        # matlab data
        #
        import scipy.io
        mat = scipy.io.loadmat(self.matlab_data_path + self.orient_path + 'allsamples_CH07.mat')

        # Note: When reading a single channel from multiplexed BV data, stdIO non-chunked reading will automatically
        #       switch to chunked reading to prevent slow per sample reading. As such, stdIO non-chunked reading
        #       is skipped here because it wouldn't really get tested
        skip_stdIO_non_chunked = self.data_orientation.lower() == 'multiplexed'
        self._test_reader_output_matrix(mat['dat'], 0, 'Matlab single channel', channels=('CH07',), skip_stdIO_non_chunked=skip_stdIO_non_chunked)
        del mat

        #
        # MNE data
        #
        import numpy as np
        from mne.io import read_raw_brainvision
        mne_raw = read_raw_brainvision(self.bv_data_path, eog=[], misc=[], preload=True, verbose=None)
        mne_raw._data *= 1000000    # MNE always returns in V, we dictate mV in our read, so convert MNE data to mV here
        try:
            channel_idx = mne_raw.info['ch_names'].index('CH07')
        except ValueError:
            self.assertEqual(0, 1, 'Could not find channel \'' + 'CH07' + '\' in the dataset')

        self._test_reader_output_matrix(np.array([mne_raw._data[channel_idx, :]]), np.finfo(np.float32).eps, 'MNE single channel', channels=('CH07',), skip_stdIO_non_chunked=skip_stdIO_non_chunked)
        mne_raw.close()
        del mne_raw._data
        del mne_raw

        ConsoleColors.print_green('Single channel test successful')


    def test02_fileio_bv_validate_data_fullset(self):
        # full dataset - all samples - all channels

        #
        # matlab data
        #
        import mat73
        mat = mat73.loadmat(self.matlab_data_path + self.orient_path + 'allsamples_allchannels.mat')
        self._test_reader_output_matrix(mat['dat'], 0, 'Matlab full dataset')
        del mat

        #
        # MNE data
        #
        import numpy as np
        from mne.io import read_raw_brainvision
        mne_raw = read_raw_brainvision(self.bv_data_path, eog=[], misc=[], preload=True, verbose=None)
        mne_raw._data *= 1000000    # MNE always returns in V, we dictate mV in our read, so convert MNE data to mV here

        self._test_reader_output_matrix(mne_raw._data, np.finfo(np.float32).eps, 'MNE full dataset')
        mne_raw.close()
        del mne_raw._data
        del mne_raw

        ConsoleColors.print_green('Full dataset test successful')
        

    def test03_fileio_bv_validate_data_100k_samples(self):
        # 100 000 samples - all channels

        #
        # matlab data
        #
        import scipy.io
        mat = scipy.io.loadmat(self.matlab_data_path + self.orient_path + '100ksamples.mat')
        self._test_reader_output_matrix(mat['dat'], 0, 'Matlab 100k samples', start_sample=1000, end_sample=101000)
        del mat

        #
        # MNE data
        #
        import numpy as np
        from mne.io import read_raw_brainvision
        mne_raw = read_raw_brainvision(self.bv_data_path, eog=[], misc=[], preload=True, verbose=None)
        mne_raw._data *= 1000000    # MNE always returns in V, we dictate mV in our read, so convert MNE data to mV here

        self._test_reader_output_matrix(mne_raw._data[:, 1000:101000], np.finfo(np.float32).eps, 'MNE 100k samples', start_sample=1000, end_sample=101000)
        mne_raw.close()
        del mne_raw._data
        del mne_raw

        ConsoleColors.print_green('100k samples test successful')
    

    """
    def test03b_fileio_bv_validate_data_100k_samples(self):
        # sliding ranges - single channel

        skip_stdIO_non_chunked = self.data_orientation.lower() == 'multiplexed'
        from ieegprep.fileio.BrainVisionReader import BrainVisionReader
        hdr = BrainVisionReader.bv_read_header(self.bv_data_path)

        #
        # MNE data
        #
        import numpy as np
        from mne.io import read_raw_brainvision
        mne_raw = read_raw_brainvision(self.bv_data_path, eog=[], misc=[], preload=True, verbose=None)
        mne_raw._data *= 1000000    # MNE always returns in V, we dictate mV in our read, so convert MNE data to mV here

        
        # test slide sample over a small number of blocks (1 and 2)
        for start_sample in range(0, 1000, 1):
            for end_sample in range(1000, 400, -1):
                if start_sample < end_sample:
                    print('Test - start sample: ' + str(start_sample) + ' - end sample:' + str(end_sample))
                    self._test_reader_output_matrix(np.array([mne_raw._data[1, start_sample:end_sample]]), np.finfo(np.float32).eps,
                                                    'MNE sliding ranges',
                                                    start_sample=start_sample, end_sample=end_sample, channels=('CH02',),
                                                    skip_stdIO_non_chunked=skip_stdIO_non_chunked, silent=True)

        # test slide over all sample possibilities where the start is between 0 and 10, and the end is between end-10 and end
        for start_sample in range(0, 10, 1):
            for end_sample in range(hdr['number_of_samples'], hdr['number_of_samples'] - 10, -1):
                print('Test - start sample: ' + str(start_sample) + ' - end sample:' + str(end_sample))
                self._test_reader_output_matrix(np.array([mne_raw._data[1, start_sample:end_sample]]), np.finfo(np.float32).eps,
                                                'MNE sliding ranges',
                                                start_sample=start_sample, end_sample=end_sample, channels=('CH02',),
                                                skip_stdIO_non_chunked=skip_stdIO_non_chunked, silent=True)

        # test slide in step of 155 over the sample possibilities where the start is between 0 and 1000, and the end is between end-1000 and end
        for start_sample in range(0, 1000, 155):
            for end_sample in range(hdr['number_of_samples'], hdr['number_of_samples'] - 1000, -155):
                print('Test - start sample: ' + str(start_sample) + ' - end sample:' + str(end_sample))
                self._test_reader_output_matrix(np.array([mne_raw._data[1, start_sample:end_sample]]), np.finfo(np.float32).eps,
                                                'MNE sliding ranges',
                                                start_sample=start_sample, end_sample=end_sample, channels=('CH02',),
                                                skip_stdIO_non_chunked=skip_stdIO_non_chunked, silent=True)


        mne_raw.close()
        del mne_raw._data
        del mne_raw

        ConsoleColors.print_green('Sliding ranges test successful')
    """

    def test04_fileio_bv_validate_data_multi_range(self):
        # multiple ranges - all channels

        #
        # matlab data
        #
        import scipy.io
        mat = scipy.io.loadmat(self.matlab_data_path + self.orient_path + 'multiranges.mat')
        self._test_reader_output_matrix(mat['dat'], 0, 'Matlab multiple sample ranges', start_sample=(1000, 101000), end_sample=(2000, 102000))
        del mat
        

        #
        # MNE data
        #
        import numpy as np
        from mne.io import read_raw_brainvision
        mne_raw = read_raw_brainvision(self.bv_data_path, eog=[], misc=[], preload=True, verbose=None)
        mne_raw._data *= 1000000    # MNE always returns in V, we dictate mV in our read, so convert MNE data to mV here

        self._test_reader_output_matrix(np.stack((mne_raw._data[:, 1000:2000], mne_raw._data[:, 101000:102000]), axis=-1), np.finfo(np.float32).eps, 'MNE multiple sample ranges', start_sample=(1000, 101000), end_sample=(2000, 102000))
        mne_raw.close()
        del mne_raw._data
        del mne_raw

        ConsoleColors.print_green('Multiple sample ranges test successful')


    def _test_reader_output_matrix(self, mat, diff_tolerance, title, channels=None, start_sample=0, end_sample=-1, skip_stdIO_non_chunked=False, silent=False):
        """
        Test the output values of four different types of reading (chunked/non-chucked and stdIO/memmap) against
        a given array of values

        mat (nparray):                      Array of values to test against
        diff_tolerance (float):             The difference between the output values that is tolerated.
                                            The precision of the float values and manipulations from MNE cause slight
                                            differences (~.00000000001 for MNE) in the outputs, this tolerance value
                                            allows for less precise testing
        title (str):                        The title of the test
        skip_stdIO_non_chunked (bool):      Whether to skip the standard IO non-chunked reading test. In a multiplexed
                                            dataset, when either not reading all channels or not reading them in
                                            the same order as in which they are stored, then each samples is read separately
                                            in nested loops. Reading per sample is very slow, and this type of reading
                                            is automatically switched to chunked reading. Therefore non-chunked reading
                                            under these conditions might not actually happen and can be skipped

        """
        from ieegprep.fileio.BrainVisionReader import BrainVisionReader
        #import numpy as np

        #
        def assert_results(hdr, data_io, type):
            self.assertEqual(hdr['data_orientation'], self.data_orientation.upper(), 'Data orientation mismatch. Make sure the \'data_orientation\' variable in the test (' + self.data_orientation.upper() + ') matches the validation set\'s \'DataOrientation\' field (' + hdr['data_orientation'] + ')')

            # find the differences in data and report on the largest difference
            diff = abs(mat - data_io)
            #max_diff = diff.max()
            #if max_diff > 0:
            #    max_diff_index = np.unravel_index(diff.argmax(), diff.shape)
            #    if not silent:
            #        print(' - A maximum difference of ' + str(max_diff) + ' was found at channel ' + hdr['channel_names'][max_diff_index[0]] + ' (index ' + str(max_diff_index[0]) + '). At least at sample: ' + str(max_diff_index[1]) + ' (read value: ' + str(data_io[max_diff_index[0], max_diff_index[1]]) + '; matlab/MNE value: ' + str(mat[max_diff_index[0], max_diff_index[1]]) + ')')

            # assert
            self.assertEqual((diff <= diff_tolerance).sum(), data_io.size, ('Validation of \'' + self.matlab_data_path + '*.mat\' failed while testing ' + title + ' (' + hdr['data_orientation'].lower() + ') and reading with ' + type))

        #
        # stdIO, chunked
        #
        if not silent:
            ConsoleColors.print_green('Testing ' + title + ' with chunked stdIO')
        hdr, data_io = BrainVisionReader.bv_read_data(self.bv_data_path, channels=channels,
                                                      start_sample=start_sample, end_sample=end_sample, unit='uV',
                                                      use_memmap=False, chunked_read=True)
        assert_results(hdr, data_io, 'chunked stdIO')
        del hdr, data_io

        #
        # stdIO, non-chunked
        #
        if not skip_stdIO_non_chunked:
            if not silent:
                ConsoleColors.print_green('Testing ' + title + ' with non-chunked stdIO')
            hdr, data_io = BrainVisionReader.bv_read_data(self.bv_data_path, channels=channels,
                                                          start_sample=start_sample, end_sample=end_sample, unit='uV',
                                                          use_memmap=False, chunked_read=False)
            assert_results(hdr, data_io, 'non-chunked stdIO')
            del data_io

        #
        # memmap, chunked
        #
        if not silent:
            ConsoleColors.print_green('Testing ' + title + ' with chunked memmap')
        hdr, data_io = BrainVisionReader.bv_read_data(self.bv_data_path, channels=channels,
                                                      start_sample=start_sample, end_sample=end_sample, unit='uV',
                                                      use_memmap=True, chunked_read=True)
        assert_results(hdr, data_io, 'chunked memmap')
        del data_io

        #
        # memmap, non-chunked
        #
        if not silent:
            ConsoleColors.print_green('Testing ' + title + ' with non-chunked memmap')
        hdr, data_io = BrainVisionReader.bv_read_data(self.bv_data_path, channels=channels,
                                                      start_sample=start_sample, end_sample=end_sample, unit='uV',
                                                      use_memmap=True, chunked_read=False)
        assert_results(hdr, data_io, 'non-chunked memmap')
        del data_io


if __name__ == '__main__':
    unittest.main()
