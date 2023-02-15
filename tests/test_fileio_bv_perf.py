"""
Unit tests to validate the performance of the BrainVision reader

This class tests measures the performance of reading (chunked or non-chunked, standard IO and Memmap):
   - a single channel (all samples)
   - the full set (all channels, all samples)
   - 100 000 samples (all channels)
   - multiple (two) ranges of 1000 samples (all channels)

The performance results are stored as a pickle (.pdat) and a Matlab (.mat) file and optionally can be visualized
using the 'matlab/visualize_perfTestResults.m' script


=====================================================
Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import unittest
import pickle
import numpy as np
from datetime import datetime
from ieegprep.utils.console import ConsoleColors
from ieegprep.utils.misc import clear_virtual_cache
from ieegprep.fileio.BrainVisionReader import BrainVisionReader

class TestFileIO(unittest.TestCase):
    """
    Validate the performance of the BrainVision reader
    """

    data_orientation = 'multiplexed'
    #data_orientation = 'vectorized'

    bv_data_path = 'D:\\BIDS_erdetect\\sub-BV\\ses-1\\ieeg\\sub-BV_ses-1_ieeg.vhdr'
    #bv_data_path = os.path.expanduser('~/Documents/ERDetect_perf/sub-BV/ses-1/ieeg/sub-BV_ses-1_ieeg.vhdr')


    def test_fileio_bv_perf(self):
        """
        Test reader performance
        """
        from ieegprep.utils.misc import time_func

        uncached_read_repetitions = 20
        cached_read_repetitions = 100

        # define tests
        tests = dict()
        tests['single_channel'] = dict()
        tests['single_channel']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['single_channel']['cached_read_repetitions'] = cached_read_repetitions
        tests['single_channel']['channels'] = ('CH07',)
        tests['single_channel']['start_sample'] = 0
        tests['single_channel']['end_sample'] = -1
        tests['single_channel']['conditions'] = []
        tests['single_channel']['conditions'].append((True, True))              # memmap & chunked
        tests['single_channel']['conditions'].append((True, False))             # memmap & non-chunked
        tests['single_channel']['conditions'].append((False, True))             # stdIO & chunked
        if self.data_orientation.lower() == 'vectorized':                           # (skip on multiplexed)
            tests['single_channel']['conditions'].append((False, False))        # stdIO & non-chunked

        tests['full'] = dict()
        tests['full']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['full']['cached_read_repetitions'] = cached_read_repetitions
        tests['full']['channels'] = ()
        tests['full']['start_sample'] = 0
        tests['full']['end_sample'] = -1
        tests['full']['conditions'] = []
        tests['full']['conditions'].append((True, True))                # memmap & chunked
        tests['full']['conditions'].append((True, False))               # memmap & non-chunked
        tests['full']['conditions'].append((False, True))               # stdIO & chunked
        tests['full']['conditions'].append((False, False))              # stdIO & non-chunked

        tests['range_100k'] = dict()
        tests['range_100k']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['range_100k']['cached_read_repetitions'] = cached_read_repetitions
        tests['range_100k']['channels'] = ()
        tests['range_100k']['start_sample'] = 1000
        tests['range_100k']['end_sample'] = 101000
        tests['range_100k']['conditions'] = []
        tests['range_100k']['conditions'].append((True, True))               # memmap & chunked
        tests['range_100k']['conditions'].append((True, False))              # memmap & non-chunked
        tests['range_100k']['conditions'].append((False, True))              # stdIO & chunked
        tests['range_100k']['conditions'].append((False, False))             # stdIO & non-chunked

        tests['multi_range'] = dict()
        tests['multi_range']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['multi_range']['cached_read_repetitions'] = cached_read_repetitions
        tests['multi_range']['channels'] = ()
        tests['multi_range']['start_sample'] = [1000, 101000]
        tests['multi_range']['end_sample'] = [2000, 102000]
        tests['multi_range']['conditions'] = []
        tests['multi_range']['conditions'].append((True, True))          # memmap & chunked
        tests['multi_range']['conditions'].append((True, False))         # memmap & non-chunked
        tests['multi_range']['conditions'].append((False, True))         # stdIO & chunked
        tests['multi_range']['conditions'].append((False, False))        # stdIO & non-chunked

        # check if orientation matches
        hdr = BrainVisionReader.bv_read_header(self.bv_data_path)
        self.assertEqual(hdr['data_orientation'], self.data_orientation.upper(), 'Data orientation mismatch. Make sure the \'data_orientation\' variable in the test (' + self.data_orientation.upper() + ') matches the validation set\'s \'DataOrientation\' field (' + hdr['data_orientation'] + ')')

        # retrieve platform data
        import platform
        platform_info = dict()
        platform_info['platform'] = platform.system()
        platform_info['platform-release'] = platform.release()
        platform_info['platform-version'] = platform.version()
        platform_info['architecture'] = platform.machine()
        platform_info['processor'] = platform.processor()


        #
        # perform tests
        #

        # loop over the tests
        for test_name, test in tests.items():
            ConsoleColors.print_green('Test: ' + test_name)

            tests[test_name]['cond_results_uncached'] = []
            tests[test_name]['cond_results_cached'] = []

            # loop over the conditions
            for condition in test['conditions']:
                text_reading_type = 'memmap' if condition[0] else 'stdIO'
                text_reading_chunked = 'chunked' if condition[1] else 'non-chunked'
                print('    - Condition - ' + text_reading_type + ' - ' + text_reading_chunked)

                # test with cache clearing
                if test['uncached_read_repetitions'] > 0:
                    print('      - Uncached (with cache-clearing)')

                    # test un-cached
                    # Note: adding 'clear_virtual_cache' as pre_func will clear the virtual cache (page memory)
                    #       before reading (but will not be taken into account in the performance measurement)
                    mean, std, range, times = time_func(self._test_reader_read, clear_virtual_cache, test['uncached_read_repetitions'], self.bv_data_path,
                                                        use_memmap=condition[0], chunked_read=condition[1],
                                                        channels=test['channels'], start_sample=test['start_sample'], end_sample=test['end_sample'])

                    print('          Mean: ' + str(round(mean, 2)) + ' - std: ' + str(round(std, 2)) + ' - range: ' + str(round(range[0], 2)) + '-' + str(round(range[1], 2)))
                    tests[test_name]['cond_results_uncached'].append(np.asarray([mean, std, range, times], dtype=object))

                # test while cached
                if test['cached_read_repetitions'] > 0:
                    print('      - Cached')

                    # once to make sure the data is cached
                    self._test_reader_read(self.bv_data_path, use_memmap=condition[0], chunked_read=condition[1],
                                           channels=test['channels'], start_sample=test['start_sample'], end_sample=test['end_sample'])

                    # test cached
                    mean, std, range, times = time_func(self._test_reader_read, None, test['cached_read_repetitions'], self.bv_data_path,
                                                        use_memmap=condition[0], chunked_read=condition[1],
                                                        channels=test['channels'], start_sample=test['start_sample'], end_sample=test['end_sample'])

                    print('          Mean: ' + str(round(mean, 2)) + ' - std: ' + str(round(std, 2)) + ' - range: ' + str(round(range[0], 2)) + '-' + str(round(range[1], 2)))
                    tests[test_name]['cond_results_cached'].append(np.asarray([mean, std, range, times], dtype=object))

                # clear the virtual cache at the end of the read condition
                clear_virtual_cache()


        #
        # Store the results to disk
        #

        # build the filename
        os_text = 'unk'
        if platform_info['platform'].lower() in ('win32', 'windows'):
            os_text = 'win'
        elif platform_info['platform'].lower() in ('linux', 'linux2'):
            os_text = 'lnx'
        elif platform_info['platform'].lower() == 'darwin':
            os_text = 'mac'
        output_file = 'results_bv_' + os_text + '_' + self.data_orientation + '__' + datetime.now().strftime("%Y%m%d_%H%M%S")

        # as pickle
        with open('./' + output_file + '.pdat', 'wb') as output_handle:
            pickle.dump([platform_info, self.data_orientation, tests], output_handle)

        # as matlab
        import scipy
        output = dict()
        output['platform'] = platform_info
        output['data_orientation'] = self.data_orientation
        output['tests'] = tests
        scipy.io.savemat('./' + output_file + '.mat', output)

        #
        ConsoleColors.print_green('Tests complete')


    @staticmethod
    def _test_reader_read(data_path, use_memmap, chunked_read, channels, start_sample=0, end_sample=-1):
        """
        Perform a single read with the reader

        Args:
            data_path (str):
            use_memmap (bool):              Whether to use numpy's memmap (which wraps around mmap) while reading the data.
            chunked_read (bool):            Whether to read or transfer the data in chunks
            channels (str/list/tuple):      The names of the channels to return the signal data from.
            start_sample (int or list):     The start-point in time (in samples) to start reading from (0-based)
            end_sample (int or list):       The sample to end the reading (0-based)

        """

        hdr, data_io = BrainVisionReader.bv_read_data(data_path,
                                                      use_memmap=use_memmap, chunked_read=chunked_read,
                                                      channels=channels, start_sample=start_sample, end_sample=end_sample,
                                                      unit='uV')
        del hdr, data_io


if __name__ == '__main__':
    unittest.main()
