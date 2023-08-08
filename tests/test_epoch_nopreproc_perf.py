"""
Unit tests to measure the performances of different epoch routines for the BrainVision, EDF and MEF3 formats

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
from ieegprep.utils.misc import time_func, clear_virtual_cache
from ieegprep.bids.data_epoch import _prepare_input, _load_data_epochs__by_channels, _load_data_epochs__by_trials, _load_data_epochs__by_channels__withPrep
from ieegprep.bids.sidecars import load_stim_event_info


class TestEpochNoPreProcPerf(unittest.TestCase):
    """
    ...
    """

    bv_data_path = 'D:\\BIDS_erdetect\\sub-BV\\ses-1\\ieeg\\sub-BV_ses-1_ieeg.vhdr'
    edf_data_path = 'D:\\BIDS_erdetect\\sub-EDF\\ses-ieeg01\\ieeg\\sub-EDF_ses-ieeg01_ieeg.edf'
    mef_data_path = 'D:\\BIDS_erdetect\\sub-MEF\\ses-ieeg01\\ieeg\\sub-MEF_ses-ieeg01_ieeg.mefd'
    #bv_data_path = os.path.expanduser('~/Documents/ERDetect_perf/sub-BV/ses-1/ieeg/sub-BV_ses-1_ieeg.vhdr')
    #edf_data_path = os.path.expanduser('~/Documents/ERDetect_perf/sub-EDF/ses-ieeg01/ieeg/sub-EDF_ses-ieeg01_ieeg.edf')
    #mef_data_path = os.path.expanduser('~/Documents/ERDetect_perf/sub-MEF/ses-ieeg01/ieeg/sub-MEF_ses-ieeg01_ieeg.edf')

    test_baseline_norm = None
    test_baseline_epoch = (-1, -0.1)
    test_trial_epoch = (-1, 3)


    def test_epoch_perf(self):
        """
        Test reader performance
        """

        uncached_read_repetitions = 25
        cached_read_repetitions = 25

        # define tests
        tests = dict()

        tests['by_channels__bv_mult'] = dict()
        tests['by_channels__bv_mult']['test_name']= 'Epoch (no preprocessing), Brainvision (multiplexed), _load_data_epochs__by_channels'
        tests['by_channels__bv_mult']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_channels__bv_mult']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_channels__bv_mult']['data_path'] = self.bv_data_path
        tests['by_channels__bv_mult']['by_routine'] = 'channels'
        tests['by_channels__bv_mult']['set_bv_orientation'] = 'MULTIPLEXED'
        tests['by_channels__bv_mult']['conditions'] = []
        tests['by_channels__bv_mult']['conditions'].append(False)            # not preloaded
        tests['by_channels__bv_mult']['conditions'].append(True)             # preloaded

        tests['by_channels__bv_vect'] = dict()
        tests['by_channels__bv_vect']['test_name']= 'Epoch (no preprocessing), Brainvision (vectorized), _load_data_epochs__by_channels'
        tests['by_channels__bv_vect']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_channels__bv_vect']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_channels__bv_vect']['data_path'] = self.bv_data_path
        tests['by_channels__bv_vect']['by_routine'] = 'channels'
        tests['by_channels__bv_vect']['set_bv_orientation'] = 'VECTORIZED'
        tests['by_channels__bv_vect']['conditions'] = []
        tests['by_channels__bv_vect']['conditions'].append(False)            # not preloaded
        tests['by_channels__bv_vect']['conditions'].append(True)             # preloaded

        tests['by_channels__edf'] = dict()
        tests['by_channels__edf']['test_name']= 'Epoch (no preprocessing), EDF, _load_data_epochs__by_channels'
        tests['by_channels__edf']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_channels__edf']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_channels__edf']['data_path'] = self.edf_data_path
        tests['by_channels__edf']['by_routine'] = 'channels'
        tests['by_channels__edf']['set_bv_orientation'] = ''
        tests['by_channels__edf']['conditions'] = []
        tests['by_channels__edf']['conditions'].append(False)            # not preloaded
        tests['by_channels__edf']['conditions'].append(True)             # preloaded

        tests['by_channels__mef'] = dict()
        tests['by_channels__mef']['test_name']= 'Epoch (no preprocessing), MEF3, _load_data_epochs__by_channels'
        tests['by_channels__mef']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_channels__mef']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_channels__mef']['data_path'] = self.mef_data_path
        tests['by_channels__mef']['by_routine'] = 'channels'
        tests['by_channels__mef']['set_bv_orientation'] = ''
        tests['by_channels__mef']['conditions'] = []
        tests['by_channels__mef']['conditions'].append(False)            # not preloaded
        tests['by_channels__mef']['conditions'].append(True)             # preloaded

        tests['by_trials__bv_mult'] = dict()
        tests['by_trials__bv_mult']['test_name']= 'Epoch (no preprocessing), Brainvision (multiplexed), _load_data_epochs__by_trials'
        tests['by_trials__bv_mult']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_trials__bv_mult']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_trials__bv_mult']['data_path'] = self.bv_data_path
        tests['by_trials__bv_mult']['by_routine'] = 'trials'
        tests['by_trials__bv_mult']['set_bv_orientation'] = 'MULTIPLEXED'
        tests['by_trials__bv_mult']['conditions'] = []
        tests['by_trials__bv_mult']['conditions'].append(False)            # not preloaded
        tests['by_trials__bv_mult']['conditions'].append(True)             # preloaded

        tests['by_trials__bv_vect'] = dict()
        tests['by_trials__bv_vect']['test_name']= 'Epoch (no preprocessing), Brainvision (vectorized), _load_data_epochs__by_trials'
        tests['by_trials__bv_vect']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_trials__bv_vect']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_trials__bv_vect']['data_path'] = self.bv_data_path
        tests['by_trials__bv_vect']['by_routine'] = 'trials'
        tests['by_trials__bv_vect']['set_bv_orientation'] = 'VECTORIZED'
        tests['by_trials__bv_vect']['conditions'] = []
        tests['by_trials__bv_vect']['conditions'].append(False)            # not preloaded
        tests['by_trials__bv_vect']['conditions'].append(True)             # preloaded

        tests['by_trials__edf'] = dict()
        tests['by_trials__edf']['test_name']= 'Epoch (no preprocessing), EDF, _load_data_epochs__by_trials'
        tests['by_trials__edf']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_trials__edf']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_trials__edf']['data_path'] = self.edf_data_path
        tests['by_trials__edf']['by_routine'] = 'trials'
        tests['by_trials__edf']['set_bv_orientation'] = ''
        tests['by_trials__edf']['conditions'] = []
        tests['by_trials__edf']['conditions'].append(False)            # not preloaded
        tests['by_trials__edf']['conditions'].append(True)             # preloaded

        tests['by_trials__mef'] = dict()
        tests['by_trials__mef']['test_name']= 'Epoch (no preprocessing), MEF3, _load_data_epochs__by_trials'
        tests['by_trials__mef']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_trials__mef']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_trials__mef']['data_path'] = self.mef_data_path
        tests['by_trials__mef']['by_routine'] = 'trials'
        tests['by_trials__mef']['set_bv_orientation'] = ''
        tests['by_trials__mef']['conditions'] = []
        tests['by_trials__mef']['conditions'].append(False)            # not preloaded
        tests['by_trials__mef']['conditions'].append(True)             # preloaded

        tests['by_prep_mem__bv_mult'] = dict()
        tests['by_prep_mem__bv_mult']['test_name']= 'Epoch (no preprocessing), Brainvision (multiplexed), _load_data_epochs__by_channels__withPrep'
        tests['by_prep_mem__bv_mult']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_prep_mem__bv_mult']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_prep_mem__bv_mult']['data_path'] = self.bv_data_path
        tests['by_prep_mem__bv_mult']['by_routine'] = 'prep_mem'
        tests['by_prep_mem__bv_mult']['set_bv_orientation'] = 'MULTIPLEXED'
        tests['by_prep_mem__bv_mult']['conditions'] = []
        tests['by_prep_mem__bv_mult']['conditions'].append(False)            # not preloaded
        tests['by_prep_mem__bv_mult']['conditions'].append(True)             # preloaded

        tests['by_prep_mem__bv_vect'] = dict()
        tests['by_prep_mem__bv_vect']['test_name']= 'Epoch (no preprocessing), Brainvision (vectorized), _load_data_epochs__by_channels__withPrep'
        tests['by_prep_mem__bv_vect']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_prep_mem__bv_vect']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_prep_mem__bv_vect']['data_path'] = self.bv_data_path
        tests['by_prep_mem__bv_vect']['by_routine'] = 'prep_mem'
        tests['by_prep_mem__bv_vect']['set_bv_orientation'] = 'VECTORIZED'
        tests['by_prep_mem__bv_vect']['conditions'] = []
        tests['by_prep_mem__bv_vect']['conditions'].append(False)            # not preloaded
        tests['by_prep_mem__bv_vect']['conditions'].append(True)             # preloaded

        tests['by_prep_mem__edf'] = dict()
        tests['by_prep_mem__edf']['test_name']= 'Epoch (no preprocessing), EDF, _load_data_epochs__by_channels__withPrep'
        tests['by_prep_mem__edf']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_prep_mem__edf']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_prep_mem__edf']['data_path'] = self.edf_data_path
        tests['by_prep_mem__edf']['by_routine'] = 'prep_mem'
        tests['by_prep_mem__edf']['set_bv_orientation'] = ''
        tests['by_prep_mem__edf']['conditions'] = []
        tests['by_prep_mem__edf']['conditions'].append(False)            # not preloaded
        tests['by_prep_mem__edf']['conditions'].append(True)             # preloaded

        tests['by_prep_mem__mef'] = dict()
        tests['by_prep_mem__mef']['test_name']= 'Epoch (no preprocessing), MEF3, _load_data_epochs__by_channels__withPrep'
        tests['by_prep_mem__mef']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_prep_mem__mef']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_prep_mem__mef']['data_path'] = self.mef_data_path
        tests['by_prep_mem__mef']['by_routine'] = 'prep_mem'
        tests['by_prep_mem__mef']['set_bv_orientation'] = ''
        tests['by_prep_mem__mef']['conditions'] = []
        tests['by_prep_mem__mef']['conditions'].append(False)            # not preloaded
        tests['by_prep_mem__mef']['conditions'].append(True)             # preloaded

        tests['by_prep_speed__bv_mult'] = dict()
        tests['by_prep_speed__bv_mult']['test_name']= 'Epoch (no preprocessing), Brainvision (multiplexed), _load_data_epochs__by_channels__withPrep'
        tests['by_prep_speed__bv_mult']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_prep_speed__bv_mult']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_prep_speed__bv_mult']['data_path'] = self.bv_data_path
        tests['by_prep_speed__bv_mult']['by_routine'] = 'prep_speed'
        tests['by_prep_speed__bv_mult']['set_bv_orientation'] = 'MULTIPLEXED'
        tests['by_prep_speed__bv_mult']['conditions'] = []
        tests['by_prep_speed__bv_mult']['conditions'].append(False)            # not preloaded
        tests['by_prep_speed__bv_mult']['conditions'].append(True)             # preloaded

        tests['by_prep_speed__bv_vect'] = dict()
        tests['by_prep_speed__bv_vect']['test_name']= 'Epoch (no preprocessing), Brainvision (vectorized), _load_data_epochs__by_channels__withPrep'
        tests['by_prep_speed__bv_vect']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_prep_speed__bv_vect']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_prep_speed__bv_vect']['data_path'] = self.bv_data_path
        tests['by_prep_speed__bv_vect']['by_routine'] = 'prep_speed'
        tests['by_prep_speed__bv_vect']['set_bv_orientation'] = 'VECTORIZED'
        tests['by_prep_speed__bv_vect']['conditions'] = []
        tests['by_prep_speed__bv_vect']['conditions'].append(False)            # not preloaded
        tests['by_prep_speed__bv_vect']['conditions'].append(True)             # preloaded

        tests['by_prep_speed__edf'] = dict()
        tests['by_prep_speed__edf']['test_name']= 'Epoch (no preprocessing), EDF, _load_data_epochs__by_channels__withPrep'
        tests['by_prep_speed__edf']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_prep_speed__edf']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_prep_speed__edf']['data_path'] = self.edf_data_path
        tests['by_prep_speed__edf']['by_routine'] = 'prep_speed'
        tests['by_prep_speed__edf']['set_bv_orientation'] = ''
        tests['by_prep_speed__edf']['conditions'] = []
        tests['by_prep_speed__edf']['conditions'].append(False)            # not preloaded
        tests['by_prep_speed__edf']['conditions'].append(True)             # preloaded

        tests['by_prep_speed__mef'] = dict()
        tests['by_prep_speed__mef']['test_name']= 'Epoch (no preprocessing), MEF3, _load_data_epochs__by_channels__withPrep'
        tests['by_prep_speed__mef']['uncached_read_repetitions'] = uncached_read_repetitions
        tests['by_prep_speed__mef']['cached_read_repetitions'] = cached_read_repetitions
        tests['by_prep_speed__mef']['data_path'] = self.mef_data_path
        tests['by_prep_speed__mef']['by_routine'] = 'prep_speed'
        tests['by_prep_speed__mef']['set_bv_orientation'] = ''
        tests['by_prep_speed__mef']['conditions'] = []
        tests['by_prep_speed__mef']['conditions'].append(False)            # not preloaded
        tests['by_prep_speed__mef']['conditions'].append(True)             # preloaded


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
            ConsoleColors.print_green('Test: ' + tests[test_name]['test_name'])

            tests[test_name]['cond_prep_results_uncached'] = []
            tests[test_name]['cond_prep_results_cached'] = []
            tests[test_name]['cond_epoch_results_uncached'] = []
            tests[test_name]['cond_epoch_results_cached'] = []

            # load the stim trial onsets
            trial_onsets, _, _ = load_stim_event_info(tests[test_name]['data_path'][0:tests[test_name]['data_path'].rindex("_ieeg")] + '_events.tsv')

            # loop over the conditions
            for preload_condition in test['conditions']:
                text_preloaded = 'preloaded' if preload_condition else 'not_preloaded'
                print('    - Condition - ' + text_preloaded)

                # test with cache clearing
                if test['uncached_read_repetitions'] > 0:
                    print('      - Uncached (with cache-clearing)')
                    clear_virtual_cache()

                    # test un-cached _prepare_input
                    # Note: adding 'clear_virtual_cache' as pre_func will clear the virtual cache (page memory)
                    #       before reading (but will not be taken into account in the performance measurement)
                    print('          - _prepare_input (with cache-clearing)')
                    mean, std, range, times = time_func(self._time_prepare_input, clear_virtual_cache, test['uncached_read_repetitions'],
                                                        tests[test_name]['data_path'], preload_condition)

                    print('              Mean: ' + str(round(mean, 2)) + ' - std: ' + str(round(std, 2)) + ' - range: ' + str(round(range[0], 2)) + '-' + str(round(range[1], 2)))
                    tests[test_name]['cond_prep_results_uncached'].append(np.asarray([mean, std, range, times], dtype=object))

                    # prepare input
                    clear_virtual_cache()
                    data_reader, baseline_method, out_of_bound_method = _prepare_input(tests[test_name]['data_path'],
                                                                                       trial_epoch=self.test_trial_epoch, baseline_norm=self.test_baseline_norm, baseline_epoch=self.test_baseline_epoch,
                                                                                       out_of_bound_handling='error', preload_data=preload_condition)
                    if tests[test_name]['set_bv_orientation'] is not None and tests[test_name]['set_bv_orientation'] != '':
                        data_reader.bv_hdr['data_orientation'] = tests[test_name]['set_bv_orientation']

                    # test un-cached
                    # Note: adding 'clear_virtual_cache' as pre_func will clear the virtual cache (page memory)
                    #       before reading (but will not be taken into account in the performance measurement)
                    print('          - _load_data_epochs (with cache-clearing)')
                    if tests[test_name]['by_routine'] == 'channels':
                        mean, std, range, times = time_func(_load_data_epochs__by_channels, clear_virtual_cache, test['uncached_read_repetitions'],
                                                            data_reader, data_reader.channel_names, trial_onsets,
                                                            trial_epoch=self.test_trial_epoch,
                                                            baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                            out_of_bound_method=out_of_bound_method)

                    elif tests[test_name]['by_routine'] == 'trials':
                        mean, std, range, times = time_func(_load_data_epochs__by_trials, clear_virtual_cache, test['uncached_read_repetitions'],
                                                            data_reader, data_reader.channel_names, trial_onsets,
                                                            trial_epoch=self.test_trial_epoch,
                                                            baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                            out_of_bound_method=out_of_bound_method)

                    elif tests[test_name]['by_routine'] in ('prep_mem', 'prep_speed'):
                        mean, std, ret_range, times = time_func(_load_data_epochs__by_channels__withPrep, clear_virtual_cache, test['uncached_read_repetitions'],
                                                                False, data_reader, data_reader.channel_names, trial_onsets,
                                                                trial_epoch=self.test_trial_epoch,
                                                                baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                                out_of_bound_method=out_of_bound_method, metric_callbacks=None,
                                                                high_pass=False, early_reref=None, line_noise_removal=None, late_reref=None,
                                                                priority='mem' if tests[test_name]['by_routine'] == 'prep_mem' else 'speed')

                    data_reader.close()

                    print('              Mean: ' + str(round(mean, 2)) + ' - std: ' + str(round(std, 2)) + ' - range: ' + str(round(range[0], 2)) + '-' + str(round(range[1], 2)))
                    tests[test_name]['cond_epoch_results_uncached'].append(np.asarray([mean, std, range, times], dtype=object))


                # test while cached
                if test['cached_read_repetitions'] > 0:
                    print('      - Cached')
                    clear_virtual_cache()

                    # once to make sure the data is cached
                    self._time_prepare_input(tests[test_name]['data_path'], preload_condition)

                    # test cached _prepare_input
                    print('          - _prepare_input (cached)')
                    mean, std, range, times = time_func(self._time_prepare_input, None, test['cached_read_repetitions'],
                                                        tests[test_name]['data_path'], preload_condition)
                    print('              Mean: ' + str(round(mean, 2)) + ' - std: ' + str(round(std, 2)) + ' - range: ' + str(round(range[0], 2)) + '-' + str(round(range[1], 2)))
                    tests[test_name]['cond_prep_results_cached'].append(np.asarray([mean, std, range, times], dtype=object))

                    # prepare input
                    clear_virtual_cache()
                    data_reader, baseline_method, out_of_bound_method = _prepare_input(tests[test_name]['data_path'],
                                                                                       trial_epoch=self.test_trial_epoch, baseline_norm=self.test_baseline_norm, baseline_epoch=self.test_baseline_epoch,
                                                                                       out_of_bound_handling='error', preload_data=preload_condition)
                    if tests[test_name]['set_bv_orientation'] is not None and tests[test_name]['set_bv_orientation'] != '':
                        data_reader.bv_hdr['data_orientation'] = tests[test_name]['set_bv_orientation']

                    # once to make sure the data is cached
                    if tests[test_name]['by_routine'] == 'channels':
                        _load_data_epochs__by_channels(data_reader, data_reader.channel_names, trial_onsets,
                                                       trial_epoch=self.test_trial_epoch,
                                                       baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                       out_of_bound_method=out_of_bound_method)

                    elif tests[test_name]['by_routine'] == 'trials':
                        _load_data_epochs__by_trials(data_reader, data_reader.channel_names, trial_onsets,
                                                     trial_epoch=self.test_trial_epoch,
                                                     baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                     out_of_bound_method=out_of_bound_method)

                    elif tests[test_name]['by_routine'] in ('prep_mem', 'prep_speed'):
                        _load_data_epochs__by_channels__withPrep(False, data_reader, data_reader.channel_names, trial_onsets,
                                                                 trial_epoch=self.test_trial_epoch,
                                                                 baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                                 out_of_bound_method=out_of_bound_method, metric_callbacks=None,
                                                                 high_pass=False, early_reref=None, line_noise_removal=None, late_reref=None,
                                                                 priority='mem' if tests[test_name]['by_routine'] == 'prep_mem' else 'speed')

                    # test cached
                    print('          - _load_data_epochs (cached)')
                    if tests[test_name]['by_routine'] == 'channels':
                        mean, std, range, times = time_func(_load_data_epochs__by_channels, None, test['cached_read_repetitions'],
                                                            data_reader, data_reader.channel_names, trial_onsets,
                                                            trial_epoch=self.test_trial_epoch,
                                                            baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                            out_of_bound_method=out_of_bound_method)

                    elif tests[test_name]['by_routine'] == 'trials':
                        mean, std, range, times = time_func(_load_data_epochs__by_trials, None, test['cached_read_repetitions'],
                                                            data_reader, data_reader.channel_names, trial_onsets,
                                                            trial_epoch=self.test_trial_epoch,
                                                            baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                            out_of_bound_method=out_of_bound_method)

                    elif tests[test_name]['by_routine'] in ('prep_mem', 'prep_speed'):
                        mean, std, ret_range, times = time_func(_load_data_epochs__by_channels__withPrep, None, test['cached_read_repetitions'],
                                                                False, data_reader, data_reader.channel_names, trial_onsets,
                                                                trial_epoch=self.test_trial_epoch,
                                                                baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                                out_of_bound_method=out_of_bound_method, metric_callbacks=None,
                                                                high_pass=False, early_reref=None, line_noise_removal=None, late_reref=None,
                                                                priority='mem' if tests[test_name]['by_routine'] == 'prep_mem' else 'speed')

                    data_reader.close()

                    print('              Mean: ' + str(round(mean, 2)) + ' - std: ' + str(round(std, 2)) + ' - range: ' + str(round(range[0], 2)) + '-' + str(round(range[1], 2)))
                    tests[test_name]['cond_epoch_results_cached'].append(np.asarray([mean, std, range, times], dtype=object))

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
        output_file = 'results_epoch_' + os_text + '__' + datetime.now().strftime("%Y%m%d_%H%M%S")

        # as pickle
        with open('./' + output_file + '.pdat', 'wb') as output_handle:
            pickle.dump([platform_info, tests], output_handle)

        # as matlab
        import scipy
        output = dict()
        output['platform'] = platform_info
        output['tests'] = tests
        scipy.io.savemat('./' + output_file + '.mat', output)

        #
        ConsoleColors.print_green('Tests complete')


    def _time_prepare_input(self, data_path, preload_condition):
        data_reader, baseline_method, out_of_bound_method = _prepare_input(data_path,
                                                                           trial_epoch=self.test_trial_epoch, baseline_norm=self.test_baseline_norm, baseline_epoch=self.test_baseline_epoch,
                                                                           out_of_bound_handling='error', preload_data=preload_condition)
        data_reader.close()


if __name__ == '__main__':
    unittest.main()
