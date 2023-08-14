"""
Unit tests to monitor the memory usage while loading and epoching-averaging different data types


=====================================================
Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import sys
import unittest
from ieegprep.bids.data_epoch import _prepare_input, _load_data_epoch_averages__by_channel_condition_trial, _load_data_epoch_averages__by_condition_trials, _load_data_epochs__by_channels__withPrep
from ieegprep.bids.sidecars import load_elec_stim_events
from ieegprep.utils.console import ConsoleColors
from ieegprep.utils.misc import clear_virtual_cache
from memory_profiler import memory_usage, profile


class TestEpochAverageNoPreProcMem(unittest.TestCase):
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


    #
    # by_channel_condition_trial
    #

    def test01_epoch__by_channels__bv_multiplexed__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_channel_condition_trial, Brainvision (multiplexed), no preload'
        self._run_test(test_name, self.bv_data_path, 'channel_condition_trial', preload_data=False, set_bv_orientation='MULTIPLEXED')

    def test02_epoch__by_channel_condition_trial__bv_vectorized__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_channel_condition_trial, Brainvision (vectorized), no preload'
        self._run_test(test_name, self.bv_data_path, 'channel_condition_trial', preload_data=False, set_bv_orientation='VECTORIZED')

    def test03_epoch__by_channel_condition_trial__bv_multiplexed__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_channel_condition_trial, Brainvision (multiplexed), preloaded'
        self._run_test(test_name, self.bv_data_path, 'channel_condition_trial', preload_data=True, set_bv_orientation='MULTIPLEXED')

    def test04_epoch__by_channel_condition_trial__bv_vectorized__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_channel_condition_trial, Brainvision (vectorized), preloaded'
        self._run_test(test_name, self.bv_data_path, 'channel_condition_trial', preload_data=True, set_bv_orientation='VECTORIZED')

    def test05_epoch__by_channel_condition_trial__edf__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_channel_condition_trial, EDF, no preload'
        self._run_test(test_name, self.edf_data_path, 'channel_condition_trial', preload_data=False)

    def test06_epoch__by_channel_condition_trial__edf__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_channel_condition_trial, EDF, preloaded'
        self._run_test(test_name, self.edf_data_path, 'channel_condition_trial', preload_data=True)
    
    def test07_epoch__by_channel_condition_trial__mef__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_channel_condition_trial, MEF, no preload'
        self._run_test(test_name, self.mef_data_path, 'channel_condition_trial', preload_data=False)

    def test08_epoch__by_channel_condition_trial__mef__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_channel_condition_trial, MEF, preloaded'
        self._run_test(test_name, self.mef_data_path, 'channel_condition_trial', preload_data=True)


    #
    # by_condition_trials
    #

    def test09_epoch__by_condition_trials__bv_multiplexed__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_condition_trials, Brainvision (multiplexed), no preload'
        self._run_test(test_name, self.bv_data_path, 'condition_trials', preload_data=False, set_bv_orientation='MULTIPLEXED')

    def test10_epoch__by_condition_trials__bv_vectorized__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_condition_trials, Brainvision (vectorized), no preload'
        self._run_test(test_name, self.bv_data_path, 'condition_trials', preload_data=False, set_bv_orientation='VECTORIZED')

    def test11_epoch__by_condition_trials__bv_multiplexed__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_condition_trials, Brainvision (multiplexed), preloaded'
        self._run_test(test_name, self.bv_data_path, 'condition_trials', preload_data=True, set_bv_orientation='MULTIPLEXED')

    def test12_epoch__by_condition_trials__bv_vectorized__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_condition_trials, Brainvision (vectorized), preloaded'
        self._run_test(test_name, self.bv_data_path, 'condition_trials', preload_data=True, set_bv_orientation='VECTORIZED')

    def test13_epoch__by_condition_trials__edf__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_condition_trials, EDF, no preload'
        self._run_test(test_name, self.edf_data_path, 'condition_trials', preload_data=False)

    def test14_epoch__by_condition_trials__edf__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_condition_trials, EDF, preloaded'
        self._run_test(test_name, self.edf_data_path, 'condition_trials', preload_data=True)

    def test15_epoch__by_condition_trials__mef__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_condition_trials, MEF, no preload'
        self._run_test(test_name, self.mef_data_path, 'condition_trials', preload_data=False)

    def test16_epoch__by_condition_trials__mef__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epoch_averages__by_condition_trials, MEF, preloaded'
        self._run_test(test_name, self.mef_data_path, 'condition_trials', preload_data=True)


    #
    # _load_data_epochs__by_channels__withPrep (mem)
    #

    def test17_epoch__by_prep_mem__bv_multiplexed__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (mem), Brainvision (multiplexed), no preload'
        self._run_test(test_name, self.bv_data_path, 'prep_mem', preload_data=False, set_bv_orientation='MULTIPLEXED')

    def test18_epoch__by_prep_mem__bv_vectorized__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (mem), Brainvision (vectorized), no preload'
        self._run_test(test_name, self.bv_data_path, 'prep_mem', preload_data=False, set_bv_orientation='VECTORIZED')

    def test19_epoch__by_prep_mem__bv_multiplexed__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (mem), Brainvision (multiplexed), preloaded'
        self._run_test(test_name, self.bv_data_path, 'prep_mem', preload_data=True, set_bv_orientation='MULTIPLEXED')

    def test20_epoch__by_prep_mem__bv_vectorized__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (mem), Brainvision (vectorized), preloaded'
        self._run_test(test_name, self.bv_data_path, 'prep_mem', preload_data=True, set_bv_orientation='VECTORIZED')

    def test21_epoch__by_prep_mem__edf__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (mem), EDF, no preload'
        self._run_test(test_name, self.edf_data_path, 'prep_mem', preload_data=False)

    def test22_epoch__by_prep_mem__edf__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (mem), EDF, preloaded'
        self._run_test(test_name, self.edf_data_path, 'prep_mem', preload_data=True)

    def test23_epoch__by_prep_mem__mef__no_preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (mem), MEF, no preload'
        self._run_test(test_name, self.mef_data_path, 'prep_mem', preload_data=False)

    def test24_epoch__by_prep_mem__mef__preload_mem(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (mem), MEF, preloaded'
        self._run_test(test_name, self.mef_data_path, 'prep_mem', preload_data=True)


    #
    # _load_data_epochs__by_channels__withPrep (speed)
    #

    def test25_epoch__by_prep_mem__bv_multiplexed__no_preload_speed(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (speed), Brainvision (multiplexed), no preload'
        self._run_test(test_name, self.bv_data_path, 'prep_speed', preload_data=False, set_bv_orientation='MULTIPLEXED')

    def test26_epoch__by_prep_mem__bv_vectorized__no_preload_speed(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (speed), Brainvision (vectorized), no preload'
        self._run_test(test_name, self.bv_data_path, 'prep_speed', preload_data=False, set_bv_orientation='VECTORIZED')

    def test27_epoch__by_prep_mem__bv_multiplexed__preload_speed(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (speed), Brainvision (multiplexed), preloaded'
        self._run_test(test_name, self.bv_data_path, 'prep_speed', preload_data=True, set_bv_orientation='MULTIPLEXED')

    def test28_epoch__by_prep_mem__bv_vectorized__preload_speed(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (speed), Brainvision (vectorized), preloaded'
        self._run_test(test_name, self.bv_data_path, 'prep_speed', preload_data=True, set_bv_orientation='VECTORIZED')

    def test29_epoch__by_prep_mem__edf__no_preload_speed(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (speed), EDF, no preload'
        self._run_test(test_name, self.edf_data_path, 'prep_speed', preload_data=False)

    def test30_epoch__by_prep_mem__edf__preload_speed(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (speed), EDF, preloaded'
        self._run_test(test_name, self.edf_data_path, 'prep_speed', preload_data=True)

    def test31_epoch__by_prep_mem__mef__no_preload_speed(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (speed), MEF, no preload'
        self._run_test(test_name, self.mef_data_path, 'prep_speed', preload_data=False)

    def test32_epoch__by_prep_mem__mef__preload_speed(self):
        test_name = 'Epoch & Average (no preprocessing), _load_data_epochs__by_channels__withPrep (speed), MEF, preloaded'
        self._run_test(test_name, self.mef_data_path, 'prep_speed', preload_data=True)


    @profile
    def _prepare_and_epoch(self, data_path, by_routine, conditions_onsets, preload_data, set_bv_orientation=None):
        """

        """

        #
        data_reader, baseline_method, out_of_bound_method = _prepare_input(data_path,
                                                                           trial_epoch=self.test_trial_epoch, baseline_norm=self.test_baseline_norm, baseline_epoch=self.test_baseline_epoch,
                                                                           out_of_bound_handling='error', preload_data=preload_data)
        if set_bv_orientation is not None:
            data_reader.bv_hdr['data_orientation'] = set_bv_orientation

        if by_routine == 'channel_condition_trial':
            sampling_rate, data, _ = _load_data_epoch_averages__by_channel_condition_trial(data_reader, data_reader.channel_names, conditions_onsets,
                                                       trial_epoch=self.test_trial_epoch,
                                                       baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                       out_of_bound_method=out_of_bound_method, metric_callbacks=None)

        elif by_routine == 'condition_trials':
            sampling_rate, data, _ = _load_data_epoch_averages__by_condition_trials(data_reader, data_reader.channel_names, conditions_onsets,
                                                     trial_epoch=self.test_trial_epoch,
                                                     baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                     out_of_bound_method=out_of_bound_method, metric_callbacks=None)

        elif by_routine in ('prep_mem', 'prep_speed'):
            sampling_rate, data, _ = _load_data_epochs__by_channels__withPrep(True, data_reader, data_reader.channel_names, conditions_onsets,
                                                                              trial_epoch=self.test_trial_epoch,
                                                                              baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                                              out_of_bound_method=out_of_bound_method, metric_callbacks=None,
                                                                              high_pass=False, early_reref=None, line_noise_removal=None, late_reref=None,
                                                                              priority='mem' if by_routine == 'prep_mem' else 'speed')

        data_reader.close()


    def _run_test(self, test_name, data_path, by_routine, preload_data, set_bv_orientation=None):
        ConsoleColors.print_green('Test mem: ' + test_name)
        print('  - data_path: ' + data_path)
        print('  - by_routine: ' + by_routine)
        print('  - preload_data: ' + str(preload_data))
        if set_bv_orientation is not None:
            print('  - set_bv_orientation: ' + set_bv_orientation)

        # load the trial onsets for each of the stimulation conditions
        clear_virtual_cache()
        _, _, conditions_onsets, _ = load_elec_stim_events(data_path[0:data_path.rindex('_ieeg')] + '_events.tsv',
                                                                                concat_bidirectional_stimpairs=True)

        # test the memory usage of the preparation and average epoch
        if set_bv_orientation is None:
            mem_usage = memory_usage((self._prepare_and_epoch, (data_path, by_routine, conditions_onsets, preload_data)),
                                     interval=0.005, include_children=True, multiprocess=True, max_usage=True)
        else:
            mem_usage = memory_usage((self._prepare_and_epoch, (data_path, by_routine, conditions_onsets, preload_data), {'set_bv_orientation': set_bv_orientation}),
                                     interval=0.005, include_children=True, multiprocess=True, max_usage=True)

        #
        print('Peak memory usage: ' + str(mem_usage))
        ConsoleColors.print_green('Test ' + test_name + ' successful\n\n\n')


if __name__ == '__main__':
    unittest.main()
