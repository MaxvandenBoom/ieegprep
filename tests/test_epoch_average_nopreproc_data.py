"""
Unit tests to test whether the different subroutines for epoching-averaging yield exactly the same results (for different data-types)


=====================================================
Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import unittest
from ieegprep.bids.data_epoch import _prepare_input, _load_data_epoch_averages__by_channel_condition_trial, _load_data_epoch_averages__by_condition_trials, _load_data_epochs__by_channels__withPrep
from ieegprep.bids.sidecars import load_elec_stim_events
from ieegprep.utils.console import ConsoleColors


class TestEpochAverageNoPreProcData(unittest.TestCase):
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

    def test01_epoch__bv_multiplexed__no_preload_data(self):
        test_name = 'Epoch & Average (no preprocessing), Brainvision (multiplexed), no preload'
        self._run_test(test_name, self.bv_data_path, preload_data=False, set_bv_orientation='MULTIPLEXED')

    def test02_epoch__bv_multiplexed__preload_data(self):
        test_name = 'Epoch & Average (no preprocessing), Brainvision (multiplexed), preloaded'
        self._run_test(test_name, self.bv_data_path, preload_data=True, set_bv_orientation='MULTIPLEXED')

    def test03_epoch__bv_vectorized__no_preload_data(self):
        test_name = 'Epoch & Average (no preprocessing), Brainvision (vectorized), no preload'
        self._run_test(test_name, self.bv_data_path, preload_data=False, set_bv_orientation='VECTORIZED')

    def test04_epoch__bv_vectorized__preload_data(self):
        test_name = 'Epoch & Average (no preprocessing), Brainvision (vectorized), preloaded'
        self._run_test(test_name, self.bv_data_path, preload_data=True, set_bv_orientation='VECTORIZED')

    def test05_epoch__edf__no_preload_data(self):
        test_name = 'Epoch & Average (no preprocessing), EDF, no preload'
        self._run_test(test_name, self.edf_data_path, preload_data=False)

    def test06_epoch__edf__preload_data(self):
        test_name = 'Epoch & Average (no preprocessing), EDF, preloaded'
        self._run_test(test_name, self.edf_data_path, preload_data=True)

    def test07_epoch__mef__no_preload_data(self):
        test_name = 'Epoch & Average (no preprocessing), MEF, no preload'
        self._run_test(test_name, self.mef_data_path, preload_data=False)

    def test08_epoch__mef__preload_data(self):
        test_name = 'Epoch & Average (no preprocessing), MEF, preloaded'
        self._run_test(test_name, self.mef_data_path, preload_data=True)


    def _run_test(self, test_name, data_path, preload_data, set_bv_orientation=None):
        ConsoleColors.print_green('Test data: ' + test_name)
        print('  - data_path: ' + data_path)
        print('  - preload_data: ' + str(preload_data))
        if set_bv_orientation is not None:
            print('  - set_bv_orientation: ' + set_bv_orientation)

        # load the trial onsets for each of the stimulation conditions
        _, _, conditions_onsets, _ = load_elec_stim_events(data_path[0:data_path.rindex('_ieeg')] + '_events.tsv',
                                                                                concat_bidirectional_stimpairs=True)




        #
        data_reader, baseline_method, out_of_bound_method = _prepare_input(data_path,
                                                                           trial_epoch=self.test_trial_epoch, baseline_norm=self.test_baseline_norm, baseline_epoch=self.test_baseline_epoch,
                                                                           out_of_bound_handling='error', preload_data=preload_data)
        if set_bv_orientation is not None:
            data_reader.bv_hdr['data_orientation'] = set_bv_orientation

        channel_condition_trial__sampling_rate, \
        channel_condition_trial__data, _ = _load_data_epoch_averages__by_channel_condition_trial(data_reader, data_reader.channel_names, conditions_onsets,
                                                                                                 trial_epoch=self.test_trial_epoch,
                                                                                                 baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                                                                 out_of_bound_method=out_of_bound_method, metric_callbacks=None)

        condition_trials__sampling_rate, \
        condition_trials__data, _ = _load_data_epoch_averages__by_condition_trials(data_reader, data_reader.channel_names, conditions_onsets,
                                                                                   trial_epoch=self.test_trial_epoch,
                                                                                   baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                                                   out_of_bound_method=out_of_bound_method, metric_callbacks=None)

        prep_mem__sampling_rate, \
        prep_mem__data, _ = _load_data_epochs__by_channels__withPrep(True, data_reader, data_reader.channel_names, conditions_onsets,
                                                                     trial_epoch=self.test_trial_epoch,
                                                                     baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                                     out_of_bound_method=out_of_bound_method, metric_callbacks=None,
                                                                     high_pass=False, early_reref=None, line_noise_removal=None, late_reref=None,
                                                                     priority='mem')

        prep_speed__sampling_rate, \
        prep_speed__data, _ = _load_data_epochs__by_channels__withPrep(True, data_reader, data_reader.channel_names, conditions_onsets,
                                                                       trial_epoch=self.test_trial_epoch,
                                                                       baseline_method=baseline_method, baseline_epoch=self.test_baseline_epoch,
                                                                       out_of_bound_method=out_of_bound_method, metric_callbacks=None,
                                                                       high_pass=False, early_reref=None, line_noise_removal=None, late_reref=None,
                                                                       priority='speed')

        # compare by_channel_condition_trial and by_condition_trials
        diff1 = abs(channel_condition_trial__data - condition_trials__data)
        self.assertEqual((diff1 == 0).sum(), channel_condition_trial__data.size, ('Validation of test \'' + test_name + '\' failed while comparing to \'by_channel_condition_trial\' and \'by_condition_trials\''))

        # compare by_channel_condition_trial and by_channels__withPrep (mem)
        diff2 = abs(channel_condition_trial__data - prep_mem__data)
        self.assertEqual((diff2 == 0).sum(), channel_condition_trial__data.size, ('Validation of test \'' + test_name + '\' failed while comparing to \'by_channel_condition_trial\' and \'by_channels__withPrep (mem)\''))

        diff3 = abs(channel_condition_trial__data - prep_speed__data)
        self.assertEqual((diff3 == 0).sum(), channel_condition_trial__data.size, ('Validation of test \'' + test_name + '\' failed while comparing to \'by_channel_condition_trial\' and \'by_channels__withPrep (speed)\''))

        #
        data_reader.close()
        ConsoleColors.print_green('Test ' + test_name + ' successful\n\n\n')


if __name__ == '__main__':
    unittest.main()
