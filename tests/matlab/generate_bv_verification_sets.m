%
% This script reads the BrainVision test data using Fieldtrip and stores various outputs:
%   - single channel (all samples)					 -> bv_allsamples_CH07.mat
%   - full set (all channels, all samples)  		 -> bv_allsamples_allchannels.mat
%   - 100 000 samples (all channels)				 -> bv_100ksamples.mat
%   - multiple ranges of 1000 samples (all channels) -> bv_multiranges.mat
%
% Consequently, these subsets of data can be used to automatically validate
% the python BrainVision reader by running it's unittest (test_fileio_bv_data.py)
%
%
% Note: the 'DataOrientation' field in the '.vhdr' can be set to either 'MULTIPLEXED' or
%       'VECTORIZED' to produce output subsets and test both orientations
% 
% Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

output_path = 'D:\BIDS_erdetect\';
bv_file = 'D:\BIDS_erdetect\sub-BV\ses-1\ieeg\sub-BV_ses-1_ieeg';

% 
hdr = read_brainvision_vhdr([bv_file, '.vhdr']);

% single channel (all samples)
dat = read_brainvision_eeg([bv_file, '.eeg'], hdr, 1, hdr.nSamples, 7);     % 'CH07'
save([output_path, 'bv_allsamples_CH07.mat'], 'hdr', 'dat');
clear dat

% full set (all channels, all samples)
dat = read_brainvision_eeg([brainvision_file, '.eeg'], hdr, 1, hdr.nSamples, []);
save([output_path, 'bv_allsamples_allchannels.mat'], 'hdr', 'dat', '-v7.3');
clear dat

% 100 000 samples (all channels)
dat = read_brainvision_eeg([brainvision_file, '.eeg'], hdr, 1001, 101000, []);
save([output_path, 'bv_100ksamples.mat'], 'hdr', 'dat');
clear dat

% multiple ranges (all channels)
dat_1 = read_brainvision_eeg([brainvision_file, '.eeg'], hdr, 1001, 2000, []);
dat_2 = read_brainvision_eeg([brainvision_file, '.eeg'], hdr, 101001, 102000, []);
dat = cat(3, dat_1, dat_2);
save([output_path, 'bv_multiranges.mat'], 'hdr', 'dat');
clear dat_1 dat_2 dat
