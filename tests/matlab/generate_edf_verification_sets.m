%
% This script reads the EDF test data using Fieldtrip and stores various outputs:
%   - single channel (all samples)					 -> edf_allsamples_CH07.mat
%   - full set (all channels, all samples)  		 -> edf_allsamples_allchannels.mat
%   - 100 000 samples (all channels)				 -> edf_100ksamples.mat
%   - multiple ranges of 1000 samples (all channels) -> edf_multiranges.mat
%
% Consequently, these subsets of data can be used to automatically validate
% the python EDF reader by running it's unittest (test_fileio_edf_data.py)
% 
% Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

output_path = 'D:\BIDS_erdetect\';
edf_file = 'D:\BIDS_erdetect\sub-EDF\ses-ieeg01\ieeg\sub-EDF_ses-ieeg01_ieeg.edf';

% 
hdr = read_edf(edf_file);

% single channel
dat = read_edf(edf_file, hdr, 1, hdr.nSamples, 6);     % 'EEG LMS2-Ref'
save([output_path, 'edf_allsamples_CH-LMS2.mat'], 'hdr', 'dat');
clear dat

% full set (all channels, all samples)
% Note: since the header read already took out the last 'annotations' channel, all channels in the header should be read)
dat = read_edf(edf_file, hdr, 1, hdr.nSamples, 1:length(hdr.label));
save([output_path, 'edf_allsamples_allchannels.mat'], 'hdr', 'dat', '-v7.3');
clear dat

% 100 000 samples (all channels)
% Note: since the header read already took out the last 'annotations' channel, all channels in the header should be read)
dat = read_edf(edf_file, hdr, 1001, 101000, 1:length(hdr.label));
save([output_path, 'edf_100ksamples.mat'], 'hdr', 'dat');
clear dat

% multiple ranges (all channels)
% Note: since the header read already took out the last 'annotations' channel, all channels in the header should be read)
dat_1 = read_edf(edf_file, hdr, 1001, 2000, 1:length(hdr.label));
dat_2 = read_edf(edf_file, hdr, 101001, 102000, 1:length(hdr.label));
dat = cat(3, dat_1, dat_2);
save([output_path, 'edf_multiranges.mat'], 'hdr', 'dat');
clear dat_1 dat_2 dat
