# IeegPrep
IeegPrep is a python library to read, pre-process and epoch Intracranial Electroencephalography (iEEG) data that is structured according to the Brain Imaging Data Structure (BIDS) 


## Install

```
pip install ieegprep
```

## Usage

After installing, the library can be used directly to:

Read EDF, BrainVision or MEF3 data
```
from ieegprep import IeegDataReader

# initialize a reader for a specific dataset
reader = IeegDataReader('/bids_data_root/subj-01/ieeg/sub-01_run-06_ieeg.vhdr', data_preload=False)

# print metadata
print(reader.sampling_rate)
print(reader.channel_names)

# retrieve all the data from single channel
channel_data      = reader.retrieve_channel_data('CH05')

# retrieve a range of sample-data from all, a single or multiple channels
range_data_all    = reader.retrieve_sample_range_data(sample_start=1000, sample_end=2000)
range_data_single = reader.retrieve_sample_range_data(sample_start=1000, sample_end=2000, channels='CH01')
range_data_multi  = reader.retrieve_sample_range_data(sample_start=1000, sample_end=2000, channels=('CH01', 'CH05'))

# optionally, close the reader and release the memory
reader.close()
```

Read BIDS sidecar metadata files
```
from ieegprep import load_event_info, load_channel_info

channels = load_channel_info('/bids_data_root/subj-01/ieeg/sub-01_run-06_channels.tsv')
events   = load_event_info('/bids_data_root/subj-01/ieeg/sub-01_run-06_events.tsv')
```

Load, pre-process and epoched data as a matrix
```
from ieegprep import load_data_epochs

# retrieve epoched data
[srate, epochs] = load_data_epochs( '/bids_data_root/subj-01/ieeg/sub-01_run-06_ieeg.vhdr',
                                    retrieve_channels = channels['name'].tolist(),
                                    onsets            = numpy.asfarray(events['onset']))
data_ch0_trial13 = epochs[1, 13, :]

# retrieve epoched data specifying the epoch window and baseline normalization
[srate, epochs] = load_data_epochs( '/bids_data_root/subj-01/ieeg/sub-01_run-06_ieeg.vhdr',
                                    retrieve_channels = channels['name'].tolist(),
                                    onsets            = numpy.asfarray(events['onset']),
                                    trial_epoch       = (-1, 2),                         #  -1s < onset < 2s  
                                    baseline_norm     = 'Median',
                                    baseline_epoch    = (-1, -0.1))
                            
# retrieve epoched data with pre-processing (high-pass filtering, CAR re-referencing and 50Hz line-noise removal)
from ieegprep import RerefStruct
[srate, epochs] = load_data_epochs( '/bids_data_root/subj-01/ieeg/sub-01_run-06_ieeg.vhdr',
                                    retrieve_channels  = channels['name'].tolist(),
                                    onsets             = numpy.asfarray(events['onset']),
                                    high_pass          = True,
                                    early_reref        = RerefStruct.generate_car(channels['name'].tolist()),
                                    line_noise_removal = 50)
```

Load, epoch and get the average of each stimulated electrode pair (minimizing memory usage)
```
from ieegprep import load_data_epochs_averages

# seperate the different stimulated electrode-pairs (e.g. Ch01-Ch02...) and use each stim-pair as a trial "condition"
conditions_onsets = dict()
for row in events.itertuples():
    conditions_onsets.setdefault(row.electrical_stimulation_site, list())
    conditions_onsets[row.electrical_stimulation_site].append(float(row.onset))

# retrieve epoch data averaged over conditions
[srate, epochs, _] =  load_data_epochs_averages('/bids_data_root/subj-01/ieeg/sub-01_run-06_ieeg.vhdr',
                                                retrieve_channels = channels['name'].tolist(),
                                                conditions_onsets = list(conditions_onsets.values()))
data_respCh01_stimCh02_03 = epochs[1, 4, :]
```

## Acknowledgements

- Written by Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)
- Dependencies:
  - PyMef by Jan Cimbalnik, Matt Stead, Ben Brinkmann, and Dan Crepeau (https://github.com/msel-source/pymef)
  - NumPy
  - SciPy
  - Pandas
  - psutil
- The new EDF and BrainVision readers are in part adaptations of the Fieldtrip code (by Robert Robert Oostenveld) and replicate some of the header logic from the MNE package (by Teon Brooks, Stefan Appelhoff and others)  
  
- This project was funded by the National Institute Of Mental Health of the National Institutes of Health Award Number R01MH122258 to Dora Hermes

