"""
Helper class for the re-referencing of BIDS data


=====================================================
Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
from ieegprep.utils.misc import is_number


class RerefStruct:
    """
    Sets up and stores a re-referencing structure
    """

    groups = list()                     # re-referencing groups, each group entry holds the channels (names) that should be re-referenced over
    channel_group = dict()              # for each channel, the re-reference group that should be used to re-referencing it
    channel_exclude_epochs = None       # a dictionary with for each channel the epochs that should be excluded from re-referencing

    def __init__(self, groups, channel_reref_group):
        self.groups = groups
        self.channel_group = channel_reref_group

    @classmethod
    def generate_car(cls, channels):
        """
        Factory method to generate a Common Average Re-referencing (CAR) setup struct

        Args:
            channels (list or tuple):  Channels that might need to be re-referenced (or are needed for re-referencing)

        """

        # create a single group with all the channels from the channels argument
        groups = list()
        groups.append(channels.copy())

        # set each channel to be referenced to the same group
        channel_reref_group = dict()
        for channel in channels:
            channel_reref_group[channel] = 0

        return cls(groups, channel_reref_group)

    @classmethod
    def generate_car_per_headbox(cls, channel_names, channel_headboxes):
        """
        Factory method to generate a Common Average Re-referencing (CAR) per headbox setup struct

        Args:
            channel_names (list or tuple):      Channel names that might need to be re-referenced (or are needed for re-referencing)
            channel_headboxes (list or tuple):  Channel headboxes, order should correspond with the channels_names input argument

        """

        # create a single group with all the channels from the channels argument
        groups = list()
        channel_reref_group = dict()

        # loop over the unique headboxes
        unique_headboxes = sorted(list(set(channel_headboxes)))
        group_counter = 0
        for headbox in unique_headboxes:
            if is_number(headbox) and not headbox.lower() == 'nan' and not headbox.lower() == 'n/a':

                # find the channel names that belong to this headbox
                headbox_channels = [channel_names[ind] for ind, x in enumerate(channel_headboxes) if x == headbox]

                # store the channel names as a group
                groups.append(headbox_channels)

                # store for each channel in this group that they below to this group
                for channel_name in headbox_channels:
                    channel_reref_group[channel_name] = group_counter

                # raise the group counter
                group_counter += 1

        return cls(groups, channel_reref_group)


    def get_required_channels(self, retrieve_channels):
        """
        Lookup all the channels (names) that are needed for re-referencing, given the channels that need to be retrieved

        Args:
            retrieve_channels (list or tuple):  Channels that of which the data needs to be retrieved and re-referenced

        Note:   The list of channels that is returned can be different from the channels to need to be referenced (e.g. maybe
                only ECOG channels need to be re-referenced while re-referencing does have to occur over both ECOG and SEEG channels)
        """

        # create a list for all the re-references channels that are required for the channels that need to be retrieved
        all_channels = list()

        # loop over all requested channels
        for channel in retrieve_channels:
            if channel not in self.channel_group.keys():
                logging.error('Could not find requested channel ' + channel + ' in reref struct, make sure each included channel is also set for re-referencing')
                raise ValueError('Could not find requested channel')

            else:

                # for the requested channel, loop over the channels in the group that it needs for re-referencing
                for grp_channel in self.groups[self.channel_group[channel]]:

                    # add to list of required channels
                    if grp_channel not in all_channels:
                        all_channels.append(grp_channel)

        return all_channels


    def get_required_groups(self, retrieve_channels):
        """
        Retrieve all the groups that are needed for re-referencing, given the channels that need to be retrieved

        Args:
            retrieve_channels (list or tuple):  Channels that of which the data needs to be retrieved and re-referenced

        Note:   These can be different from the channels to need to be referenced (e.g. maybe only ECOG channels need
                to be re-referenced while re-referencing does have to occur over both ECOG and SEEG channels)
        """

        # create a list with all the groups required
        all_groups = list()

        # loop over all requested channels
        for channel in retrieve_channels:
            if channel not in self.channel_group.keys():
                logging.error('Could not find requested channel ' + channel + ' in reref struct')
                raise ValueError('Could not find requested channel')

            else:

                # add the group to the list of required groups
                if self.channel_group[channel] not in all_groups:
                    all_groups.append(self.channel_group[channel])

        return all_groups

    def set_exclude_reref_epochs(self, exclude_onsets, exclude_epoch=(-.01, 1.0), split_channel_names=None):
        """
        Set channel-epochs that should be excluded from re-referencing
        (this can be used when electrical stimulation was performed on specific channels at specific moments in the data)

        Args:
            exclude_onsets (dict with lists):   The onsets of the channel-epochs that need to be excluded from
                                                re-referencing. The argument should be a dictionary where the key
                                                of each entry in the dictionary represents the channel-name of which
                                                epochs should be excluded, and the value of the entry a list of
                                                onsets (in time) that needs to be excluded for that channel.
                                                A window around each onset - as defined by epoch - will define what
                                                will be excluded from re-referencing
            exclude_epoch (tuple):              The time-span that will be excluded around each onset in the
                                                channel-data. Expressed as a tuple with the start- and end-point in
                                                seconds relative to the onset (e.g. the standard tuple of '-1, 3' will
                                                exclude the signal in the period from 1s before the onset to 3s after
                                                onset).
            split_channel_names (str):          The channel-names to exclude ('exclude_onsets argument' keys) can actually
                                                consist of two channel names (which is the case for stimulated electrode
                                                pairs; e.g. a key could be 'Ch01-Ch02', where actually the onsets to
                                                exclude apply to both channel Ch01 and Ch02). If set, this arguments
                                                indicates on which character to split the channel-names (key) before
                                                adding the onsets to each of the separate channel-names to exclude.
        """

        # for each channel store the epoch windows which should be excluded from re-referencing
        self.channel_exclude_epochs = dict()
        for channel, onsets in exclude_onsets.items():

            if split_channel_names is None:
                # exclude_onsets dictionary keys are simply the channel-names

                self.channel_exclude_epochs[channel] = list()
                for onset in onsets:
                    self.channel_exclude_epochs[channel].append((onset + exclude_epoch[0], onset + exclude_epoch[1]))

            else:
                # exclude_onsets dictionary keys indicate multiple channel-names

                # separate the key into the separate channel-names
                split_channels = channel.split(split_channel_names)
                for sub_channel in split_channels:

                    # make sure the (sub) channel-name exists to hold exclusion epoch windows
                    if sub_channel not in self.channel_exclude_epochs.keys():
                        self.channel_exclude_epochs[sub_channel] = list()

                    for onset in onsets:
                        self.channel_exclude_epochs[sub_channel].append((onset + exclude_epoch[0], onset + exclude_epoch[1]))

