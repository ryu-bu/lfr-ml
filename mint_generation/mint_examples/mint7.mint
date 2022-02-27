DEVICE ml7

LAYER FLOW

MIXER mixer_1
DIAMOND CHAMBER diamond_chamber_1
MIXER mixer_2
DROPLET SORTER droplet_sorter_1
MIXER mixer_3


CHANNEL channel_1 from MIXER mixer_1 1 to DIAMOND CHAMBER diamond_chamber_1 2 channelWidth=400;
CHANNEL channel_2 from DIAMOND CHAMBER diamond_chamber_1 1 to MIXER mixer_2 2 channelWidth=400;
CHANNEL channel_3 from MIXER mixer_2 1 to DROPLET SORTER droplet_sorter_1 2 channelWidth=400;
CHANNEL channel_4 from DROPLET SORTER droplet_sorter_1 1 to MIXER mixer_3 2 channelWidth=400;

END LAYER