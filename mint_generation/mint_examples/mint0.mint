DEVICE ml0

LAYER FLOW

DIAMOND CHAMBER diamond_chamber_1
MIXER mixer_1
MIXER mixer_2
DROPLET SPLITTER droplet_splitter_1
DIAMOND CHAMBER diamond_chamber_2


CHANNEL channel_1 from DIAMOND CHAMBER diamond_chamber_1 1 to MIXER mixer_1 2 channelWidth=400;
CHANNEL channel_2 from MIXER mixer_1 1 to MIXER mixer_2 2 channelWidth=400;
CHANNEL channel_3 from MIXER mixer_2 1 to DROPLET SPLITTER droplet_splitter_1 2 channelWidth=400;
CHANNEL channel_4 from DROPLET SPLITTER droplet_splitter_1 1 to DIAMOND CHAMBER diamond_chamber_2 2 channelWidth=400;

END LAYER