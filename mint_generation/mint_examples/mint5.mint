DEVICE ml5

LAYER FLOW

PICOINJECTOR picoinjector_1
PICOINJECTOR picoinjector_2
MIXER mixer_1
PICOINJECTOR picoinjector_3
DROPLET SORTER droplet_sorter_1


CHANNEL channel_1 from PICOINJECTOR picoinjector_1 1 to PICOINJECTOR picoinjector_2 2 channelWidth=400;
CHANNEL channel_2 from PICOINJECTOR picoinjector_2 1 to MIXER mixer_1 2 channelWidth=400;
CHANNEL channel_3 from MIXER mixer_1 1 to PICOINJECTOR picoinjector_3 2 channelWidth=400;
CHANNEL channel_4 from PICOINJECTOR picoinjector_3 1 to DROPLET SORTER droplet_sorter_1 2 channelWidth=400;

END LAYER