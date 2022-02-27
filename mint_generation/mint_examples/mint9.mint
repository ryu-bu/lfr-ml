DEVICE ml9

LAYER FLOW

NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1
MIXER mixer_1
MIXER mixer_2
PICOINJECTOR picoinjector_1
DIAMOND CHAMBER diamond_chamber_1


CHANNEL channel_1 from NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1 1 to MIXER mixer_1 2 channelWidth=400;
CHANNEL channel_2 from MIXER mixer_1 1 to MIXER mixer_2 2 channelWidth=400;
CHANNEL channel_3 from MIXER mixer_2 1 to PICOINJECTOR picoinjector_1 2 channelWidth=400;
CHANNEL channel_4 from PICOINJECTOR picoinjector_1 1 to DIAMOND CHAMBER diamond_chamber_1 2 channelWidth=400;

END LAYER