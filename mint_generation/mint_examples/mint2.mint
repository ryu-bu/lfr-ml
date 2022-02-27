DEVICE ml2

LAYER FLOW

DIAMOND CHAMBER diamond_chamber_1
PICOINJECTOR picoinjector_1
MIXER mixer_1
PICOINJECTOR picoinjector_2
NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1


CHANNEL channel_1 from DIAMOND CHAMBER diamond_chamber_1 1 to PICOINJECTOR picoinjector_1 2 channelWidth=400;
CHANNEL channel_2 from PICOINJECTOR picoinjector_1 1 to MIXER mixer_1 2 channelWidth=400;
CHANNEL channel_3 from MIXER mixer_1 1 to PICOINJECTOR picoinjector_2 2 channelWidth=400;
CHANNEL channel_4 from PICOINJECTOR picoinjector_2 1 to NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1 2 channelWidth=400;

END LAYER