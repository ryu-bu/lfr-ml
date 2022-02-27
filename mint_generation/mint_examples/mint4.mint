DEVICE ml4

LAYER FLOW

PICOINJECTOR picoinjector_1
NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1
DROPLET SPLITTER droplet_splitter_1
NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2
DIAMOND CHAMBER diamond_chamber_1


CHANNEL channel_1 from PICOINJECTOR picoinjector_1 1 to NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1 2 channelWidth=400;
CHANNEL channel_2 from NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1 1 to DROPLET SPLITTER droplet_splitter_1 2 channelWidth=400;
CHANNEL channel_3 from DROPLET SPLITTER droplet_splitter_1 1 to NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2 2 channelWidth=400;
CHANNEL channel_4 from NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2 1 to DIAMOND CHAMBER diamond_chamber_1 2 channelWidth=400;

END LAYER