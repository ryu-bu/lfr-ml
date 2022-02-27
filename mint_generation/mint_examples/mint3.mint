DEVICE ml3

LAYER FLOW

NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1
NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2
PICOINJECTOR picoinjector_1
PICOINJECTOR picoinjector_2
DROPLET SPLITTER droplet_splitter_1


CHANNEL channel_1 from NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1 1 to NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2 2 channelWidth=400;
CHANNEL channel_2 from NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2 1 to PICOINJECTOR picoinjector_1 2 channelWidth=400;
CHANNEL channel_3 from PICOINJECTOR picoinjector_1 1 to PICOINJECTOR picoinjector_2 2 channelWidth=400;
CHANNEL channel_4 from PICOINJECTOR picoinjector_2 1 to DROPLET SPLITTER droplet_splitter_1 2 channelWidth=400;

END LAYER