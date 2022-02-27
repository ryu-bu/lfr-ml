DEVICE ml6

LAYER FLOW

NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1
DIAMOND CHAMBER diamond_chamber_1
DROPLET SORTER droplet_sorter_1
NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2
PICOINJECTOR picoinjector_1


CHANNEL channel_1 from NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1 1 to DIAMOND CHAMBER diamond_chamber_1 2 channelWidth=400;
CHANNEL channel_2 from DIAMOND CHAMBER diamond_chamber_1 1 to DROPLET SORTER droplet_sorter_1 2 channelWidth=400;
CHANNEL channel_3 from DROPLET SORTER droplet_sorter_1 1 to NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2 2 channelWidth=400;
CHANNEL channel_4 from NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2 1 to PICOINJECTOR picoinjector_1 2 channelWidth=400;

END LAYER