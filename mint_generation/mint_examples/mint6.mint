DEVICE ml6

LAYER FLOW

DROPLET SORTER droplet_sorter_1 ;
NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1 ;
DROPLET SORTER droplet_sorter_2 ;
DROPLET SPLITTER droplet_splitter_1 ;
DROPLET SORTER droplet_sorter_3 ;


CHANNEL channel_1 from   droplet_sorter_1 1 to    nozzle_droplet_generator_1 2 channelWidth=400 ;
CHANNEL channel_2 from    nozzle_droplet_generator_1 1 to   droplet_sorter_2 2 channelWidth=400 ;
CHANNEL channel_3 from   droplet_sorter_2 1 to   droplet_splitter_1 2 channelWidth=400 ;
CHANNEL channel_4 from   droplet_splitter_1 1 to   droplet_sorter_3 2 channelWidth=400 ;

END LAYER