DEVICE ml9

LAYER FLOW

DROPLET SORTER droplet_sorter_1 ;
NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1 ;
PICOINJECTOR picoinjector_1 ;
NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2 ;
DIAMOND CHAMBER diamond_chamber_1 ;


CHANNEL channel_1 from   droplet_sorter_1 1 to    nozzle_droplet_generator_1 2 channelWidth=400 ;
CHANNEL channel_2 from    nozzle_droplet_generator_1 1 to  picoinjector_1 2 channelWidth=400 ;
CHANNEL channel_3 from  picoinjector_1 1 to    nozzle_droplet_generator_2 2 channelWidth=400 ;
CHANNEL channel_4 from    nozzle_droplet_generator_2 1 to   diamond_chamber_1 2 channelWidth=400 ;

END LAYER