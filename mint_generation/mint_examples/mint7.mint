DEVICE ml7

LAYER FLOW

NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1 ;
DROPLET SPLITTER droplet_splitter_1 ;
NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2 ;
DIAMOND CHAMBER diamond_chamber_1 ;
PICOINJECTOR picoinjector_1 ;


CHANNEL channel_1 from    nozzle_droplet_generator_1 1 to   droplet_splitter_1 2 channelWidth=400 ;
CHANNEL channel_2 from   droplet_splitter_1 1 to    nozzle_droplet_generator_2 2 channelWidth=400 ;
CHANNEL channel_3 from    nozzle_droplet_generator_2 1 to   diamond_chamber_1 2 channelWidth=400 ;
CHANNEL channel_4 from   diamond_chamber_1 1 to  picoinjector_1 2 channelWidth=400 ;

END LAYER