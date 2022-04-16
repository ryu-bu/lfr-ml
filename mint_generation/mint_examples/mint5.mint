DEVICE ml5

LAYER FLOW

NOZZLE DROPLET GENERATOR nozzle_droplet_generator_1 ;
NOZZLE DROPLET GENERATOR nozzle_droplet_generator_2 ;
MIXER mixer_1 ;
NOZZLE DROPLET GENERATOR nozzle_droplet_generator_3 ;
PICOINJECTOR picoinjector_1 ;


CHANNEL channel_1 from    nozzle_droplet_generator_1 1 to    nozzle_droplet_generator_2 2 channelWidth=400 ;
CHANNEL channel_2 from    nozzle_droplet_generator_2 1 to  mixer_1 2 channelWidth=400 ;
CHANNEL channel_3 from  mixer_1 1 to    nozzle_droplet_generator_3 2 channelWidth=400 ;
CHANNEL channel_4 from    nozzle_droplet_generator_3 1 to  picoinjector_1 2 channelWidth=400 ;

END LAYER