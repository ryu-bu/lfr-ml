DEVICE ml0

LAYER FLOW

DROPLET SPLITTER droplet_splitter_1 ;
MIXER mixer_1 ;
DROPLET SPLITTER droplet_splitter_2 ;
PICOINJECTOR picoinjector_1 ;
MIXER mixer_2 ;


CHANNEL channel_1 from   droplet_splitter_1 1 to  mixer_1 2 channelWidth=400 ;
CHANNEL channel_2 from  mixer_1 1 to   droplet_splitter_2 2 channelWidth=400 ;
CHANNEL channel_3 from   droplet_splitter_2 1 to  picoinjector_1 2 channelWidth=400 ;
CHANNEL channel_4 from  picoinjector_1 1 to  mixer_2 2 channelWidth=400 ;

END LAYER