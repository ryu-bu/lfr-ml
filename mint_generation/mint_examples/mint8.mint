DEVICE ml8

LAYER FLOW

DROPLET SPLITTER droplet_splitter_1 ;
DROPLET SPLITTER droplet_splitter_2 ;
DIAMOND CHAMBER diamond_chamber_1 ;
MIXER mixer_1 ;
DIAMOND CHAMBER diamond_chamber_2 ;


CHANNEL channel_1 from   droplet_splitter_1 1 to   droplet_splitter_2 2 channelWidth=400 ;
CHANNEL channel_2 from   droplet_splitter_2 1 to   diamond_chamber_1 2 channelWidth=400 ;
CHANNEL channel_3 from   diamond_chamber_1 1 to  mixer_1 2 channelWidth=400 ;
CHANNEL channel_4 from  mixer_1 1 to   diamond_chamber_2 2 channelWidth=400 ;

END LAYER