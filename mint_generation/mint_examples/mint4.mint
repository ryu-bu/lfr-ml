DEVICE ml4

LAYER FLOW

DIAMOND CHAMBER diamond_chamber_1 ;
DROPLET SORTER droplet_sorter_1 ;
PICOINJECTOR picoinjector_1 ;
MIXER mixer_1 ;
MIXER mixer_2 ;


CHANNEL channel_1 from   diamond_chamber_1 1 to   droplet_sorter_1 2 channelWidth=400 ;
CHANNEL channel_2 from   droplet_sorter_1 1 to  picoinjector_1 2 channelWidth=400 ;
CHANNEL channel_3 from  picoinjector_1 1 to  mixer_1 2 channelWidth=400 ;
CHANNEL channel_4 from  mixer_1 1 to  mixer_2 2 channelWidth=400 ;

END LAYER