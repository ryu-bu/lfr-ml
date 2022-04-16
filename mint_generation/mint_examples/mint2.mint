DEVICE ml2

LAYER FLOW

PICOINJECTOR picoinjector_1 ;
DIAMOND CHAMBER diamond_chamber_1 ;
DROPLET SPLITTER droplet_splitter_1 ;
DROPLET SORTER droplet_sorter_1 ;
DROPLET SPLITTER droplet_splitter_2 ;


CHANNEL channel_1 from  picoinjector_1 1 to   diamond_chamber_1 2 channelWidth=400 ;
CHANNEL channel_2 from   diamond_chamber_1 1 to   droplet_splitter_1 2 channelWidth=400 ;
CHANNEL channel_3 from   droplet_splitter_1 1 to   droplet_sorter_1 2 channelWidth=400 ;
CHANNEL channel_4 from   droplet_sorter_1 1 to   droplet_splitter_2 2 channelWidth=400 ;

END LAYER