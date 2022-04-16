DEVICE ml1

LAYER FLOW

DIAMOND CHAMBER diamond_chamber_1 ;
DIAMOND CHAMBER diamond_chamber_2 ;
DROPLET SORTER droplet_sorter_1 ;
DROPLET SORTER droplet_sorter_2 ;
DROPLET SPLITTER droplet_splitter_1 ;


CHANNEL channel_1 from   diamond_chamber_1 1 to   diamond_chamber_2 2 channelWidth=400 ;
CHANNEL channel_2 from   diamond_chamber_2 1 to   droplet_sorter_1 2 channelWidth=400 ;
CHANNEL channel_3 from   droplet_sorter_1 1 to   droplet_sorter_2 2 channelWidth=400 ;
CHANNEL channel_4 from   droplet_sorter_2 1 to   droplet_splitter_1 2 channelWidth=400 ;

END LAYER