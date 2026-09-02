"""
Author:  WilmerZhu
Email: zhuwm@mail.iggcas.ac.cn
Date: 2023-09-24 10:31:48
LastEditTime: 2023-10-08 11:28:56
FilePath: /Plot/geology_Alps.py

"""

###############################################################################
# Description:
# This is for plotting the tectonic/geology fig in the Alps
# data from USGS, Map Showing Geology, Oil and Gas Fields, and Geologic Provinces of Europe including Turkey
###############################################################################
import pandas as pd
import pygmt
import os

###############################################################################
# define parameters for plotting
pygmt.config(
    MAP_FRAME_TYPE="plain",
    MAP_GRID_PEN_PRIMARY="0.3p,dimgrey",
    MAP_ANNOT_OBLIQUE="30",
    MAP_ANNOT_OFFSET_PRIMARY="5p",
    MAP_ANNOT_OFFSET_SECONDARY="5p",
    FONT_ANNOT_PRIMARY="10p,5",
    FONT_LABEL="10p,28,black",
    MAP_FRAME_WIDTH="2p",
    MAP_FRAME_PEN="0.5p",
    MAP_TICK_LENGTH_PRIMARY="5p",
    MAP_LABEL_OFFSET="5.5p",
)
# define the region for plotting
region_Alps = [4, 19, 38, 49]

# dealing with the tectonic data, from .gmt to .dat
os.system(
    "gmt convert data_Alps/geology/geo4_2l.gmt -aZ='GLG' > data_Alps/geology/Alps/geo4_2l.dat"
)

#########################################################################
fig = pygmt.Figure()
fig.basemap(region=region_Alps, projection="M15c", frame=["af", f"WsNe"])
# coastlines
fig.coast(
    resolution="f",
    area_thresh="100",
    shorelines="0.5p,black",
)
# plot the geology map
fig.plot(
    data="data_Alps/geology/Alps/geo4_2l.dat",
    region=region_Alps,
    fill="+z",
    projection="M15c",
    cmap="data_Alps/geology/Alps/simplified_Alps_GLG.cpt",
)
# add the legend
fig.legend(
    spec="data_Alps/geology/Alps/age_legend.txt",
    position="JBR+w15c/6c+jTR+o0c/0c+l1.5",
    box="+p0.7p+g255",
)
#########################################################################
# save the figure
fig.savefig("fig_Alps/Alps_geology_py.pdf", dpi=720, crop=True, show=True)
