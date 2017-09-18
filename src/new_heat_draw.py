# python 2.7
# Draw heatmaps using the new DataStorm class
# Brian Hardenstein 2017

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, PowerNorm

from scipy import interpolate
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.misc import imread

# for writing directly to video
import subprocess

from DataStorm import DataStorm

def main():
    storm = DataStorm(1915, 1926, vid_fps=60, smoothing=60) # not inclusive of the second date!!!


    storm.draw_video()

if __name__ == "__main__":
    main()
