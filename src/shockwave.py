import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, PowerNorm

# %matplotlib inline

from scipy import interpolate
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.misc import imread

import time

from FFMWrapper import FFMWrapper

def get_map_data(_start=1915, _end=2016):
    # establish the range of years
    # ^ moved to function parameter

    # make a list of year dataframes for the range
    years = get_data_as_yearlist(_start, _end)

#     print grid.shape

    tome_of_storms = []

    # make a temp list to hold the storm dataframes from a single year
    for _idx, year in enumerate(years):

        storms = get_storms_from_year(year)

        tome_of_storms += storms

    return tome_of_storms

def process_storms(_tome_of_storms, current_frame, num_frames):
    '''
    takes in a list of ALL storms in the time range and then process the data
    so that we have the current proportion of frames

    Return this data as a grid
    '''
    # sort of a constant but might need to be adjusted
    # best range seems to be 3.0 - 7.0 but up to 10 is still good
    # grid_scale = 0.50
    grid_scale = 13.0

    # 40 is the total latitude lines, 80 is the total longitude lines
    grid = np.zeros((int(40 * grid_scale), int(80 * grid_scale)), dtype=np.float64)

    for storm in _tome_of_storms:

        # total amount of storm frames to use...
        ratio = current_frame / (num_frames * 1.0)
        grid += storm_heat(storm, grid_scale, ratio)

    print "max of grid:", grid[...].max()

    grid = np.clip(grid, 0, 255)

    return grid

def storm_heat(storm, grid_scale=7.0, ratio=1.0):
    '''
    make storm track into two lists that will be turned into a heatmap
    '''
    # 40 is the total latitude lines, 80 is the total longitude lines
    temp_grid = np.zeros((int(40 * grid_scale), int(80 * grid_scale)), dtype=np.int16)

    if len(storm) >= 5:
        # interpolate the storm tracks like crazy
        interpolation_multiplier = 24 * 6 # 10 min per tick now

        # get the data out of the data frame in a usable form
        _lat = np.array(storm.Latitude.apply(lambda x: float(x)))
        _long = np.array(storm.Longitude.apply(lambda x: float(x)))
        # since the 'saffir_simpson_cat' column ranges from -1 to 5 add 2 to it
        # so that the range is 1 <-> 7 which a computer understands better
        # _strengths = np.array(storm.loc[:, "saffir_simpson_cat"].apply(lambda x: int((x + 2)**2)))

        # clean_lat = []
        # clean_long = []
        # # clean_storms = []
        # # before we expand by a large magnitude the size of the lists lets dump
        # # the values that aren't in the picture anyways
        # for idx in range(len(_lat)):
        #     if 10 <= _lat[idx] <= 50 and -110 <= _long[idx] <= -30:
        #         clean_lat.append(_lat[idx])
        #         clean_long.append(_long[idx])
        #         # clean_storms.append(_strengths[idx])
        #
        # # dump the memory for the obsolete lists
        # del _lat, _long, # _strengths
        #
        # _lat = np.array(clean_lat)
        # _long = np.array(clean_long)
        # # _strengths = np.array(clean_storms)
        #
        # del clean_lat, clean_long, # clean_storms

#         _strengths = storm.loc[:,"Wind(WMO)"].apply(lambda x: int(((x / 165.0) * 5.0)**2))

        # how long is the new list going to be?
        new_length = interpolation_multiplier * len(_lat)

        # some storm tracks might never appear on our screen so length of
        # _lat or _long might be zero, if so then break for this hurricane
        if len(_lat) > 5 and len(_long) > 5:


            # to help guide the interpolate method
            x = np.arange(len(_lat))

            # figure out length of new arrays
            new_x = np.linspace(x.min(), x.max(), new_length)

            # actually do the interpolation
            # these np arrays should all be the same length
            new_lat = interpolate.interp1d(x, _lat, kind='cubic')(new_x)
            new_long = interpolate.interp1d(x, _long, kind='cubic')(new_x)
            # new_strs = interpolate.interp1d(x, _strengths, kind='cubic')(new_x)

            # get the right ratio slice of the new list and add it to the grid:
            # figure out slice, it should be the same for all
            _slice = int(new_length * ratio)
            # new_lat = new_lat[:_slice]
            # new_long = new_long[:_slice]

            active_slice = int(new_length / 33.0)

            new_strs = np.zeros(shape=(new_length, 1), dtype=np.int16)

            # add the baseline value to the trail to make a sort of 'rail' for
            # the light to ride on. This will help enable the looping?
            new_strs += 2

            # create the highlighted meteor part
            active = np.linspace(2, 255, active_slice).astype(np.int16).reshape(active_slice, 1)

            # now replace the existing new_strs with the active value using
            # _slice as the location to add
            if _slice > active_slice:
                new_strs[_slice-active_slice:_slice] = active

            # if limit_slice < _slice:
            #     bright = list(np.linspace(2, 255, active_slice))
            #     dim = [2 for idx in range(_slice - limit_slice)]
            #     dim += bright
            #
            #     new_strs = np.array(dim)
            #
            # else:
            #     new_strs = np.linspace(2, 255, _slice)

            for idx in range(new_length-1):
                if 10 <= new_lat[idx] <= 50 and -110 <= new_long[idx] <= -30:
                    # figure out the X/Y coordinates for the current lat/long location
                    _Y = int(50 * grid_scale) - 2 - int(new_lat[idx] * grid_scale)
                    # was a -1 adjustment because counting starts at 0, changing to -2 because
                    # storms tend to start on the right side and not hit the left side
                    # so this is a hack-ey fix for what seems like a rounding problem
                    _X = int(110 * grid_scale) - 2 + int(new_long[idx] * grid_scale)

                    # if temp_grid[_Y, _X] < new_strs[idx]:
                    #     # print "new strs idx:", new_strs.shape
                    temp_grid[_Y, _X] = new_strs[idx]

    # keep the return at base level so even if it's a very short storm track
    # return something, otherwise you return None and things get unhappy
    return temp_grid

def create_map_buffer(grid, gam1=1.0, gam2=2.0, file_name=None, color_map=0, year=0):
    '''
    Render a frame of a heatmap from the grid provided then return it as a buffer
    in order to be processed and added to a video
    '''

    color_dict = {0:"inferno", 1:"plasma", 2:"magma", 3:"viridis", 4:"hot", 5:"afmhot",
                 6:"gist_heat", 7:"copper", 8:"bone", 9:"gnuplot", 10:"gnuplot2",
                 11:"CMRmap", 12:"pink", 13:"spring", 14:"autumn", 15:"cool_r",
                 16:"Wistia_r", 17:"seismic", 18:"RdGy_r", 19:"BrBG_r", 20:"RdYlGn_r",
                 21:"PuOr", 22:"brg", 23:"hsv", 24:"cubehelix", 25:"gist_earth",
                 26:"ocean", 27:"gist_stern", 28:"gist_rainbow_r", 29:"jet",
                 30:"nipy_spectral", 31:"gist_ncar"}

    _cmap = color_dict[color_map]

    # establish the figure
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

    ax = fig.add_subplot(111)
    ax.clear()  # maybe clear erases some of the axis settings, not just the canvas?
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_facecolor("#000000")
    ax.set_xlim(-110.0, -30.0)
    ax.set_ylim(10, 50.0)

    # make a high peg for the grid to normalize through the years
    # for grid size 0.5 97500 was the max
    # grid[-1, -1] = 6000

    w, h = fig.canvas.get_width_height()
    _heatmap = np.zeros(shape=(w, h, 4))

    # ax.imshow(grid, cmap=_cmap, extent=[-110, -30, 10, 50], alpha=1.0, aspect="auto")
    ax.imshow(grid, norm=PowerNorm(gamma=gam1/gam2), cmap=_cmap, extent=[-110, -30, 10, 50], alpha=1.0, aspect="auto")

    # paint the canvas
    fig.canvas.draw()

    # pull the paint back off the canvas into the buffer
    heat_img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)

    # print "max heat buffer:", heat_img[...,2].max()

    # and black parts transparent (if they're not already)
    heat_img[((heat_img[...,0] <= 5) & (heat_img[...,1] <= 5) & (heat_img[...,2] <= 5))] = 0

    # fix the super hot square from normalizing throughout the years
    # heat_img[((heat_img[...,0] == 255) & (heat_img[...,1] == 255) & (heat_img[...,2] == 255))] = 0

    # heat_img[...,0] = heat_img[..., 0] / 10.0
    # heat_img[...,1] = heat_img[..., 1] / 10.0
    # heat_img[...,2] = heat_img[..., 2] / 10.0

    map_image = imread("../data/ultra_map.png")

    ax.imshow(map_image, extent=[-110, -30, 10, 50], aspect="auto", alpha=1.0)

    fig.canvas.draw()

    map_buffer = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)

    final_buffer = map_buffer + heat_img

    final_buffer = np.clip(final_buffer, 0, 255) # clip buffer back into int8 range
                    # wonder if some kind of exp transform
                    # might enable hdr-like effect

    ax.imshow(final_buffer.astype(np.uint8), extent=[-110, -30, 10, 50], aspect="auto", alpha=1.0)
    # write out file info so we can see how a map was made later w/ grid search
    # desc = "grid_scale: {:2.2f}\n gam1: {:2.2f}\n gam2: {:2.2f}\n _cmap: {}".format(7.0, gam1, gam2, _cmap)
    if year != 0:
        desc = str(year)
    else:
        desc = ""
    # ax.annotate(desc, xy=(-109, 48), size=40, color='#AAAAAA')
    ax.annotate("@pixelated_brian", xy=(-109, 12), size=20, color="#BBBBBB")

    fig.canvas.draw()
    final_buffer = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)

    print "max create_map_buffer:", final_buffer[...,1].max()
    print "create_map_buffer.shape", final_buffer.shape

    plt.close("all")

    return final_buffer, desc


def draw_map(grid, gam1=1.0, gam2=2.0, file_name=None, color_map=0, year=0):
    '''
    Save a png file as a map that represents the heatmap of the grid provided
    '''

    color_dict = {0:"inferno", 1:"plasma", 2:"magma", 3:"viridis", 4:"hot", 5:"afmhot",
                 6:"gist_heat", 7:"copper", 8:"bone", 9:"gnuplot", 10:"gnuplot2",
                 11:"CMRmap", 12:"pink", 13:"spring", 14:"autumn", 15:"cool_r",
                 16:"Wistia_r", 17:"seismic", 18:"RdGy_r", 19:"BrBG_r", 20:"RdYlGn_r",
                 21:"PuOr", 22:"brg", 23:"hsv", 24:"cubehelix", 25:"gist_earth",
                 26:"ocean", 27:"gist_stern", 28:"gist_rainbow_r", 29:"jet",
                 30:"nipy_spectral", 31:"gist_ncar"}

    _cmap = color_dict[color_map]

    # establish the figure
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

    ax = fig.add_subplot(111)
    ax.clear()  # maybe clear erases some of the axis settings, not just the canvas?
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_facecolor("#000000")
    ax.set_xlim(-110.0, -30.0)
    ax.set_ylim(10, 50.0)

    # make a high peg for the grid to normalize through the years
    # for grid size 0.5 97500 was the max
    # grid[-1, -1] = 100000

    w, h = fig.canvas.get_width_height()
    _heatmap = np.zeros(shape=(w, h, 4))

    ax.imshow(grid, norm=PowerNorm(gamma=gam1/gam2), cmap=_cmap, extent=[-110, -30, 10, 50], alpha=1.0, aspect="auto")

    # paint the canvas
    fig.canvas.draw()

    # pull the paint back off the canvas into the buffer
    heat_img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)

    # print "max heat buffer:", heat_img[...,2].max()

    # and black parts transparent (if they're not already)
    heat_img[((heat_img[...,0] <= 5) & (heat_img[...,1] <= 5) & (heat_img[...,2] <= 5))] = 0

    # fix the super hot square from normalizing throughout the years
    # heat_img[((heat_img[...,0] == 255) & (heat_img[...,1] == 255) & (heat_img[...,2] == 255))] = 0

    # heat_img[...,0] = heat_img[..., 0] / 10.0
    # heat_img[...,1] = heat_img[..., 1] / 10.0
    # heat_img[...,2] = heat_img[..., 2] / 10.0

    map_image = imread("../data/ultra_map.png")

    ax.imshow(map_image, extent=[-110, -30, 10, 50], aspect="auto", alpha=1.0)

    fig.canvas.draw()

    map_buffer = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)

    final_buffer = map_buffer + heat_img

    final_buffer = np.clip(final_buffer, 0, 255) # clip buffer back into int8 range
                    # wonder if some kind of exp transform
                    # might enable hdr-like effect

    ax.imshow(final_buffer.astype(np.uint8), extent=[-110, -30, 10, 50], aspect="auto", alpha=1.0)
    # write out file info so we can see how a map was made later w/ grid search
    # desc = "grid_scale: {:2.2f}\n gam1: {:2.2f}\n gam2: {:2.2f}\n _cmap: {}".format(7.0, gam1, gam2, _cmap)
    if year != 0:
        desc = str(year)
    else:
        desc = ""
    ax.annotate(desc, xy=(-109, 48), size=40, color='#AAAAAA')
    ax.annotate("@pixelated_brian", xy=(-109, 12), size=20, color="#BBBBBB")

    if file_name != None:
        fig.savefig("../imgs/test/firebur/{}".format(file_name), pad_inches=0, transparent=True)


def run_map_grid_search(grid, _start=0, _end=1000):
    '''
    Pick hyper parameters to use in a randomized grid search to try to
    find a good looking heatmap setup
    '''

    for rdx in range(_start, _end):
        print "Making map [{:0>4}]/1000\r".format(rdx),
        _filename = "cmap_gsearch_num{:0>4d}.png".format(rdx)

        # # pick grid_scale from lognormal distribution
        # # with these values min/max of 10,000 test sample is ~0.5, 21.4
        # mu, sigma = 1.2, 0.5
        # # returns a numpy array so steal the first value in that to actually get the value
        # _grid_scale = np.random.lognormal(mu, sigma, 1)[0]

        # gamma 1: pull from a normal distribution centered around 1.0
        norm_mu, norm_sigma = 1.5, 0.5

        _gam1 = np.random.normal(norm_mu, norm_sigma, 1)[0]

        # we don't want the value to go negative so if it is negative set it to 1.0
        if _gam1 <= 0:
            _gam1 = 1.0

        #, gamma 2, cmap? if so what cmap?
        g2_mu, g2_sigma = 3.0, 1.0

        do_over = 0

        _gam2 = -10

        # we want to keep picking until _gam2 is bigger than _gam1
        # but prevent infinite loops because that's really annoying
        while _gam2 < _gam1 or do_over < 1000:
            # break infinite loop
            do_over += 1
            _gam2 = np.random.normal(g2_mu, g2_sigma, 1)[0]

        # about 2/3rds the time use the non-standard color map
        _color_map = np.random.randint(0, 32)

    #     draw_map(grid_scale, x, y, _xs, _ys, gam1=1.0, gam2=2.0, file_name=None, color_map=0)
        draw_map(grid, _gam1, _gam2, color_map=_color_map, file_name=_filename)


def make_historical_diagram():
    '''
    do all of the stuff for the graphics of making a hurricane plot

    takes in the year dataframe in order to get the necessary data

    returns ax which is the figure axis that the current hurricane track will be added upon
    '''
    # establish the figure
    figure = plt.figure(figsize=(19.2, 10.80), dpi=100)

    axis = figure.add_subplot(111)
    axis.set_facecolor("#000000")

#     figure, axis = plt.subplots(figsize=(19.2,10.800), dpi=100)

#     data = np.linspace(165.0, 0, 10000).reshape(100,100)
# #     data = np.clip(randn(250, 250), -1, 1)

#     histo_image = axis.imshow(data, interpolation='nearest', cmap="inferno")

#     divider = make_axes_locatable(axis)

#     cax = divider.append_axes("right", size="2%", pad=0.05)

#     # Add colorbar, make sure to specify tick locations to match desired ticklabels
#     cbar = figure.colorbar(histo_image, ticks=[157, 130, 111, 96, 74, 39, 0], cax=cax)
#     cbar.ax.set_yticklabels(['5^', '4^', '3^', '2^', '1^', 'T.S.^', 'T.D.^'])  # vertically oriented colorbar
#     axis.set_title("North American Hurricane Tracks " + str(_year), size=20)
#     axis.set_xlabel("Longitude", size=16)
#     axis.set_ylabel("Latitude", size=16)
#     axis.set_facecolor("black")
    axis.set_xlim(-110.0, -30.0)
    axis.set_ylim(0, 50.0)

    return figure, axis

def heatmap(ax, storm):
    '''
    make a heatmap of storm track?
    '''
    x, y = storm_heat(storm)

    ax.hexbin(x, y, gridsize=50, bins="log", cmap="inferno")

    # ax.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')

def safsimpsonize(wind):
    '''
    Takes in wind speed

    Returns saffir-simpson hurricane category with 0 and -1 for tropical storm/depression
    which doesn't make perfect sense as scales go but this maintains categories but allows
    the model/visualization to detect when something is a tropical depression

    According to: https://en.wikipedia.org/wiki/Saffir%E2%80%93Simpson_scale

    Return:  Category:   Wind Speed Range: (mph, seriously?)
    5        Cat 5      157 <= v
    4        Cat 4      130 <= v < 157
    3        Cat 3      111 <= v < 130
    2        Cat 2      96 <= v < 111
    1        Cat 1      74 <= v < 96
    0        Storm      39 <= v < 74
    -1       Depression v < 39
    '''
    if wind >= 157:
        return 5
    elif wind >= 130:
        return 4
    elif wind >= 111:
        return 3
    elif wind >= 96:
        return 2
    elif wind >= 74:
        return 1
    elif wind >= 39:
        return 0
    else:
        return -1

def load_hurricane_data(_path="../data/allstorms.csv"):
    data = pd.read_csv(_path)

    # data dictionary
    # N/A,Year,#,BB,BB,N/A,YYYY-MM-DD HH:MM:SS,N/A,deg_north,deg_east,kt,mb,N/A,%,%,N/A

    data.loc[:,"Season"] = data.loc[:,"Season"].apply(lambda x: int(x))

#     print data.loc[:,"Basin"].unique()
    # array(['BB', ' SI', ' NA', ' EP', ' SP', ' WP', ' NI', ' SA'], dtype=object)

    data.loc[:, "Basin"] = data.loc[:, "Basin"].apply(lambda x: x.replace(" ", ""))

#     print data.loc[:,"Basin"].unique()
    # ['BB' 'SI' 'NA' 'EP' 'SP' 'WP' 'NI' 'SA']

    data_na = data[data.loc[:, "Basin"] == "NA"]

    data_na.loc[:,"saffir_simpson_cat"] = data_na["Wind(WMO)"].apply(lambda x: safsimpsonize(x))

    # try to give back memory
    del data

    return data_na

def get_data_as_yearlist(start_year, end_year):
    '''
    take in data frame (north atlantic most likely) and turn it into a list of dataframes
    with each entry being a dataframe holding a year's data
    '''
    # load data frame that we want
    data_na = load_hurricane_data()

    # make a list for the years
    years = []

    # step through the Seasons (years) and make a new dataframe for each one
    for year in data_na.loc[:, "Season"].unique():
        temp = data_na[data_na.loc[:, "Season"] == year]
        years.append(temp)

    # get rid of a nan DataFrame
    years.pop(0)

    #loop through years in future gif
    start = 164 - (2016 - start_year)
    end = 164 - (2016 - end_year)

    if start != end:
        temp = years[start:end]
    else:
        # handle case where only one year is wanted
        temp = []
        temp.append(years[start])

    # try to give back memory
    del years

    return temp

def get_storms_from_year(year_df):
    '''
    year_df is the dataframe with a year's data

    returns a list of smaller dataframes each consisting of a
    unique storm track
    '''
    storms = []

    # step through the year and make a dataframe for each storm
    for storm in year_df.loc[:,"Serial_Num"].unique():
        storms.append(year_df[year_df.loc[:, "Serial_Num"] == storm])

    return storms


def main():

    vid_fps = 30

    _file = "../imgs/test/more_heat/shockwave12.mp4"
    ffm = FFMWrapper(_file, _vid_fps=vid_fps)
    # grid = get_map_data()
    #
    # run_map_grid_search(grid, 0, 100)

    #video width, height:
    w, h = 1920, 1080

    # how many frames to fade in with
    interp = 2

    # instead of doing a random grid search do a heatmap with a ten year moving
    # average
    black_buffer = np.zeros(shape=(h, w, 4), dtype=np.int16)
    old_buffer = np.zeros(shape=(h, w, 4), dtype=np.int16)

    # gets a list of all of the storms in the provided time frame
    tome_of_storms = get_map_data(1915, 2016)

    # frames per second is usually 30 (well, 24 but 30 in our case)
    # 60s max video time, 5s interval/slack time, 55s * 30
    # 1650 frames aka intervals
    num_frames = 90

    for idy in range(1, num_frames + 1):
        grid = process_storms(tome_of_storms, idy, num_frames)

        # reverse gamma for giant grid size
        # new_buffer, desc = create_map_buffer(grid, gam1=2.0, gam2=2.25, color_map=5, year=year)
        # cmap 7, scale 8.5, gam 1.5/2.75 was pretty good (shockwave08)
        new_buffer, desc = create_map_buffer(grid, gam1=1.65, gam2=2.25, color_map=5)
        # draw_map(grid, gam1=2.25, gam2=5.5, file_name=_filename, color_map=5, year=year)

        if idy == 1:

            ffm.cross_fade(old_buffer, new_buffer, int(vid_fps * 0.5), desc)
        else:
            ffm.cross_fade(old_buffer, new_buffer, interp, desc)

        # use copy to get a distinct copy that won't update each time new_buffer
        # is modified
        old_buffer = new_buffer.copy()

    # fade video out
    ffm.cross_fade(old_buffer, old_buffer, vid_fps * 2, desc)

    # fade video out
    ffm.cross_fade(old_buffer, black_buffer, int(vid_fps * 0.5), "")

    ffm.close()

    print "\n\n\nfinished writing file: {}".format(_file)

if __name__ == "__main__":
    main()
