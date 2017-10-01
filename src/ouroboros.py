import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, PowerNorm
import math

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

def get_data_as_yearlist(start_year, end_year):
    '''
    take in data frame (north atlantic most likely) and turn it into a list of dataframes
    with each entry being a dataframe holding a year's data
    '''
    # load data frame that we want
    data_na = load_hurricane_data()

    # make a list for the years
    years = []

    print data_na.loc[:, "Season"].unique()

    # step through the Seasons (years) and make a new dataframe for each one
    for year in data_na.loc[:, "Season"].unique():
        temp = data_na[data_na.loc[:, "Season"] == year]
        years.append(temp)

    # get rid of a nan DataFrame
    years.pop(0)

    #loop through years in future gif
    start = 164 - (2016 - start_year)
    end = 164 - (2017 - end_year)

    if start != end:
        temp = years[start:end]
    else:
        # handle case where only one year is wanted
        temp = []
        temp.append(years[start])

    # try to give back memory
    del years

    return temp

def load_hurricane_data(_path="../data/allstorms.csv"):
    data = pd.read_csv(_path)

    # data dictionary
    # N/A,Year,#,BB,BB,N/A,YYYY-MM-DD HH:MM:SS,N/A,deg_north,deg_east,kt,mb,N/A,%,%,N/A

#     print data.loc[:,"Basin"].unique()
    # array(['BB', ' SI', ' NA', ' EP', ' SP', ' WP', ' NI', ' SA'], dtype=object)

    data.loc[:, "Basin"] = data.loc[:, "Basin"].apply(lambda x: x.replace(" ", ""))

#     print data.loc[:,"Basin"].unique()
    # ['BB' 'SI' 'NA' 'EP' 'SP' 'WP' 'NI' 'SA']

    # since we're only looking at North Atlantic in this case
    data_na = data[data.loc[:, "Basin"] == "NA"]

    data_na.loc[:,"Season"] = data_na.loc[:,"Season"].apply(lambda x: int(x))

    # convert wind speed stuff into storm category information
    data_na.loc[:,"saffir_simpson_cat"] = data_na["Wind(WMO)"].apply(lambda x: safsimpsonize(x))

    # try to give back memory
    del data

    return data_na

def get_storms_from_year(year_df):
    '''
    year_df is the dataframe with a year's data

    returns a list of smaller dataframes each consisting of a
    unique storm track
    '''
    storms = []

    # step through the year and make a dataframe for each storm
    for storm in year_df.loc[:,"Serial_Num"].unique():

        # make a temp storm since it's needed more than once
        temp_storm = year_df[year_df.loc[:, "Serial_Num"] == storm]
        if temp_storm.count()[0] >= 5:
            storms.append(temp_storm)

        del temp_storm

    return storms

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


def make_storm_data(list_of_storms):
    '''
    In the old way storm paths were interpolated for every frame.  This is really wasteful.
    Instead make a list of 2d numpy arrays.  Each numpy array is shape (nobs, 2) where the
    columns are lat, and long

    Returns a list of 2d numpy arrays
    '''
    # data is in 6 hour intervals, interpolate to 10 minute time slices
    interpolation_multiplier = 4 * 6 # 10min per time slice

    # where the processed tracks will go
    storm_tracks = []

    for storm in list_of_storms:
        # make sure that all of the numbers are in a float format
        # otherwise weird stuff can happen later (comparing float to str for example)
        # since long is a variable name use _ to prevent use of reserved type
        _lat = np.array(storm.Latitude.apply(lambda x: float(x)))
        _long = np.array(storm.Longitude.apply(lambda x: float(x)))

        # how long is the new list going to be?
        new_length = interpolation_multiplier * len(_lat)

        # to help guide the interpolate method
        x = np.arange(len(_lat))

        # figure out length of new arrays
        new_x = np.linspace(x.min(), x.max(), new_length)

        # actually do the interpolation
        # these np arrays should all be the same length
        new_lat = interpolate.interp1d(x, _lat, kind='cubic')(new_x)
        new_long = interpolate.interp1d(x, _long, kind='cubic')(new_x)

        #############
        ### NOTE: ###
        #############

        # considered removing duplicate entries (probably a significant number)
        # but rejected because the visual velocity of the storms would be lost
        # in other words a storm that, for a period of time, was moving slowly
        # would probably have a lot of redundant lat/long entries.  But later
        # it might start moving more quickly.  This would be indicated by the
        # visual length of the trail. A short tail is a relatively slow movement
        # a long trail is relatively fast movement.  Removing duplicates would
        # make velocities appear constant, which would be undesirable (most likely)
#         if np.sum(np.isclose([5.0, 6.02], boink[-1,:], atol=0.01)) == 2:
#             print "True"
        temp = np.column_stack([new_lat, new_long])

        storm_tracks.append(temp)

        del temp

    return storm_tracks

def make_initial_railyard(storm_list, grid_scale):
    '''
    The end result of the comet animation is going to be a grid that is overlayed
    on top of a map of the SE United States and Caribbean.  To initialize that grid
    we need to know the scale of the grid. 1.0 is 1* lat/long. 10x scale means each
    grid is 0.1*

    The current vision for the animation is a dark railyard of all of the inactive
    storm tracks.  Then 1/10th (or whatever looks good) of the storms are activated
    and trace their path on the rail yard. As some complete others start.

    In any case, we need to establish the original grid that will store the brightness
    of the tracks and will serve as the original source of truth for the grid.

    Later when the animation is being rendered we will figure out only what parts
    of the tracks are active and only update those parts of the map.  This will
    be computationally far less intensive than the old "figure everything out" each
    frame manner.

    Returns:
    2d numpy array representing each possible cell in the grid. At 1.0 scale that would
    be something like 80* longitude wide and 40* latitude tall
    '''
    baseline = 1.0 # this is the minimal dark color that will represent an inactive track

    # 40 is the total latitude lines, 80 is the total longitude lines
    temp_grid = np.zeros((int(40 * grid_scale), int(80 * grid_scale)), dtype=np.int16)

    # for every storm that we have data for...
    for storm in storm_list:

        length = len(storm) # determine the length of the storm:

        # step through each set of coordinates in the storm numpy matrix
        # column 0 = lat
        # column 1 = long
        for idx in range(length):
            if 10 <= storm[idx, 0] <= 50 and -110 <= storm[idx, 1] <= -30:
                # figure out the X/Y coordinates for the current lat/long location
                _Y = int(50 * grid_scale) - 2 - int(storm[idx, 0] * grid_scale)
                # was a -1 adjustment because counting starts at 0, changing to -2 because
                # storms tend to start on the right side and not hit the left side
                # so this is a hack-ey fix for what seems like a rounding problem
                _X = int(110 * grid_scale) - 2 + int(storm[idx, 1] * grid_scale)

                # since we're initializing we know that the strength will
                # always be the baseline
                temp_grid[_Y, _X] = baseline

    return temp_grid

def return_activated_storm_area(storm_slice, temp_grid, grid_scale, after_glow=0):
    '''
    Activate the specific slice of the storm provided and activate the provided
    grid as necessary

    Optionally take in the after_glow argument to fade out leftovers
    '''

    _len = len(storm_slice) # we need to know how long to make the linspace

    # create the highlighted meteor part
    active = np.linspace(0, 255, _len).astype(np.int16)

    for idx in range(_len):
        if 10 <= storm_slice[idx, 0] <= 50 and -110 <= storm_slice[idx, 1] <= -30:
            # figure out the X/Y coordinates for the current lat/long location
            _Y = int(50 * grid_scale) - 2 - int(storm_slice[idx, 0] * grid_scale)
            # was a -1 adjustment because counting starts at 0, changing to -2 because
            # storms tend to start on the right side and not hit the left side
            # so this is a hack-ey fix for what seems like a rounding problem
            _X = int(110 * grid_scale) - 2 + int(storm_slice[idx, 1] * grid_scale)

            power = active[idx]


            if temp_grid[_Y, _X] < power:   # only replace if the new strength is higher
                temp_grid[_Y, _X] = power

                if power == 255:
                    temp_grid = fuzz_x(temp_grid, _X, _Y, 1, 180)
                    temp_grid = fuzz_x(temp_grid, _X, _Y, 2, 90)
                    # temp_grid[_Y + 1, _X + 1] = 90
                    # temp_grid[_Y + 1, _X + -1] = 90
                    # temp_grid[_Y + -1, _X + 1] = 90
                    # temp_grid[_Y + -1, _X + -1] = 90

                    temp_grid = fuzz_plus(temp_grid, _X, _Y, 1, 220)
                    temp_grid = fuzz_plus(temp_grid, _X, _Y, 2, 130)

                    # temp_grid[_Y + 1, _X] = 220
                    # temp_grid[_Y + -1, _X] = 220
                    # temp_grid[_Y, _X + 1] = 220
                    # temp_grid[_Y, _X -1] = 220

                if power > 32:
                    temp_grid = fuzz_x(temp_grid, _X, _Y, 1, int(power/2.0))
                    temp_grid = fuzz_plus(temp_grid, _X, _Y, 1, int(power/2.0))
                    temp_grid = fuzz_x(temp_grid, _X, _Y, 2, int(power/4.0))
                    temp_grid = fuzz_plus(temp_grid, _X, _Y, 2, int(power/4.0))

    return temp_grid

def fuzz_x(grid, x, y, fd, _pow):
    '''
    edit passed in grid at x, y with pow
    '''
    if 0 <= y + fd < grid.shape[0] and 0 <= x + fd < grid.shape[1] and 0 <= y - fd < grid.shape[0] and 0 <= x - fd < grid.shape[1]:
        if grid[y + fd, x + fd] < _pow:
            grid[y + fd, x + fd] = _pow

        if grid[y + fd, x - fd] < _pow:
            grid[y + fd, x - fd] = _pow

        if grid[y - fd, x + fd] < _pow:
            grid[y - fd, x + fd] = _pow

        if grid[y - fd, x - fd] < _pow:
            grid[y - fd, x - fd] = _pow

    return grid

def fuzz_plus(grid, x, y, fd, _pow):
    '''
    edit passed in grid at x, y with pow
    '''
    if 0 <= y + fd < grid.shape[0] and 0 <= x + fd < grid.shape[1] and 0 <= y - fd < grid.shape[0] and 0 <= x - fd < grid.shape[1]:
        if grid[y + fd, x ] < _pow:
            grid[y + fd, x ] = _pow

        if grid[y - fd, x ] < _pow:
            grid[y - fd, x ] = _pow

        if grid[y, x + fd] < _pow:
            grid[y, x + fd] = _pow

        if grid[y, x - fd] < _pow:
            grid[y, x - fd] = _pow

    return grid

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

    # fig.savefig("../imgs/test/ouroboros/debug.png", pad_inches=0, transparent=True)

    final_buffer = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)

    # print "max create_map_buffer:", final_buffer[...,1].max()
    # print "create_map_buffer.shape", final_buffer.shape

    plt.close("all")

    return final_buffer, desc


def run_shockwave():
    '''
    Effectively a main() function that contains all of the execution steps
    '''
    book_of_storms = get_map_data()

    storm_data = make_storm_data(book_of_storms)

    #############################
    ### establish video stuff ###
    #############################

    vid_fps = 30

    _file = "../imgs/test/ouroboros/ouroboros16.mp4"
    ffm = FFMWrapper(_file, _vid_fps=vid_fps)

    # #video width, height:
    # w, h = 1920, 1080

    num_storms = len(storm_data)  # seems like 1314

    grid_scale = 20

    INTERVALS = 87.72 * 0.1 # arithmetically this seems to give each storm 5 seconds worth of frames
    # INTERVALS = 87.72 * 1.0 # arithmetically this seems to give each storm 5 seconds worth of frames

    # establish the inital grid and source of truth:
    core_grid = make_initial_railyard(storm_data, grid_scale)

    # black_buffer = np.zeros(shape=(h, w, 4), dtype=np.int16)
    # old_buffer = np.zeros(shape=(h, w, 4), dtype=np.int16)

    # worm_length = 14.933 # how many storms are active at any given time
    worm_length = num_storms / INTERVALS

    int_worm_length = int(worm_length) + 1 # because we'll have to do a for loop with integers

    worm_head = 17.0 # index that represents the current starting point of the worm

    # num_intervals = num_storms / (worm_length * 1.0) # storms / storms per interval ie worm_length

    # incremental step forward is 1 / (num_frames / num_intervals) * frame
    step = 1.0 / (1710 / INTERVALS)

    for frame in range(1710): # 1710 is 55 seconds at 30 frames/sec

        # if frame < 1150:
        #     break

        frame_grid = core_grid.copy() # make it a copy so the original is not modified

        worm_head = 17 + step * frame # 87.72 is the max. 17.0 so that we don't start at 0 and make the start obvious

        if worm_head > INTERVALS:  # wrap around so that the full range is covered, since we're starting at 17
            worm_head -= INTERVALS

    #     print worm_head, worm_tail

        magic_mult = 9.0

        # starting_storm = worm_head * worm_length
        starting_storm = worm_head * worm_length
        starting_storm_index = int(starting_storm) + 1
        trailing_edge = -1 * (int(starting_storm) - (starting_storm)) # figure out the lagging remainder to activate

    #     print "Raw Storm: {:3.2f} Index: {:2d} T.E.: {:2.2f}".format(starting_storm, starting_storm_index, trailing_edge)
        print "Frame: ", frame

        # do something a bit different if worm_head < 17 because need to do the trailing part too

        # if the current index is higher than the end of the storm_data array
        # then move the pointer to the start of the storm_data array
        if starting_storm_index >= num_storms:
            starting_storm_index -= num_storms

        # for segment in range(int(int_worm_length * magic_mult -1)):
        for segment in range(int(int_worm_length * magic_mult)):

            active_index = starting_storm_index - segment # first one is the front of the worm

            # because we can wrap around it is possible that active_index is negative
            # which means that it really needs to be back at the end of the list
            # is fixing the overloop this easy?
            if active_index < 0:
                # since it's below 0 add active_index to the length instead of
                # subtracting
                active_index = num_storms + active_index

            # if active_index >= len(storm_data):
            #     active_index -= len(storm_data)
            #     # print "weird, not expected. active_index is > than len(storm_data)"

            # the sneaky bug was probably because active index is positive
            # when each time segment increases active_index should be smaller
            _len = len(storm_data[active_index])

            _slice = int(1.0/ worm_length * _len * magic_mult)

            start_slice = int(_slice * (segment / magic_mult))
            end_slice = start_slice + _slice

#             print "len: {:4d} start: {:4d} end: {:4d}".format(_len, start_slice, end_slice)

            frame_grid = return_activated_storm_area(storm_data[active_index][start_slice:end_slice,:],
                                                     frame_grid, grid_scale)


        # ideally come back and light the tiny trailing edge with a bit of afterglow
        # not the whole slice but like the trailing_edge * slice_length of the active area

        # create map buffer
        # for cmap 0: gam 1.5/2.5 is good
        # cmap 0 actually looks really cool with comet tails shading
        # for cmap 7: copper seems good at 1.5/3.25
        new_buffer, _ = create_map_buffer(frame_grid, gam1=1.5, gam2=3.25, color_map=7)

#         if frame == 1710:
#             ffm.cross_fade_frames(old_buffer, new_buffer, int(vid_fps * 0.5), "")
#         else:
#             ffm.cross_fade_frames(old_buffer, new_buffer, 2, "")

#         old_buffer = new_buffer.copy()

        # seems like bugging out in the FFMWrapper Class
        ffm.add_frame(new_buffer)

    ffm.close()

if __name__ == "__main__":
    run_shockwave()
