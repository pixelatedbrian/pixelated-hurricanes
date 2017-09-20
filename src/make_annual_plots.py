import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# %matplotlib inline

from scipy import interpolate
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.misc import imread

# for writing directly to video
import subprocess

# ffmpeg -r 30 -f image2 -s 1920x1080 -i xplot_1916_%2d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p 1916.mp4
# http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
# https://trac.ffmpeg.org/wiki/Concatenate


def main():
    map_image = imread("../data/final_ultra_map_reg.png")
    map_key = imread("../data/final_ultra_map_key_reg.png")

    START_OF_RUN = False

    _start = 1915
    _end = 2016

    lat_range = (10, 50)
    long_range = (-110, -30)

    vid_fps = 30
    # how much of a second should the video pause before going to the next year
    lag_time = 0.33

    # Open an ffmpeg process
    outf = '../imgs/test/final_hr/swarm_w_lag2.mp4'
    cmdstring = ('ffmpeg',
        '-y', '-r', str(vid_fps), # overwrite, 30fps
        '-s', '{:d}x{:d}'.format(1920, 1080), # size of image string
        '-pix_fmt', 'argb', # format
        '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
        '-vcodec', 'libx264', '-crf', '25', outf) # output encoding

    # what we've been using when doing it manually
    # ffmpeg -r 24 -f image2 -s 1920x1080 -i xplot_2012_%2d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test1.mp4
    proc = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

    _extent=[long_range[0], long_range[1], lat_range[0], lat_range[1]]

    years = get_data_as_yearlist(_start, _end)

    num_years = len(years)
    base_range = 0

    if START_OF_RUN == False:
        num_years += 6
        base_range = 6

    storms = []
    # make a temp list to hold the storm dataframes from a single year
    # for _idx, year in enumerate(years):
    for _idx in range(base_range, num_years):
        print "idx:", _idx, " len years:", len(years)
        year = years[_idx-base_range]
        storms = get_storms_from_year(year)

        num_storms = len(storms)
        print "Storms:", num_storms

        # figure out what the year is
        _year = year.loc[:, "Season"].unique()[0]

        storm_max_track_length = np.array([len(storm) for storm in storms]).max()

        # currently frames per year is based off of 1 year per second
        _frames_per_year = int(vid_fps * (1.0 - lag_time))

        # each tick is 24 hours so go through until all ticks have been
        # accounted for
        for _x in range(1, _frames_per_year+1):
            total_layers = 0
            #
            # if _x > 1:
            #     break

            _slice = _x / (_frames_per_year * 1.0)

            _filename = "xplot_{}_{:0>2}.png".format(_year, _x)

            fig, ax = make_historical_diagram(long_range, lat_range)

            # additive code from:
            # http://samreay.github.io/blog/2016/10/01/additive.html
            fig.subplots_adjust(0, 0, 1, 1)

            # Draw the empty axis, which we use as a base.
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            _buffer = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)

            # the first buffer which is used for the new track in color
            first = _buffer.astype(np.int16).reshape(h, w, -1) #int16 so we dont overflow
            first[first[:, :, -1] == 0] = 0 # Set transparent pixels to 0

            # the second buffer for loading in the historical trails from files
            second = _buffer.astype(np.int16).reshape(h, w, -1) #int16 so we dont overflow
            second[second[:, :, -1] == 0] = 0 # Set transparent pixels to 0

            for storm in storms:
                total_layers += 1
                # if total_layers > 1:
                #     break

                # the longest storm in in the year is known by storm_max_track
                scaler = storm_max_track_length / (len(storm) * 1.0)

                ax.clear()  # maybe clear erases some of the axis settings, not just the canvas?
                ax.set_xlim(long_range[0], long_range[1])
                ax.set_ylim(lat_range[0], lat_range[1])
                ax.patch.set_facecolor('#000000')
                plot_current_storm(fig, ax, storm, _slice, scaler)
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)
                img[img[:, :, -1] == 0] = 0
                first += img # Add these particles to the main layer

            # # attempt to average out values some so that there's not a ton
            # # of bright stuff that got clipped
            # print "first max", first.max()
            # _max = first.max()
            #
            # first[:,:,0] = np.sqrt(first[:,:,0]) * (255.0 / np.sqrt(_max))
            # first[:,:,1] = np.sqrt(first[:,:,1]) * (255.0 / np.sqrt(_max))
            # first[:,:,2] = np.sqrt(first[:,:,2]) * (255.0 / np.sqrt(_max))
            # print "total_layers:", total_layers
            #
            # print "max first:", first.max()
            #
            # first = first / (total_layers * 1.0)**0.5

            first = np.clip(first, 0, 255) # clip buffer back into int8 range
                                            # wonder if some kind of exp transform
                                            # might enable hdr-like effect

            # had a problem where the swarm was blocking the map even where
            # there were no trails. This makes every black pixel transparent
            # so the map isn't blocked unless it needs to be
            # for a in range(first.shape[0]):
            #     for b in range(first.shape[1]):
            #         if first[a][b][0] <= 3 and first[a][b][1] <= 3 and first[a][b][2] <= 3:
            #             first[a][b][3] = 0

            # and black parts transparent (if they're not already)
            first[((first[:,:,0] == 0) & (first[:,:,1] == 0) & (first[:,:,2] == 0))] = 0

            # build up a map that also has the previous nine years of storms on it
            # nine so the total showing is ten
            # map_image = imread("../data/grey_blue_na_2.png")

            layer_count = 0

            if _idx < 5:
                num_historical_years = _idx
            else:
                num_historical_years = 6
            for _y in range(1, num_historical_years):

                if _idx > 0:
                    layer_count += 1
                    # figure out which year we need and the path to that historic swarm

                    # current year - _y = old_yr
                    old_y = _year - _y
                    _path = "../imgs/xplots/xplot_{}_{}.png".format(old_y, _y)

                    # load old swarm
                    old_swarm = imread(_path)

                    ax.clear()  # maybe clear erases some of the axis settings, not just the canvas?
                    ax.set_xlim(long_range[0], long_range[1])
                    ax.set_ylim(lat_range[0], lat_range[1])
                    ax.patch.set_facecolor('#000000')

                    # apply swarm to current ax
                    # ax.imshow(old_swarm, alpha=1.0/(_y * 2.0), extent=_extent, aspect='auto')
                    # ax.imshow(old_swarm, alpha=1.0/(_y**2 / 2.0 + _y), extent=_extent, aspect='auto')
                    ax.imshow(old_swarm, alpha=1.0/(_y**2.0 * 1.0), extent=_extent, aspect='auto')
                    # ax.imshow(map_image, alpha=0.5, extent=_extent, aspect='auto')

                    fig.canvas.draw()
                    img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)
                    img[img[:, :, -1] == 0] = 0
                    second += img # Add these particles to the main layer

            # try to clean up the ghost trails some so it's not overpowering
            # the current year's storms
            if _idx > 0:
                # attempt to average out values some so that there's not a ton
                # of bright stuff that got clipped
                print "first max", second[:,:,0].max()
                _max = second[:,:,0].max()

                if _max == 0:
                    _max = 1

                second[:,:,0] = np.sqrt(second[:,:,0]) * (255.0 / np.sqrt(_max))
                second[:,:,1] = np.sqrt(second[:,:,1]) * (255.0 / np.sqrt(_max))
                second[:,:,2] = np.sqrt(second[:,:,2]) * (255.0 / np.sqrt(_max))
                print "total_layers:", layer_count

                print "max first:", second[:,:,0].max()

                second = second / ((layer_count  + 1 )* 1.0)**0.5

                second = np.clip(second, 0, 255) # clip buffer back into int8 range
                                                # wonder if some kind of exp transform
                                                # might enable hdr-like effect

            # had a problem where the swarm was blocking the map even where
            # there were no trails. This makes every black pixel transparent
            # so the map isn't blocked unless it needs to be
            # for a in range(second.shape[0]):
            #     for b in range(second.shape[1]):
            #         if second[a][b][0] == 0 and second[a][b][1] == 0 and second[a][b][2] == 0:
            #             second[a][b][3] = 0

            # and black parts transparent (if they're not already)
            second[((second[:,:,0] == 0) & (second[:,:,1] == 0) & (second[:,:,2] == 0))] = 0

            ###########################################################
            ## Try manually adding in the map at the end and reclip ###
            ###########################################################

            ax.clear()  # maybe clear erases some of the axis settings, not just the canvas?
            ax.set_xlim(long_range[0], long_range[1])
            ax.set_ylim(lat_range[0], lat_range[1])
            ax.patch.set_facecolor('#000000')

            # apply swarm to current ax
            # ax.imshow(old_swarm, alpha=1.0/((_y + 1.0) * 1.0), extent=_extent, aspect='auto')
            ax.imshow(map_image, alpha=1.0, extent=_extent, aspect='auto')

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)
            img[img[:, :, -1] == 0] = 0
            second += img # Add these particles to the main layer

            ### At this point it's the old swarm and the map, just clip again and hope it's good
            second = np.clip(second, 0, 255) # clip buffer back into int8 range
                                            # wonder if some kind of exp transform
                                            # might enable hdr-like effect

            ax.clear()
            plt.axis("off")
            # ax.imshow(map_image, alpha=1.0, extent=_extent, aspect='auto')

            ax.imshow(second.astype(np.uint8), aspect='auto', alpha=1.0, extent=_extent)

            ax.imshow(first.astype(np.uint8), aspect='auto', alpha=1.0, extent=_extent)

            # make a color bar axis to put inside
            # cbaxis = inset_axes(ax, width="5%", height="50%", loc=3)
            #
            # plt.colorbar(cax=cbaxis, ticks=[157, 130, 111, 96, 74, 39, 0])

            # Add colorbar, make sure to specify tick locations to match desired ticklabels
            # cbar = figure.colorbar(histo_image, ticks=[157, 130, 111, 96, 74, 39, 0], cax=cax)
            # cbar.ax.set_yticklabels(['5^', '4^', '3^', '2^', '1^', 'T.S.^', 'T.D.^'])  # vertically oriented colorbar

            ax.imshow(map_key, aspect="auto", alpha=1.0, extent=_extent)

            ax.annotate(str(_year), xy=(-109, 47), size=40, color='#AAAAAA')
            ax.annotate("@pixelated_brian", xy=(-104, 12), size=15, color="#999999")
            # fig.savefig("../imgs/swarm/{}".format(_filename), pad_inches=0, transparent=True)

            fig.canvas.draw()

            _exit_buffer = fig.canvas.tostring_argb()
            # write to pipe
            proc.stdin.write(_exit_buffer)

            ax.clear()

            plt.close("all")

            # delete the buffers or stuff will accumulate (fun bug)
            del first
            del second

        # try to add some pause time between years, how about 0.15 of a second
        for _ in range(int(vid_fps * 0.15)):
            proc.stdin.write(_exit_buffer)

    for idx in range(vid_fps * 3):
        proc.stdin.write(_exit_buffer)

    # try to write the video
    proc.communicate()

def make_diagram(_year, long_range, lat_range):
    '''
    do all of the stuff for the graphics of making a hurricane plot

    takes in the year dataframe in order to get the necessary data

    returns ax which is the figure axis that the current hurricane track will be added upon
    '''
        # establish the figure
    figure, axis = plt.subplots(figsize=(19.2,10.80), dpi=100)

    data = np.linspace(165.0, 0, 100).reshape(10,10)
#     data = np.clip(randn(250, 250), -1, 1)

    histo_image = axis.imshow(data, interpolation='nearest', cmap="inferno")

    divider = make_axes_locatable(axis)

    cax = divider.append_axes("right", size="2%", pad=0.05)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = figure.colorbar(histo_image, ticks=[157, 130, 111, 96, 74, 39, 0], cax=cax)
    cbar.ax.set_yticklabels(['5^', '4^', '3^', '2^', '1^', 'T.S.^', 'T.D.^'])  # vertically oriented colorbar

    axis.imshow(map_image, extent=_extent,zorder=0)
    axis.set_title("North American Hurricane Tracks " + str(_year), size=20)
    axis.set_xlabel("Longitude", size=16)
    axis.set_ylabel("Latitude", size=16)
    axis.set_facecolor("black")
    axis.set_xlim(long_range[0], long_range[1])
    axis.set_ylim(lat_range[0], lat_range[1])

    return figure, axis

def make_historical_diagram(long_range, lat_range):
    '''
    do all of the stuff for the graphics of making a hurricane plot

    takes in the year dataframe in order to get the necessary data

    returns ax which is the figure axis that the current hurricane track will be added upon
    '''
    # establish the figure
    figure = plt.figure(figsize=(19.2, 10.80), dpi=100)

    axis = figure.add_subplot(111)
    axis.set_facecolor("#000000")

#     figure, axis = plt.subplots(figsize=(19.2,12.00), dpi=100)

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
    axis.set_xlim(long_range[0], long_range[1])
    axis.set_ylim(lat_range[0], lat_range[1])

    return figure, axis

def plot_historical_storm(fig, ax, old_storm):
    '''
    returns axis that has had a historical storm scatter plot applied

    in long run make the rest of the scatter plot transparent so it can be additively added to historical stuff
    '''

    # get data for the plot
    old_length, old_lat, old_long, old_width, old_colors = transform_storm(old_storm, True)

    # if the storm track is too short the values above will equal 0, and no data is returned
    # in that case don't render the path but simply pass

    if old_length != 0:
        # actually make the plot
        ax.scatter(old_long, old_lat, alpha=1, s=old_width, c=old_colors, lw=0)
    else:
        pass

def plot_current_storm(fig, ax, storm, _slice=0, scaling=1.0):
    '''
    returns axis that has had a historical storm scatter plot applied

    in long run make the rest of the scatter plot transparent so it can be additively added to historical stuff
    '''

    # get data for the plot
    old_length, old_lat, old_long, old_width, old_colors = transform_storm(storm, False, _slice, scaling)

    # if the storm track is too short the values above will equal 0, and no data is returned
    # in that case don't render the path but simply pass

    if old_length != 0:
        # actually make the plot
        ax.scatter(old_long, old_lat, alpha=1, s=old_width, c=old_colors, lw=0)
    else:
        pass

def scale_lat(y, _height):
    '''
    Take a value from latitude range and modify it to screen position

    return the fixed value
    '''
    # top of picture is 0 and bottom of picture is _height?
    # ^ might have to reverse this but that is easy
    #
    #   ylim is 0 to 50 so for a few values of y:
    # y = 0    ypx = 1199
    # y = 25   ypx = ~600
    # y = 35   ypx = (50 - y)/50 * 1200 = ~360
    # y = 50   ypx = 0
    return int((y) / 50.0 * _height)


def scale_long(x, _width):
    '''
    Take a value from longitude range and modify it to screen position

    return the fixed value
    '''
    # left of picture should be 0, right side of picture should be 1920
    # ^ again might have reversed
    #
    # xlim is -120 to -30 so for a few values of x:
    # x = -120  xpx = 0
    # x = -75   xpx = ~960
    # x = -40
    # x = -30   xpx = 1920
    #
    # initial range is -120 to -30
    #
    # x + 120 range becomes:
    # 0 to 90
    # then (x / 90.0) * 1920
    #
    # therefore xpx = ((x + 120) / 90.0) * 1920
    return ((x + 120) / 90.0) * 1920

def transform_storm(storm, historical=False, _slice=0, scaling=1.0):
    ##################################
    # get np columns for key data ####
    ##################################

    if len(storm) >= 10:
        interpolation_multiplier = int(24 * scaling)

        _lat = storm.Latitude.apply(lambda x: float(x))
        _long = storm.Longitude.apply(lambda x: float(x))

        _width = (storm.loc[:, "saffir_simpson_cat"] + 2)**1.3 * 100
        _colors = storm.loc[:,"Wind(WMO)"].apply(lambda x: x / 165.0)

        x = np.arange(len(_lat))

        # Interpolate the data using a cubic spline to "new_length" samples
        # going from daily intervals to (interpolated) hourly
        if historical is False:
            new_length = interpolation_multiplier * len(_lat)
        else:
            new_length = interpolation_multiplier * len(_lat)

        # figure out length of new arrays
        new_x = np.linspace(x.min(), x.max(), new_length)

        _chop = int(_slice * new_length)

        # actually do the interpolation
        new_lat = interpolate.interp1d(x, _lat, kind='cubic')(new_x)
        new_long = interpolate.interp1d(x, _long, kind='cubic')(new_x)
        new_width = interpolate.interp1d(x, _width, kind='cubic')(new_x)
        new_colors = interpolate.interp1d(x, _colors, kind='cubic')(new_x)

        if historical is False:
            # convert colors from float to rgba using colormap
            new_colors = cm.inferno(new_colors)
        else:
            # try bone colors
            new_colors = cm.Greys_r(new_colors)

        # if slice is that the default value then return all of the data
        if _slice == 0:
            return new_length, new_lat, new_long, new_width, new_colors
        else:
            return new_length, new_lat[:_chop], new_long[:_chop], \
                new_width[:_chop], new_colors[:_chop]

    # the storm track is too short, return guardian values
    else:
        return 0, 0, 0, 0, 0

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
    del years, data_na

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

if __name__ == "__main__":
    main()
