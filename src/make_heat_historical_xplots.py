import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# %matplotlib inline

from scipy import interpolate
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.misc import imread

def main():
    map_image = imread("../data/north_america_edited.png")

    lat_range = (10, 50)
    long_range = (-110, -30)

    _extent=[long_range[0], long_range[1], lat_range[0], lat_range[1]]

    _start = 1900
    _end = 2016

    years = get_data_as_yearlist(_start, _end)

    storms = []
    # make a temp list to hold the storm dataframes from a single year
    for year in years:

        fig, ax = make_historical_diagram(long_range, lat_range)

        # additive code from:
        # http://samreay.github.io/blog/2016/10/01/additive.html
        fig.subplots_adjust(0, 0, 1, 1)

        # Draw the empty axis, which we use as a base.
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        _buffer = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
        first = _buffer.astype(np.int16).reshape(h, w, -1) #int16 so we dont overflow
        first[first[:, :, -1] == 0] = 0 # Set transparent pixels to 0

        storms = get_storms_from_year(year)

        num_storms = len(storms)

        total_layers = 0

        # figure out what the year is
        _year = year.loc[:, "Season"].unique()[0]

        _filename = "xplot_{}.png".format(_year)

        for storm in storms:
            total_layers += 1

            ax.clear()  # maybe clear erases some of the axis settings, not just the canvas?
            ax.set_xlim(long_range[0], long_range[1])
            ax.set_ylim(lat_range[0], lat_range[1])
            ax.patch.set_facecolor('#000000')
            plot_historical_storm(fig, ax, storm, 1)
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)
            img[img[:, :, -1] == 0] = 0
            first += img # Add these particles to the main layer

        # peak color is yellow/white but yellow is blue + green so look for
        # blue r(B)g, not red (R)bg
        _max = first[:,:,1].max()
        # print "first max", _max


        # first[:,:,0] = np.sqrt(first[:,:,0]) * (255.0 / np.sqrt(_max))
        # first[:,:,1] = np.sqrt(first[:,:,1]) * (255.0 / np.sqrt(_max))
        # first[:,:,2] = np.sqrt(first[:,:,2]) * (255.0 / np.sqrt(_max))


        ##################################################
        ### Start color processing weirdness #############
        ##################################################
        # divide by 25 so that no particular year should max out the
        # pixel brightness, in that manner all years are relative
        #
        # leave it to the rendering of the heatmap to make the brightness
        # relative to the strength overall
        #
        # highest value seen is ~740 so / 3.0 will keep all values below 255
        # print "boink", first[..., 0:3].shape
        # ^ boink (1200, 1920, 3)
        # whut?

        # works but doesn't seem to make sense isn't pythonic anyways
        # first[...,0:3] = first[...,0:3] / np.array([3.0, 3.0, 3.0])

        first[...,0] = first[...,0] / 3.0
        first[...,1] = first[...,1] / 3.0
        first[...,2] = first[...,2] / 3.0
        ##################################################
        ### End color processing weirdness ###############
        ##################################################

        _endmax = first[:,:,1].max()

        print "year: {:4d} storms: {:0>2d} Initial Max: {:0>3d} Final Max: {:0>3d}".format(_year, total_layers, _max, _endmax)
        # first = first / (total_layers * 1.0)**0.5

        first = np.clip(first, 0, 255) # clip buffer back into int8 range
                                        # wonder if some kind of exp transform
                                        # might enable hdr-like effect

        # if the rgb part is 0 then it's black, make it transparent then
        # for a in range(first.shape[0]):
        #     for b in range(first.shape[1]):
        #         if first[a][b][0] == 0 and first[a][b][1] == 0 and first[a][b][2] == 0:
        #             first[a][b][3] = 0
        # ^ supposedly the following does the above
        # thanks to the following link this section runs 10x + faster
        # https://stackoverflow.com/questions/19770361/find-index-positions-where-3d-array-meets-multiple-conditions
        first[((first[:,:,0] == 0) & (first[:,:,1] == 0) & (first[:,:,2] == 0))] = 0

        ax.clear()
        plt.axis("off")
        # ax.imshow(map_image, extent=_extent)
        ax.imshow(first.astype(np.uint8), aspect='auto', alpha=1.0, extent=_extent)
        fig.savefig("../imgs/xplots4/{}".format(_filename), pad_inches=0, transparent=True)
        ax.clear()

        del first
        plt.close("all")

def make_diagram(_year):
    '''
    do all of the stuff for the graphics of making a hurricane plot

    takes in the year dataframe in order to get the necessary data

    returns ax which is the figure axis that the current hurricane track will be added upon
    '''
        # establish the figure
    figure, axis = plt.subplots(figsize=(19.2,12.00), dpi=100)

    data = np.linspace(165.0, 0, 10000).reshape(100,100)
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
    figure = plt.figure(figsize=(19.2, 12.0), dpi=100)

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

def plot_historical_storm(fig, ax, old_storm, thickness=0):
    '''
    returns axis that has had a historical storm scatter plot applied

    in long run make the rest of the scatter plot transparent so it can be additively added to historical stuff
    '''

    if thickness == 0:
        # get data for the plot
        old_length, old_lat, old_long, old_width, old_colors = transform_storm(old_storm, historical=True)
    else:
        old_length, old_lat, old_long, old_width, old_colors = transform_storm(old_storm, historical=True, thickness=thickness)


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
    # y = 35   ypx = (50 - y)/50 * 100 = ~360
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
    # xlim is -110 to -30 so for a few values of x:
    # x = -110  xpx = 0
    # x = -75   xpx = ~960
    # x = -40
    # x = -30   xpx = 1920
    #
    # initial range is -110 to -30
    #
    # x + 110 range becomes:
    # 0 to 90
    # then (x / 90.0) * 1920
    #
    # therefore xpx = ((x + 110) / 90.0) * 1920
    return ((x + 110) / 90.0) * 1920

def transform_storm(storm, historical=False, thickness=0):
    ##################################
    # get np columns for key data ####
    ##################################

    if len(storm) >= 10:

        _lat = storm.Latitude.apply(lambda x: float(x))
        _long = storm.Longitude.apply(lambda x: float(x))

        if historical == False:
            _width = (storm.loc[:, "saffir_simpson_cat"] + 2)**1.3 * 100
        else:
            # figure out the thickness:
            if thickness == 0:
                _width = (storm.loc[:, "saffir_simpson_cat"] + 2)**1.3 * 25
            else:
                _width = (storm.loc[:, "saffir_simpson_cat"] + 2)**1.3 * 100 * (1/(1.0* thickness))

        _colors = storm.loc[:,"Wind(WMO)"].apply(lambda x: x / 165.0)

        x = np.arange(len(_lat))

        # Interpolate the data using a cubic spline to "new_length" samples
        # going from daily intervals to (interpolated) hourly
        if historical is False:
            new_length = 24 * len(_lat)
        else:
            new_length = 24 * len(_lat)

        # figure out length of new arrays
        new_x = np.linspace(x.min(), x.max(), new_length)

        # actually do the interpolation
        new_lat = interpolate.interp1d(x, _lat, kind='cubic')(new_x)
        new_long = interpolate.interp1d(x, _long, kind='cubic')(new_x)
        new_width = interpolate.interp1d(x, _width, kind='cubic')(new_x)
        new_colors = interpolate.interp1d(x, _colors, kind='cubic')(new_x)

        if historical is False:
            # convert colors from float to rgba using colormap
            new_colors = cm.inferno(new_colors)
        else:
            # must be doing a historical run, use a different color schema
    #         new_colors = cm.Greys(new_colors)

            # try bone colors
            new_colors = cm.Greys_r(new_colors)

#             # make white the 'hottest' and black the coolest
#             new_colors = 1.0 - new_colors

#             a_line = np.linspace(0.01, 0.01, new_length).reshape(1, new_length)

            # fix new_color alpha's with a_line alphas
    #         new_colors[:,3] = a_line

            # try making the plot transparency 1.0 for historical
#             new_colors[:,3] = 1.0

        return new_length, new_lat, new_long, new_width, new_colors

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

if __name__ == "__main__":
    main()
