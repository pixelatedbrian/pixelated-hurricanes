import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# used to convert float to rgba colormap
import matplotlib.cm as cm

# %matplotlib inline

from scipy import interpolate
# used for making mercator project map
# from mpl_toolkits.basemap import Basemap
# used for formating colormap on the side
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.misc import imread

def main():

    data = pd.read_csv("../data/allstorms.csv")

    # data dictionary
    # N/A,Year,#,BB,BB,N/A,YYYY-MM-DD HH:MM:SS,N/A,deg_north,deg_east,kt,mb,N/A,%,%,N/A

    data.loc[:,"Season"] = data.loc[:,"Season"].apply(lambda x: int(x))

    map_image = imread("../data/north_america_edited.png")

    # fix basin labels
    data.loc[:, "Basin"] = data.loc[:, "Basin"].apply(lambda x: x.replace(" ", ""))

    # break out NA Atlantic basin so we can focus on that
    data_na = data[data.loc[:, "Basin"] == "NA"]

    # use wind speed to determine hurricane category
    data_na.loc[:,"saffir_simpson_cat"] = data_na["Wind(WMO)"].apply(lambda x: safsimpsonize(x))

    make_plots(data_na, map_image)

def make_plots(data_na, map_image):
    '''
    Main function that steps through each year and writes plots for each storm
    in sequence in frames where each frame is 24 hours of interpolated storm data

    output: dumps files in ../imgs/ dir to be post processed and stitched into video
    '''
    # make a list for the years
    years = []

    # step through the Seasons (years) and make a new dataframe for each one
    for year in data_na.loc[:, "Season"].unique():
        temp = data_na[data_na.loc[:, "Season"] == year]
        years.append(temp)

    # get rid of a nan DataFrame
    years.pop(0)

    #loop through years in future gif
    start = 164 - (2016 - 2000)
    end = 164 - (2016 - 2016)

    print start, end

    print "total years", len(years)

    #how many historical years
    trailing_years = 0

    # max window size in years
    yr_window = 3

    # start a new frame count for each year
    frame_count = 0

    for idz, year in enumerate(years[start:end]):

        # make a temp list to hold the storm dataframes from a single year
        storms = []

        # figure out what the year is
        _year = year.loc[:, "Season"].unique()[0]

    #         this_year = str(year.loc[:, "Season"].unique()[0])
    #         print "year", this_year

        # step through the year and make a dataframe for each storm
        for storm in year.loc[:,"Serial_Num"].unique():
            storms.append(year[year.loc[:, "Serial_Num"] == storm])

        # now have a list of storm dataframes



        # step through all of the storms and add their tracks to the map
        for idy, storm in enumerate(storms):
            # old place to make diagram

            _lat = storm.Latitude.apply(lambda x: float(x))
            _long = storm.Longitude.apply(lambda x: float(x))
    #         _cat = storm.loc[:, "saffir_simpson_cat"].apply(lambda x: float(20.0)**(x/2.0))
            _width = (storm.loc[:, "saffir_simpson_cat"] + 2) * 150
            _colors = storm.loc[:,"Wind(WMO)"].apply(lambda x: x / 165.0)


            # if the track has a very small number of data points then skip it
            if len(_lat) < 10:
                pass
            else:
                # try some crazy interpolation stuff

                x = np.arange(len(_lat))

                # Interpolate the data using a cubic spline to "new_length" samples
                # going from daily intervals to (interpolated) hourly
                new_length = 24 * len(_lat)

                # figure out length of new arrays
                new_x = np.linspace(x.min(), x.max(), new_length)

                # actually do the interpolation
                new_lat = interpolate.interp1d(x, _lat, kind='cubic')(new_x)
                new_long = interpolate.interp1d(x, _long, kind='cubic')(new_x)
                new_width = interpolate.interp1d(x, _width, kind='cubic')(new_x)
                new_colors = interpolate.interp1d(x, _colors, kind='cubic')(new_x)

                # convert colors from float to rgba using colormap
                new_colors = cm.inferno(new_colors)

                # set trail duration, maybe 3 days which 24 ticks per day is 72
                track_length = 24

                # figure out number of frames for the track
                frames = new_length // track_length

                # figure out the number of leftover extra frames to use all steps
                left_over_frames = new_length % track_length

                a_line = np.linspace(0.1, 0, left_over_frames)[::-1].reshape(1, left_over_frames)

                # frame calculated stuff
                t_long = new_long[:left_over_frames]
                t_lat = new_lat[:left_over_frames]
                t_width = new_width[:left_over_frames]
                t_colors = new_colors[:left_over_frames]

                # fix t_color alpha's with a_line alphas
                t_colors[:,3] = a_line

    #             print "a_line.shape", a_line.shape
    #             print "t_color.shape", t_color.shape

                fig, ax = make_diagram(year, map_image)

                ax.scatter(t_long, t_lat, s=t_width, c=t_colors)

                # for loop for each frame
                for frame in range(frames):
                    # print the year_storm_frame for practice for the file name saving
                    _filename = "{:0>8}.png".format(frame_count)
                    _path = "../imgs/" + _filename

                    # time window start:
                    _ts = track_length * frame

                    # time window end:
                    _te = track_length * (frame + 1)

                    a_line = np.linspace(0.1, -0.1, _te).reshape(1, _te)

                    a_line[a_line < 0.0] = 0

                    # frame calculated stuff
                    t_long = new_long[: _te][::-1]
                    t_lat = new_lat[: _te][::-1]
                    t_width = new_width[: _te][::-1]
                    t_colors = new_colors[: _te][::-1]

                    # fix t_color alpha's with a_line alphas
                    t_colors[:,3] = a_line

                    fig, ax = make_diagram(year, map_image)
                    ax.scatter(t_long, t_lat, s=t_width, c=t_colors)

                    plt.tight_layout()

                    print "Saving {:>15} \r".format(_filename)
                    fig.savefig(_path)
                    plt.close("all")
                    frame_count += 1

def make_diagram(year_df, map_image):
    '''
    do all of the stuff for the graphics of making a hurricane plot

    takes in the year dataframe in order to get the necessary data

    returns ax which is the figure axis that the current hurricane track will be added upon
    '''
        # establish the figure
    figure, axis = plt.subplots(figsize=(19.2,12.00), dpi=100)

    _year = str(year_df.loc[:, "Season"].unique()[0])

    data = np.linspace(165.0, 0, 10000).reshape(100,100)
#     data = np.clip(randn(250, 250), -1, 1)

    histo_image = axis.imshow(data, interpolation='nearest', cmap="inferno")

    divider = make_axes_locatable(axis)

    cax = divider.append_axes("right", size="2%", pad=0.05)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = figure.colorbar(histo_image, ticks=[157, 130, 111, 96, 74, 39, 0], cax=cax)
    cbar.ax.set_yticklabels(['5^', '4^', '3^', '2^', '1^', 'T.S.^', 'T.D.^'])  # vertically oriented colorbar

    axis.imshow(map_image, extent=[-120, -30, 0, 50],zorder=0)
    axis.set_title("North American Hurricane Tracks " + _year, size=20)
    axis.set_xlabel("Longitude", size=16)
    axis.set_ylabel("Latitude", size=16)
    axis.set_facecolor("black")
    axis.set_xlim(-120.0, -30.0)
    axis.set_ylim(0, 50.0)

    return figure, axis

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

if __name__ == "__main__":
    main()
