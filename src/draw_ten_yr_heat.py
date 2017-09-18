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

def easy_buffer(_filename, h, w, verbose=False):
    '''
    load file with filename, turn into np buffer

    returns:
    new np buffer
    '''
    # load track file
    storm_tracks = imread("../imgs/xplots4/{}".format(_filename))

    new_buffer = np.frombuffer(storm_tracks, np.uint8)

    temp_buffer = new_buffer.astype(np.int16).reshape(h, w, 4) #int16 so we dont overflow

    if verbose == True:
        print "new_buffer shape:", temp_buffer.shape
        print "max value in current heat buffer:", temp_buffer[..., 1].max()

    temp_buffer[temp_buffer[:, :, -1] == 0] = 0

    return temp_buffer

def establish_canvas():
    '''
    basic canvas configuration for matplotlib

    Returns:
    fig, ax to use for the diagram
    '''
    # establish the figure
    fig = plt.figure(figsize=(19.2, 12.0), dpi=100)

    ax = fig.add_subplot(111)
    ax.set_facecolor("#000000")
    ax.clear()  # maybe clear erases some of the axis settings, not just the canvas?
    fig.subplots_adjust(0, 0, 1, 1)

    ax.set_xlim(-110.0, -30.0)
    ax.set_ylim(10, 50.0)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    _buffer = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)

    # the first buffer which is used for the new track in color
    main_buffer = _buffer.astype(np.int16).reshape(h, w, -1) #int16 so we dont overflow
    main_buffer[main_buffer[:, :, -1] == 0] = 0 # Set transparent pixels to 0

    return fig, ax, main_buffer

def draw_map(gam1=1.0, gam2=2.0, y1=1915, y2=2016, file_name=None,
            color_map=0, last_alpha=1.0):

    color_dict = {0:"inferno", 1:"plasma", 2:"magma", 3:"viridis", 4:"hot", 5:"afmhot",
                 6:"gist_heat", 7:"copper", 8:"bone", 9:"gnuplot", 10:"gnuplot2",
                  11:"CMRmap", 12:"pink", 13:"spring", 14:"autumn_r", 15:"cool",
                 16:"Wistia", 17:"seismic", 18:"RdGy_r", 19:"BrBG_r", 20:"RdYlGn_r",
                 21:"PuOr", 22:"brg", 23:"hsv", 24:"cubehelix", 25:"gist_earth",
                 26:"ocean", 27:"gist_stern", 28:"gist_rainbow_r", 29:"jet",
                 30:"nipy_spectral", 31:"gist_ncar"}

    _cmap = color_dict[color_map]

    # make the canvas
    fig, ax, first = establish_canvas()
    w, h = fig.canvas.get_width_height()

    # load in the background map image file
    map_image = imread("../data/new_ultra_map.png")

    # draw the map
    ax.imshow(map_image, extent=[-110, -30, 10, 50], aspect="auto")

    # paint the canvas so we can pull it into a map buffer
    fig.canvas.draw()

    # get the map image into a buffer
    map_img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)
    map_img[map_img[:, :, -1] == 0] = 0

    # print "max map heat:", map_img[..., 1].max()

    first += map_img

    ###############################################
    ### Finish the standard base of the diagram ###
    ###############################################

    # make a new buffer for the heatmap itself
    ax.clear()
    ax.set_xlim(-110.0, -30.0)
    ax.set_ylim(10.0, 50.0)
    ax.set_facecolor('#000000')
#     _heatmap = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
#     _heatmap = _heatmap.astype(np.int16).reshape(h, w, -1) #int16 so we dont overflow

#     # see if we really need this...
#     _heatmap[_heatmap[:, :, -1] == 0] = 0 # Set transparent pixels to 0
    _heatmap = np.zeros(shape=(h, w, 4))

    ###########################################################
    # how many years to include in the heatmap part         ###
    # Gather up layers for the heatmap and fade the last    ###
    # layer as needed                                       ###
    ###########################################################
    for x in range(y2-y1+1):
        # which year to start pulling heatmaps
        _year_zero = y1

        # figure out the filename for the current year
        _filename = "xplot_{}.png".format(_year_zero + x)

        print "opening contrail file: {} {:0.2f}".format(_filename, last_alpha)
        temp_buffer = easy_buffer(_filename, h, w)

        if last_alpha != 1.0 and x == (y2-y1):
            print "smoothing last frame in set"
            # if smoothing is turned on then try to fade the new data being
            # added into the frame depending on how far into the smoothing
            # we are

            temp_buffer[...,0] = temp_buffer[...,0] * last_alpha
            temp_buffer[...,1] = temp_buffer[...,1] * last_alpha
            temp_buffer[...,2] = temp_buffer[...,2] * last_alpha
            temp_buffer[...,3] = temp_buffer[...,3] * last_alpha

        # load the transparent background map and add it's values to the buffer
        _heatmap += temp_buffer

    # print "max heat coming out of layer processing:", _heatmap[..., 1].max()

    # currently the buffer is RGBA but since it's grey we only need one channel
    _buff = _heatmap[...,1]
    # max heat is 1500 so it seems like data is being loaded ok

    # prepare to draw the new buffer
    ax.clear()  # maybe clear erases some of the axis settings, not just the canvas?
    ax.set_xlim(-110.0, -30.0)
    ax.set_ylim(10, 50.0)
    ax.patch.set_facecolor('#000000')

    # draw the new combined heatmap
    ax.imshow(_buff, norm=PowerNorm(gamma=gam1/gam2), cmap=_cmap, extent=[-110, -30, 10, 50], alpha=1.0, aspect="auto")

#     fig.savefig("../imgs/test/heat_render/test_more.png", pad_inches=0, transparent=True)
    # paint the canvas
    fig.canvas.draw()

    # pull the paint back off the canvas into the buffer
    heat_img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)

    # print "max heat from heatmap buffer:", heat_img[..., 1].max()
    # make transparent parts black
#     heat_img[heat_img[:, :, -1] == 0] = 0

    # and black parts transparent (if they're not already)
    heat_img[((heat_img[:,:,0] == 0) & (heat_img[:,:,1] == 0) & (heat_img[:,:,2] == 0))] = 0

    # add the heatmap to the final buffer
    first += heat_img

    # print "max final heat:", first[..., 1].max()

    ###############################################
    ### Finish the dynamic heatmap part of plot ###
    ###############################################

    ax.clear()  # maybe clear erases some of the axis settings, not just the canvas?
    ax.set_xlim(-110.0, -30.0)
    ax.set_ylim(10.0, 50.0)
    ax.patch.set_facecolor('#000000')

    first = np.clip(first, 0, 255) # clip buffer back into int8 range
                    # wonder if some kind of exp transform
                    # might enable hdr-like effect

    # make the final draw
    ax.imshow(first.astype(np.uint8), extent=[-110, -30, 10, 50], aspect='auto', alpha=1.0)
    # write out file info so we can see how a map was made later w/ grid search
    desc = "{}-{}".format(y1, y2)
    ax.annotate(desc, xy=(-109, 48), size=40, color='#A0A0A0')

    # plt.close("all")
    if file_name != None:
        fig.savefig("../imgs/test/final_hr/{}".format(file_name), pad_inches=0, transparent=True)
        print "Wrote image: {}\r".format(file_name)
        plt.close("all")

    fig.canvas.draw()
    # extract the image as an ARGB string
    _exit_buffer = fig.canvas.tostring_argb()

    # keep memory from getting exhausted if there's a ton of draws
    plt.close("all")
    return _exit_buffer

def main():
    ################################
    ### Try to write to video?!  ###
    ################################

    # Thanks to Patrick Mende for providing the following link to enable
    # making the video files within python
    # https://stackoverflow.com/questions/4092927/generating-movie-from-python-without-saving-individual-frames-to-files/29525514#29525514

    # Open an ffmpeg process
    outf = '../imgs/test/final_hr/heat_movie.mp4'
    cmdstring = ('ffmpeg',
        '-y', '-r', '60', # overwrite, 30fps
        '-s', '{:d}x{:d}'.format(1920, 1200), # size of image string
        '-pix_fmt', 'argb', # format
        '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
        '-vcodec', 'libx264', '-crf', '25', outf) # output encoding

    # what we've been using when doing it manually
    # ffmpeg -r 24 -f image2 -s 1920x1080 -i xplot_2012_%2d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test1.mp4
    proc = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

    _start = 1915
    _end = 2016

    # load up list of files so we don't have to reload them every time that
    # we make a frame
    contrail_list = []

    for idx in range(_start, _end, 1):
        _filename = "xplot_{}.png".format(idx)

        # load track file
        storm_tracks = imread("../imgs/xplots4/{}".format(_filename))
        contrail_list.append(storm_tracks)

    for year2 in range(_start, _end, 1):

        # number of frames to ease in the new year of data:
        smoothing = 9

        for _toc in range(1,smoothing + 1):

            # figure out the alpha adjustment for the last frame
            frame_alpha = _toc / (smoothing * 1.0)

            _filename = "test_{}".format(year2)
            _e_buffer = draw_map(contrail_list, 3.5, 4.0, y1=1915, y2=year2, last_alpha=frame_alpha, color_map=5)

            # write to pipe
            proc.stdin.write(_e_buffer)

    # try to write the video
    proc.communicate()

if __name__ == "__main__":
    main()
