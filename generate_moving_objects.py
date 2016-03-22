import cv2
import sys
import os
import math
import numpy as np
from PIL import Image
from images2gif import writeGif

import utils as u

###########################################################################################
# script to generate moving objects as in http://arxiv.org/pdf/1602.00991v2.pdf
# saves in hdf5, npz, or jpg (individual frames) format
###########################################################################################

def get_line_iterator(img, P1, P2):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a
    line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel
        (shape: [numPixels, 3], row = [x,y,intensity])
        
    source http://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = float(dX)/float(dY)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = float(dY)/float(dX)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer

# generates and returns video frames in uint8 array
def generate_frames(shape=(64,64), seq_len=128, seqs=5, balls_per_image=6, ball_width=7):
    width, height = shape
    lims = (x_lim, y_lim) = width-ball_width, height-ball_width
    dataset = np.empty((seq_len*seqs, 2, width, height), dtype=np.uint8)
    for seq_idx in xrange(seqs):
        if seq_idx % 3==0:
            print seq_idx
        # randomly generate direc/speed/position, calculate velocity vector
        direcs = np.pi * (np.random.rand(balls_per_image)*2 - 1)
        speeds = np.random.randint(5, size=balls_per_image)+2
        veloc = [(v*math.cos(d), v*math.sin(d)) for d,v in zip(direcs, speeds)]
        positions = [(np.random.rand()*x_lim, np.random.rand()*y_lim) for _ in
                xrange(balls_per_image)]
        for frame_idx in xrange(seq_len):
            groundtruth = np.ones((height,width), dtype=np.uint8)
            for i,pos in enumerate(positions):
                cv2.circle(groundtruth, (int(pos[0]), int(pos[1])), ball_width/2, 0,
                        thickness=-1)
            origin = x0,y0 = width/2,6
            occluded_scene = np.copy(groundtruth)
            sensor_readings = np.zeros_like(groundtruth)
            def occlude(xx,yy):
                blocked = False
                for row in get_line_iterator(groundtruth, (np.float(xx),np.float(yy)),
                        (x0,y0))[::-1]:
                    if blocked:
                        occluded_scene[row[1], row[0]] = 0
                    if row[2] == 0 and not blocked:
                        sensor_readings[row[1], row[0]] = 1
                        blocked = True
            for y1 in xrange(-1,height+1):
                for x1 in [-1, width]:
                    occlude(x1,y1)
            for x1 in xrange(-1,width+1):
                for y1 in [-1, height]:
                    occlude(x1,y1)
            dataset[seq_len*seq_idx+frame_idx,0] = sensor_readings
            dataset[seq_len*seq_idx+frame_idx,1] = occluded_scene
            next_pos = [map(sum, zip(p,v)) for p,v in zip(positions, veloc)]
            def contains(p0, p1, box_size):
                xmin,xmax,ymin,ymax = p0[0]-box_size,p0[0]+box_size,p0[1]-box_size,p0[1]+box_size
                ret = p1[0]>xmin and p1[0]<xmax and p1[1]>ymin and p1[1]<ymax
                return ret
            # bounce off wall or occluded_scene area if we hit one
            for i, pos  in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j]+2 or (contains(origin, pos, 8) and
                            ((j==0 and positions[i][1]<=14) or (j==1 and positions[i][1]>14))):
                        veloc[i] = tuple(list(veloc[i][:j]) + [-1 * veloc[i][j]] +
                                list(veloc[i][j+1:]))
            positions = [map(sum, zip(p,v)) for p,v in zip(positions, veloc)]
            for pos, nex in zip(positions, next_pos):
                if contains(origin, pos, 8):
                    print pos, nex
        '''
        sens_imgs = [Image.fromarray(255*d[0]) for d in dataset]
        occl_imgs = [Image.fromarray(255*d[1]) for d in dataset]
        writeGif("sens.gif",sens_imgs,duration=0.1,dither=0)
        writeGif("occl.gif",occl_imgs,duration=0.1,dither=0)
        '''
    return dataset

def main(dest='', filetype='hdf5', frame_size=64, seq_len=128, seqs=5, ball_width=7,
        balls_per_image=6):
    dat = generate_frames(shape=(frame_size,frame_size), seq_len=seq_len, seqs=seqs, \
                                ball_width=ball_width, balls_per_image=balls_per_image)
    if filetype == 'hdf5':
        u.save_hd5py({'images': dat}, dest, 10)
    elif filetype == 'npz':
        np.savez(dest, dat)

if __name__ == '__main__':
    # make all arguments of main(...) command line arguments (with type inferred from
    # the default value) - this doesn't work on bools so those are strings when
    # passed into main.
    import argparse, inspect
    parser = argparse.ArgumentParser(description='Command line options')
    ma = inspect.getargspec(main)
    for arg_name,arg_type in zip(ma.args[-len(ma.defaults):],[type(de) for de in ma.defaults]):
        parser.add_argument('--{}'.format(arg_name), type=arg_type, dest=arg_name)
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
