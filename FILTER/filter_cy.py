from __future__ import print_function
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, time
from PIL import Image
from argparse import ArgumentParser
import scipy, scipy.ndimage
import multiprocessing as mp


def cai_filter(image):

    cai_image = np.zeros_like(image) # create an empty image with the same dimensions
    h = cai_image.shape[0]
    w = cai_image.shape[1]
    size = h*w

    i = 0

    for y in range(0, h-1):         # for all pixels in y axis
        for x in range(0, w-1):     # for all pixels in x axis
            
            no = int(image[y - 1, x])
            ne = int(image[y - 1, x + 1])
            nw = int(image[y - 1, x - 1])
            so = int(image[y + 1, x])
            se = int(image[y + 1, x + 1])
            sw = int(image[y + 1, x - 1])
            ea = int(image[y, x + 1])
            we = int(image[y, x - 1])

            cai_array = [nw, no, ne, we, ea, sw, so, se]

            if (np.max(cai_array) - np.min(cai_array)) <= 20:       # If the max number of the neighbouring pixels are less than or equal to
                px = np.mean(cai_array)                             # 20 in value(0-255) then just set the pixel to the mean
            elif (np.absolute(ea - we) - np.absolute(no - so)) > 20:   # If the absolute value(not negative. F.ex. -5 = 5) of that is more than 20
                px = (no + so) / 2                                    # the value is northern + southern pixel divided by 2
            elif (np.absolute(no - so) - np.absolute(ea - we)) > 20:
                px = (ea + we) / 2
            elif (np.absolute(ne - sw) - np.absolute(se - nw)) > 20:   # If the absolute value(not negative. F.ex. -5 = 5) of that is more than 20
                px = (se + nw) / 2                                    # the value is northern + southern pixel divided by 2
            elif (np.absolute(se - nw) - np.absolute(ne - sw)) > 20:
                px = (ne + we) / 2
            else:
                px = np.median(cai_array)                           # Median is backup. Median just selects the item that is most in the middle.

            px = int(px)

            cai_image[y, x] = px            # Set the value of the current pixel to px. 

    return cai_image

"""
def calc_sigma(image):
    d = image
    m = 3
    sigma_0 = 9
    #Wu et al.
    #http://ws.binghamton.edu/fridrich/Research/double.pdf
    #m = the neighbourhood pixels, here the 3x3 pixels around the selected pixel
    #sum the value of the neighbourhood - the overall variance of the SPN.
    #Select the max value, so if the value is negative, it returns a black(empty) pixel

    sigsum = np.sum((sum(d)**2) -(float(sigma_0)))
    sigmas=(0, ((1/(m**2))*sigsum))

    local_variance = max(sigmas)

    return local_variance
"""

def calc_sigma(image):
    d = image
    m = 3
    sigma_0 = 9
    h = d.shape[0]
    w = d.shape[1]
    #Wu et al.
    #http://ws.binghamton.edu/fridrich/Research/double.pdf
    #m = the neighbourhood pixels, here the 3x3 pixels around the selected pixel
    #sum the value of the neighbourhood - the overall variance of the SPN.
    #Select the max value, so if the value is negative, it returns a black(empty) pixel
    neigh = []
    for y in range(0, h):         # for all pixels in y axis
        for x in range(0, w):     # for all pixels in x axis
            the_sub = ((d[y,x])**2) -(float(sigma_0))
            neigh.append(the_sub)


    sigsum = np.sum(neigh)

    sigmas=[0, ((1/(m**2))*sigsum)]

    local_variance = max(sigmas)
    return local_variance


def wavelet(image):
    d = image
    sigma_0 = 9

    wav_image = np.zeros_like(d) # create an empty image with the same dimensions
    h = d.shape[0]
    w = d.shape[1]


    
    for y in range(0, h-1):         # for all pixels in y axis
        for x in range(0, w-1):     # for all pixels in x axis

            d_px = d[y, x]          # Select current pixel
                                    # Select the 9 pixels around current pixel from the CAI subtraction            
            no = int(image[y - 1, x])
            ne = int(image[y - 1, x + 1])
            nw = int(image[y - 1, x - 1])
            so = int(image[y + 1, x])
            se = int(image[y + 1, x + 1])
            sw = int(image[y + 1, x - 1])
            ea = int(image[y, x + 1])
            we = int(image[y, x - 1])
            neighbour = np.array([[nw, no, ne],
                                  [we, d_px, ea],
                                  [sw, so, se]], dtype=np.float)
                                                                    # According to the formulas in Wu et al.
            sigma_div = sigma_0/(calc_sigma(neighbour) + sigma_0)   # get the estimated local variance for the pixel
            px = d_px * sigma_div                                   # multiply subtracted CAI with the local variances
            px = int(px)                                            # Estimated camera reference SPN
            wav_image[y,x] = px


    return wav_image

def get_spn(wav_image):

    try:
        average = wav_image[wav_image!=0].mean()    #average all the noise and add them 
        return average
    except Exception as e:
        print(e)

def crop_center(img, cropy, cropx):
    """
    :param img: array
        2D input data to be cropped
    :param cropx: int
        x axis of the pixel amount to be cropped
    :param cropy: int
        y axis of the pixel amount to be cropped
    :return:
        return cropped image 2d data array.
    """
    if cropx == 0 or cropy == 0:
        return img
    else:
        y, x = img.shape

        startx = (x // 2) - (cropx // 2)
        starty = (y // 2) - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]


def filter(img, h, w):

    if os.path.isfile(img):
        original = cv2.imread(img, 0)                  # the 0 means read as grayscale

        if h == 0 or w == 0:
            h = original.shape[0]
            w = original.shape[1]

        average_orig = original.mean()    #average all the noise and add them 


        cropped = crop_center(original, h, w)


        cai_image = cai_filter(cropped)


        d_image = cv2.subtract(cropped, cai_image)


        wav_image = wavelet(d_image)

        return(wav_image)

        
    else:
        print("file does not exist")
        return None

def filter_main(folder, h, w):

    start_time = time.time()
    images = []
    j = 0
    onlyfiles = next(os.walk(folder))[2] #dir is your directory path as string
    allfiles=os.listdir(folder)
    size = len(onlyfiles)
    est_camera_ref = []
    imlist=[filename for filename in allfiles if filename[-4:] in [".jpg",".JPG"]]
    for f in imlist:
        paths = folder+f
        i = filter(paths, int(h), int(w))
        spn = get_spn(i)
        est_camera_ref.append(spn)
        images.append(i)
        kake = "filtered\\"
        path_create = folder + kake
        path_new = folder + kake + f
        if not os.path.exists(path_create):
            os.makedirs(path_create)
        plt.imsave(path_new, i, cmap="gray" )
        during_time = time.time() - start_time
        progress = round((j / size) * 100, 1)
        progressbar = int(progress / 4)
        frogress = "{0:.2f}".format(progress)
        elapse = "{0:.2f}".format(during_time)
        
        if j != 0:
            est_time = "{0:.2f}".format((during_time/j)*(size-j))
        else:
            est_time = "{0:.2f}".format(progress*size)
        
        print('\r|{}|{}% Time elapsed: {}, estimate: {}'.format(("█" * progressbar), frogress, elapse, est_time), end="", flush=True)
        j += 1

    elapsed_time = time.time() - start_time
    print('\r|{}|{}%'.format(("█" * 25), 100), end="", flush=True)
    print("Time taken:", elapsed_time)
    print("Estimated camera reference:", (sum(est_camera_ref)/len(est_camera_ref)))

"""
parser = ArgumentParser()
parser.add_argument("path", help="python filter_cy PATHHERE height width")
parser.add_argument("height", help="python filter_cy PATHHERE height width")
parser.add_argument("width", help="python filter_cy PATHHERE height width")

args = parser.parse_args()
filter_main(args.path, args.height, args.width)
"""

"""
path1 = 'F:\\Dropbox\\Dropbox\\SPN\\AlexC\\'
path2 = 'F:\\Dropbox\\Dropbox\\SPN\\Bjarke\\'
path3 = 'F:\\Dropbox\\Dropbox\\SPN\\Gabbi\\'
path4 = 'F:\\Dropbox\\Dropbox\\SPN\\Wenche\\'
path5 = 'F:\\Dropbox\\Dropbox\\SPN\\Monica\\'


lista = [path1, path2, path3, path4, path5]
for p in lista:
    filter_main(p, 512, 512)
"""

#filter_main('F:\\Dropbox\\Dropbox\\SPN\\lena\\', 512, 512)

filter_main('F:\\Dropbox\\Dropbox\\SPN\\Monica\\', 512, 512)