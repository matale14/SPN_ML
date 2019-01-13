from __future__ import print_function
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, time
from PIL import Image
from argparse import ArgumentParser


cpdef unsigned char[:, :] cai_filter(unsigned char [:, :] image):
    
    cdef unsigned char[:, :] cai_image
    cai_image = np.zeros_like(image) # create an empty image with the same dimensions
    cdef int n, e, s, we, x, y, i, j, h, w, size, px
    cdef char[4] cai_array
    h = cai_image.shape[0]
    w = cai_image.shape[1]
    size = h*w
    cdef unsigned char[:, :] cai_diff
    cai_diff = np.zeros_like(image)  # Another clone just for the affected pixels

    i = 0
    j = 0

    for y in range(0, h):         # for all pixels in y axis
        for x in range(0, w):     # for all pixels in x axis
              
            try:
                n = image[y - 1, x] # North, east, south, west pixels
                e = image[y, x + 1]
                s = image[y + 1, x]
                we = image[y, x - 1]
            except IndexError:
                n = image[y, x]
                e = image[y, x]
                s = image[y, x]
                we = image[y, x]

            cai_array = [n, e, s, we]


            if (np.max(cai_array) - np.min(cai_array)) <= 20:       # If the max number of the neighbouring pixels are less than or equal to
                px = np.mean(cai_array)                             # 20 in value(0-255) then just set the pixel to the mean
            elif (np.absolute(e - we) - np.absolute(n - s)) > 20:   # If the absolute value(not negative. F.ex. -5 = 5) of that is more than 20
                px = (n + s) / 2                                    # the value is northern + southern pixel divided by 2
            elif (np.absolute(n - s) - np.absolute(e - we)) > 20:
                px = (e + we) / 2
            else:
                px = np.median(cai_array)                           # Median is backup. Median just selects the item that is most in the middle.


            if image[y, x] != int(px):      # If a pixel changes value, count it and apply in the cai_diff image
                cai_diff[y, x] = int(px)
                i += 1
            j += 1
            cai_image[y, x] = px            # Set the value of the current pixel to px. 


    return cai_image

cpdef int calc_sigma(unsigned char[:, :] image):
    d = image
    cdef int m, sigma_0, sigmas, local_variance, sigsum
    m = 3
    sigma_0 = 9

    #Wu et al.
    #http://ws.binghamton.edu/fridrich/Research/double.pdf
    #m = the neighbourhood pixels, here the 3x3 pixels around the selected pixel
    #sum the value of the neighbourhood - the overall variance of the SPN.
    #Select the max value, so if the value is negative, it returns a black(empty) pixel

    sigsum = ((1/m**2)* np.sum((d**2) -(float(sigma_0))))      

    sigmas=(0, sigsum)
    local_variance = max(sigmas)

    return local_variance

cpdef unsigned char[:, :] wavelet(unsigned char [:, :] image):
    cdef unsigned char[:, :] d, wav_image
    d = image
    cdef int sigma_0, h, w, size, j, d_px, px, sigma_div
    sigma_0 = 9

    wav_image = np.zeros_like(d) # create an empty image with the same dimensions
    h = d.shape[0]
    w = d.shape[1]
    size = h*w

    j = 0

    
    for y in range(0, h-3):         # for all pixels in y axis
        for x in range(0, w-3):     # for all pixels in x axis           

                                                                    # According to the formulas in Wu et al.
            d_px = d[y, x]                                          # Select current pixel
            local_area = d[range(y-3, y+3), range(x-3,x+3)]         # Select the 9 pixels around it from the CAI subtraction
            sigma_div = sigma_0/(calc_sigma(local_area) + sigma_0)  # get the estimated local variance for the pixel
            px = d_px * sigma_div                                   # multiply subtracted CAI with the local variances
            px= int(px)                                             # Estimated camera reference SPN
            wav_image[y,x]= px
            j += 1

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
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[starty:starty + cropy, startx:startx + cropx]


def filter(img, h, w):

    if os.path.isfile(img):
        original = cv2.imread(img, 0)                  # the 0 means read as grayscale
        plt.imsave('0_Original.png', original, cmap="gray") # cmap="gray" means save as grayscale

        if h == 0 or w == 0:
            h = original[0]
            w = original[1]

        average_orig = original.mean()    #average all the noise and add them 


        cropped = crop_center(original, h, w)


        cai_image = cai_filter(cropped)


        d_image = cv2.subtract(cai_image, cropped)


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
    size = len(onlyfiles)
    est_camera_ref = []
    for filename in os.listdir(folder):
        paths = folder+filename
        i = filter(paths, h, w)
        spn = get_spn(i)
        est_camera_ref.append(spn)
        images.append(i)
        kake = "filtered\\"
        path_create = folder + kake
        path_new = folder + kake + filename
        if not os.path.exists(path_create):
            os.makedirs(path_create)
        plt.imsave(path_new, i, cmap="gray" )
        during_time = time.time() - start_time
        progress = round((j / size) * 100, 1)
        progressbar = int(progress / 4)
        frogress = "{0:.2f}".format(progress)
        elapse = "{0:.2f}".format(during_time)
        print('\r|{}|{}% Time elapsed: {}'.format(("█" * progressbar), frogress, elapse), end="", flush=True)
        j += 1

    elapsed_time = time.time() - start_time
    print('\r|{}|{}%'.format(("█" * 25), 100), end="", flush=True)
    print("Time taken:", elapsed_time)
    print("Estimated camera reference:", est_camera_ref.mean())


parser = ArgumentParser()
parser.add_argument("path", help="python filter_cy PATHHERE height width")
parser.add_argument("height", help="python filter_cy PATHHERE height width")
parser.add_argument("width", help="python filter_cy PATHHERE height width")

args = parser.parse_args()
filter_main(args.path, args.height, args.width)