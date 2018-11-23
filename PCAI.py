from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from argparse import ArgumentParser


def cai_filter(image, h, w):

    cai_image = np.zeros_like(image) # create an empty image with the same dimensions

    print(h, "x", w)
    print(h*w, "pixels")
    size = h*w
    cai_diff = np.zeros_like(image)  # Another clone just for the affected pixels

    print("Starting loop")
    i = 0
    j = 0

    for y in range(0, h):         # for all pixels in y axis For some reason, I have to specify the amounts
        for x in range(0, w):     # for all pixels in x axis The variables above changes for some reason

            progress = round((j / size) * 100, 1)
            progressbar = int(progress / 4)
            print('\r|{}|{}%'.format(("█" * progressbar), progress), end="", flush=True)
            
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

            n = int(n)
            e = int(e)
            s = int(s)
            we = int(we)

            if (np.max(cai_array) - np.min(cai_array)) <= 20:       # If the max number of the neighbouring pixels are less than or equal to
                px = np.mean(cai_array)                             # 20 in value(0-255) then just set the pixel to the mean
            elif (np.absolute(e - we) - np.absolute(n - s)) > 20:    # If the absolute value(not negative. F.ex. -5 = 5) of that is more than 20
                px = (n + s) / 2                                    # the value is northern + southern pixel divided by 2
            elif (np.absolute(n - s) - np.absolute(e - we)) > 20:
                px = (e + we) / 2
            else:
                px = np.median(cai_array)                           # Median is backup. Median just selects the item that is most in the middle.

            px = int(px)


            if image[y, x] != int(px):      # If a pixel changes value, count it and apply in the cai_diff image
                cai_diff[y, x] = int(px)
                i += 1
            j += 1
            cai_image[y, x] = px            # Set the value of the current pixel to px. 

            """
            Debug prints
            
            print("Array:", cai_array)
            print("mean:", np.mean(cai_array))
            print("max - min:", np.max(cai_array) - np.min(cai_array))
            print("e-w - n-s:", np.absolute(e - w) - np.absolute(n - s))
            print("n-s - e-w:", np.absolute(n - s) - np.absolute(e - w))
            print("median:", np.median(cai_array))
            """
    print()
    print(i, "Values changed out of:", j)
    print("CAI complete")
    plt.imsave('99_CAI_diff.png', cai_diff, cmap="gray")
    return cai_image

def calc_sigma(image):
    d = image
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

def wavelet(dimage, h, w):
    d = dimage
    sigma_0 = 9

    wav_image = np.zeros_like(d) # create an empty image with the same dimensions

    size = h*w

    print("Wavelet transform")
    j = 0

    
    for y in range(0, h-3):         # for all pixels in y axis
        for x in range(0, w-3):     # for all pixels in x axis

            progress = round((j / size) * 100, 1)
            progressbar = int(progress / 4)
            print('\r|{}|{}%'.format(("█" * progressbar), progress), end="", flush=True)


                                                                    # According to the formulas in Wu et al.
            d_px = d[y, x]                                          # Select current pixel
            local_area = d[range(y-3, y+3), range(x-3,x+3)]         # Select the 9 pixels around it from the CAI subtraction
            sigma_div = sigma_0/(calc_sigma(local_area) + sigma_0)  # get the estimated local variance for the pixel
            px = d_px * sigma_div                                   # multiply subtracted CAI with the local variances
            px= int(px)                                             # Estimated camera reference SPN
            wav_image[y,x]= px
            j += 1

    print('\r|{}|{}%'.format(("█" * 25), "100.0"), end="", flush=True)
    print()
    return wav_image

def get_spn(wav_image):

    average = wav_image[wav_image!=0].mean()    #average all the noise and add them 
    return average

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


if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("file_path", help="Select file to filter")
    parser.add_argument("height", help="Select height of crop(0 to not crop)")
    parser.add_argument("width", help="Select width of crop(0 to not crop)")
    args = parser.parse_args()

    img_path = args.file_path
    if os.path.isfile(img_path):
        original = cv2.imread(img_path, 0)                  # the 0 means read as grayscale
        plt.imsave('0_Original.png', original, cmap="gray") # cmap="gray" means save as grayscale

        #h = original.shape[0]
        #w = original.shape[1]
        h = int(args.height)
        w = int(args.width)

        if h == 0 or w == 0:
            h = original[0]
            w = original[1]

        average_orig = original.mean()    #average all the noise and add them 

        cropped = crop_center(original, h, w)
        plt.imsave('1_Cropped.png', cropped, cmap="gray") 

        cai_image = cai_filter(cropped, h, w)
        plt.imsave('2_CAI.png', cai_image, cmap="gray")

        d_image = cv2.subtract(cai_image, cropped)
        plt.imsave('3_D_image.png', d_image, cmap="gray")

        wav_image = wavelet(d_image, h, w)
        plt.imsave('4_Wav_image.png', wav_image, cmap="gray")

        spn = get_spn(wav_image)
        print("Sensor Pattern Noise reference number:", spn)


    else:
        print("file does not exist")
