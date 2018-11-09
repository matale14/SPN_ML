from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.restoration import estimate_sigma
import os


def cai_filter(image):

    cai_image = np.zeros_like(image) # create an empty image with the same dimensions
    h = image.shape[0]               # Find out heigh x width for the loop
    w = image.shape[1]
    print(h, "x", w)
    print(h*w, "pixels")
    size = h*w
    cai_diff = np.zeros_like(image)  # Another clone just for the affected pixels

    print("Starting loop")
    i = 0
    j = 0

    h = int(h)
    w = int(w)
    for y in range(0, 4032):         # for all pixels in y axis
        for x in range(0, 3024):     # for all pixels in x axis

            progress = round((j / size) * 100, 1)
            progressbar = int(progress / 4)
            print('\r|{}|{}%'.format(("█" * progressbar), progress), end="", flush=True)
            
            try:
                n = image[y - 1, x] # North, east, south, west pixels
                e = image[y, x + 1]
                s = image[y + 1, x]
                w = image[y, x - 1]
            except IndexError:
                n = image[y, x]
                e = image[y, x]
                s = image[y, x]
                w = image[y, x]

            cai_array = [n, e, s, w]

            n = int(n)
            e = int(e)
            s = int(s)
            w = int(w)
            if (np.max(cai_array) - np.min(cai_array)) <= 20:       # If the max number of the neighbouring pixels are less than or equal to
                px = np.mean(cai_array)                             # 20 in value(0-255) then just set the pixel to the mean
            elif (np.absolute(e - w) - np.absolute(n - s)) > 20:    # If the absolute value(not negative. F.ex. -5 = 5) of that is more than 20
                px = (n + s) / 2                                    # the value is northern + southern pixel divided by 2
            elif (np.absolute(n - s) - np.absolute(e - w)) > 20:
                px = (e + w) / 2
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

    print(i, "Values changed out of:", j)
    print("CAI complete")
    plt.imsave('99_CAI_diff.png', cai_diff, cmap="gray")
    return cai_image

def calc_sigma(image):
    d = image
    m = 3
    sigma_0 = 9
    sigmage = np.zeros_like(d)
    h = d.shape[0]               # Find out heigh x width for the loop
    w = d.shape[1]
    size = h*w

    print("Calculating sigma image")
    i = 0
    j = 0

    h = int(h)
    w = int(w)
    
    for y in range(0, 4032):         # for all pixels in y axis
        for x in range(0, 3024):     # for all pixels in x axis

            progress = round((j / size) * 100, 1)
            progressbar = int(progress / 4)
            print('\r|{}|{}%'.format(("█" * progressbar), progress), end="", flush=True)

            #Sigsum here. Need to read up more on it from source[3] in Wu et al.
            #http://ws.binghamton.edu/fridrich/Research/double.pdf

            #for now using est_sigma() which uses estimate_sigma() from skimage
            #that uses D. L. Donoho and I. M. Johnstone. “Ideal spatial adaptation by wavelet shrinkage.” 
            #Biometrika 81.3 (1994): 425-455. DOI:10.1093/biomet/81.3.425
            #http://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.estimate_sigma

            sigmas=(0, sigsum)
            px = max(sigmas)
            sigmage[y, x] = px

    return sigmage

def est_sigma(image):

    sigma_est = estimate_sigma(image, multichannel=True, average_sigmas=True)
    return sigma_est

def wavelet(dimage, sigmage):
    d = dimage
    sigma = sigmage
    sigma_0 = 9

    wav_image = np.zeros_like(d) # create an empty image with the same dimensions
    h = d.shape[0]               # Find out heigh x width for the loop
    w = d.shape[1]
    size = h*w

    print("Wavelet transform")
    i = 0
    j = 0

    h = int(h)
    w = int(w)
    
    for y in range(0, 4032):         # for all pixels in y axis
        for x in range(0, 3024):     # for all pixels in x axis

            progress = round((j / size) * 100, 1)
            progressbar = int(progress / 4)
            print('\r|{}|{}%'.format(("█" * progressbar), progress), end="", flush=True)
            
            d_px = d[y, x]          
            sigma_div = sigma_0/(sigma + sigma_0) # According to the formulas in Wu et al.
            px = d_px * sigma_div                       # link here later(Or check wiki)
            wav_image[y, x] = px

    return wav_image

def get_spn(wav_image):

    average = wav_image[wav_image!=0].mean()    #average all the noise and add them
    spn = average + wav_image
    return spn


def crop_center(img, cropx, cropy):
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
    img_path = 'example.jpg'
    if os.path.isfile(img_path):
        original = cv2.imread(img_path, 0)                  # the 0 means read as grayscale
        plt.imsave('0_Original.png', original, cmap="gray") # cmap="gray" means save as grayscale

        cropped = crop_center(original, 0, 0)
        plt.imsave('1_Cropped.png', cropped, cmap="gray") 

        cai = cai_filter(cropped)
        plt.imsave('2_CAI.png', cai, cmap="gray")

        d_image = cv2.subtract(cai, cropped)
        plt.imsave('3_D_image.png', d_image, cmap="gray")

        sig_image = est_sigma(d_image)
        plt.imsave('4_Sig_image.png', sig_image, cmap="gray")

        #sig_image = calc_sigma(d_image)
        #plt.imsave('4_Sig_image.png', sig_image, cmap="gray")

        wav_image = wavelet(sig_image, d_image)
        plt.imsave('5_Wav_image.png', wav_image, cmap="gray")

        spn_image = get_spn(wav_image, d_image)
        plt.imsave('6_SPN_image.png', spn_image, cmap="gray")
    else:
        print("file does not exist")
