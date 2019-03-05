from __future__ import print_function
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, time
from PIL import Image
import scipy, scipy.ndimage

class Image:

    def __init__(self, img_path, hw = [512, 512]):
        self.folder_path = os.path.dirname(img_path)
        self.name = os.path.basename(img_path)
        self.image = cv2.imread(img_path, 0)
        h = hw[0]
        w = hw[1]
        self.cropped_image = self.crop(h, w)
        self.cai_image = self.cai_filter()
        self.d_image = cv2.subtract(self.cropped_image, self.cai_image)
        self.wavelet_image = self.wavelet()
        self.save_image()


    def crop(self, cropx = 512, cropy = 512):
        if cropx == 0 or cropy == 0:
            return self.image
        else:
            y, x = self.image.shape

            startx = (x // 2) - (cropx // 2)
            starty = (y // 2) - (cropy // 2)
            return self.image[starty:starty + cropy, startx:startx + cropx]

    def cai_filter(self):

        cai_image = np.zeros_like(self.cropped_image) # create an empty image with the same dimensions
        img = self.cropped_image
        h = cai_image.shape[0]
        w = cai_image.shape[1]
        size = h*w

        i = 0

        for y in range(0, h-1):         # for all pixels in y axis
            for x in range(0, w-1):     # for all pixels in x axis
                
                no = int(img[y - 1, x])
                ne = int(img[y - 1, x + 1])
                nw = int(img[y - 1, x - 1])
                so = int(img[y + 1, x])
                se = int(img[y + 1, x + 1])
                sw = int(img[y + 1, x - 1])
                ea = int(img[y, x + 1])
                we = int(img[y, x - 1])

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

    @staticmethod
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

    def wavelet(self):
        d = self.d_image
        sigma_0 = 9

        wav_image = np.zeros_like(d) # create an empty image with the same dimensions
        h = d.shape[0]
        w = d.shape[1]


        
        for y in range(0, h-1):         # for all pixels in y axis
            for x in range(0, w-1):     # for all pixels in x axis

                d_px = d[y, x]          # Select current pixel
                                        # Select the 9 pixels around current pixel from the CAI subtraction            
                no = int(self.d_image[y - 1, x])
                ne = int(self.d_image[y - 1, x + 1])
                nw = int(self.d_image[y - 1, x - 1])
                so = int(self.d_image[y + 1, x])
                se = int(self.d_image[y + 1, x + 1])
                sw = int(self.d_image[y + 1, x - 1])
                ea = int(self.d_image[y, x + 1])
                we = int(self.d_image[y, x - 1])
                neighbour = np.array([[nw, no, ne],
                                      [we, d_px, ea],
                                      [sw, so, se]], dtype=np.float)
                                                                        # According to the formulas in Wu et al.
                sigma_div = sigma_0/(self.calc_sigma(neighbour) + sigma_0)   # get the estimated local variance for the pixel
                px = d_px * sigma_div                                   # multiply subtracted CAI with the local variances
                px = int(px)                                            # Estimated camera reference SPN
                wav_image[y,x] = px

        return wav_image

    def save_image(self):
        path_create = os.path.join(self.folder_path, 'filtered')
        save_path = os.path.join(path_create, self.name)
        if not os.path.exists(path_create):
            os.makedirs(path_create)

        plt.imsave(save_path, self.wavelet_image, cmap="gray")
