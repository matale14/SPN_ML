from __future__ import print_function
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, time
from PIL import Image
import scipy, scipy.ndimage

"""

Created by Alexander Mackenzie-Low

Based on:
Xiangui Kang, Jiansheng Chen, Kerui Lin and Peng Anjie. 2014. 
A context-adaptive SPN predictor for trustworthy source camera identification
https://jivp-eurasipjournals.springeropen.com/articles/10.1186/1687-5281-2014-19
"""


class Image:

    def __init__(self, img_path, filter_path, hw: list = [512, 512]):
        """

        Sends the image on its journey through all the filters.
        Returns the complete sensor pattern noise image.

        Args:
            img_path: string
            filter_path: string
            h: int
            w: int

            img_path is the full path to the image.
            filter_path is path to save
            h and w are height and width of the wanted cropping,
            leave 0 for full size.

        Variables:
            int: h, w
            unsigned char[:, :]: original, cropped, cai_image, d_image, wav_image

        Returns:
            unsigned char[:, :] / numpy 2d image array

        """
        self.folder_path = filter_path
        self.name = os.path.basename(img_path)
        self.image = cv2.imread(img_path, 0)
        h = hw[0]
        w = hw[1]
        self.cropped_image = self.crop(h, w)
        self.cai_image = self.cai_filter()
        self.d_image = cv2.subtract(self.cropped_image, self.cai_image)
        self.wavelet_image = self.wavelet()
        self.save_image()

    def crop(self, cropx: int = 512, cropy: int = 512):
        """

        Crop image to specifications. Filter takes a decent while, running on
        512x512 and smaller is reccomended. Returns a cropped image.

        Args:
            img: unsigned char[:, :] / numpy 2d image array
            cropy: int
            cropx: int

            img is the original grayscale image.
            The two crop parameters are the width and height.

        Variables:
            int: cropx, cropy, startx, starty
            unsigned char[:, :]: img

        Returns:
            unsigned char[:, :] / numpy 2d image array

        """
        if cropx == 0 or cropy == 0:
            return self.image
        else:
            y, x = self.image.shape

            startx = (x // 2) - (cropx // 2)
            starty = (y // 2) - (cropy // 2)
            return self.image[starty:starty + cropy, startx:startx + cropx]

    def cai_filter(self):
        """

        Compares every pixel in the image to those around it,
        and uses a cai formula to change the pixel value. returns
        a numpy 2d array of the filtered image.

        Args:
            image: unsigned char[:, :] / numpy 2d image array

            the cropped and grayscale image.

        Variables:
            unsigned char[:, :]: cai_image
            int: no, ea, so, we, x, y, i, j, h, w, size, px, ne, nw, se, sw
            char[8]: cai_array

        Returns:
            unsigned char[:, :] / numpy 2d image array

        """
        cai_image = np.zeros_like(self.cropped_image)  # create an empty image with the same dimensions
        img = self.cropped_image
        h = cai_image.shape[0]
        w = cai_image.shape[1]
        size = h * w

        i = 0

        for y in range(0, h - 1):  # for all pixels in y axis
            for x in range(0, w - 1):  # for all pixels in x axis

                no = int(img[y - 1, x])
                ne = int(img[y - 1, x + 1])
                nw = int(img[y - 1, x - 1])
                so = int(img[y + 1, x])
                se = int(img[y + 1, x + 1])
                sw = int(img[y + 1, x - 1])
                ea = int(img[y, x + 1])
                we = int(img[y, x - 1])

                cai_array = [nw, no, ne, we, ea, sw, so, se]

                if (np.max(cai_array) - np.min(
                        cai_array)) <= 20:  # If the max number of the neighbouring pixels are less than or equal to
                    px = np.mean(cai_array)  # 20 in value(0-255) then just set the pixel to the mean
                elif (np.absolute(ea - we) - np.absolute(
                        no - so)) > 20:  # If the absolute value(not negative. F.ex. -5 = 5) of that is more than 20
                    px = (no + so) / 2  # the value is northern + southern pixel divided by 2
                elif (np.absolute(no - so) - np.absolute(ea - we)) > 20:
                    px = (ea + we) / 2
                elif (np.absolute(ne - sw) - np.absolute(
                        se - nw)) > 20:  # If the absolute value(not negative. F.ex. -5 = 5) of that is more than 20
                    px = (se + nw) / 2  # the value is northern + southern pixel divided by 2
                elif (np.absolute(se - nw) - np.absolute(ne - sw)) > 20:
                    px = (ne + we) / 2
                else:
                    px = np.median(
                        cai_array)  # Median is backup. Median just selects the item that is most in the middle.

                px = int(px)

                cai_image[y, x] = px  # Set the value of the current pixel to px.

        return cai_image

    @staticmethod
    def calc_sigma(image):
        """

        Calculates the local variance of the pixel. A number that represents
        the difference in values of the surrounding pixels. Returns the
        local variance of the pixel. Used in the wavelet Wiener filter.

        Args:
            image: unsigned char[:, :] / numpy 2d image array

            The neighbourhood pixels of a certain pixel. In essence a small image
            centered on a certain pixel.

        Variables:
            int: m, sigma_0, sigmas, local_variance, sigsum, h, w

        Returns:
            int

        """
        d = image
        m = 3
        sigma_0 = 9
        h = d.shape[0]
        w = d.shape[1]
        # Wu et al.
        # http://ws.binghamton.edu/fridrich/Research/double.pdf
        # m = the neighbourhood pixels, here the 3x3 pixels around the selected pixel
        # sum the value of the neighbourhood - the overall variance of the SPN.
        # Select the max value, so if the value is negative, it returns a black(empty) pixel
        neigh = []
        for y in range(0, h):  # for all pixels in y axis
            for x in range(0, w):  # for all pixels in x axis
                the_sub = ((d[y, x]) ** 2) - (float(sigma_0))
                neigh.append(the_sub)

        sigsum = np.sum(neigh)

        sigmas = [0, ((1 / (m ** 2)) * sigsum)]

        local_variance = max(sigmas)
        return local_variance

    def wavelet(self):
        """

        A wavelet Wiener filter designed to detect edges and smooth
        them out. Returns the filtered image as a numpy 2d array.

        Args:
            image: unsigned char[:, :] / numpy 2d image array

            The cropped and grayscale image minus the CAI filtered image
            essentially leaving just the sensor pattern noise.

        Variables:
            int: sigma_0, h, w, size, j, d_px, px, sigma_div, no, ne, nw, so, se, sw, ea, we
            unsigned char[:, :]: d, wav_image, neighbour / numpy 2d image array

        Returns:
            unsigned char[:, :] / numpy 2d image array

        """
        d = self.d_image
        sigma_0 = 9

        wav_image = np.zeros_like(d)  # create an empty image with the same dimensions
        h = d.shape[0]
        w = d.shape[1]

        for y in range(0, h - 1):  # for all pixels in y axis
            for x in range(0, w - 1):  # for all pixels in x axis

                d_px = d[y, x]  # Select current pixel
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
                sigma_div = sigma_0 / (
                            self.calc_sigma(neighbour) + sigma_0)  # get the estimated local variance for the pixel
                px = d_px * sigma_div  # multiply subtracted CAI with the local variances
                px = int(px)  # Estimated camera reference SPN
                wav_image[y, x] = px

        return wav_image

    def save_image(self):
        """

        Saves the image to the correct path

        """
        plt.imsave(os.path.join(self.folder_path, self.name), self.wavelet_image, cmap="gray")
