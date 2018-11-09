from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def cai_filter(image):

    # grab the image dimensions
    cai_image = np.zeros_like(image)
    h = image.shape[0]
    w = image.shape[1]
    print(h, "x", w)
    print(h*w, "pixels")
    cai_diff = np.zeros_like(image)

    print("Starting loop")
    i = 0
    j = 0

    h = int(h)
    w = int(w)

    for y in range(0, h):
        for x in range(0, w):
            try:
                n = image[y - 1, x]
                e = image[y, x + 1]
                s = image[y + 1, x]
                w = image[y, x - 1]
            except IndexError:
                n = image[y, x]
                e = image[y, x]
                s = image[y, x]
                w = image[y, x]

            cai_array = [n, s, e, w]

            n = int(n)
            e = int(e)
            s = int(s)
            w = int(w)
            if (np.max(cai_array) - np.min(cai_array)) <= 20:
                px = np.mean(cai_array)
            elif (np.absolute(e - w) - np.absolute(n - s)) > 20:
                px = (n + s) / 2
            elif (np.absolute(n - s) - np.absolute(e - w)) > 20:
                px = (e + w) / 2
            else:
                px = np.median(cai_array)

            px = int(px)


            if image[y, x] != int(px):
                cai_diff[y, x] = int(px)
                i += 1
            j += 1
            cai_image[y, x] = px

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
    plt.imsave('CAI_diff.png', cai_diff, cmap="gray")
    return cai_image

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
        original = cv2.imread(img_path, 0)
        plt.imsave('Original.png', original, cmap="gray")
        cropped = crop_center(original, 500, 500)
        plt.imsave('Cropped.png', cropped, cmap="gray")        
        cai = cai_filter(cropped)
        plt.imsave('CAI.png', cai, cmap="gray")
        subtract_cai = cv2.subtract(cropped, cai)
        plt.imsave('subtracted_cai.png', subtract_cai, cmap="gray")
    else:
        print("file does not exist")
