""" docstring """
import cv2
import numpy as np
from statistics import mean
from math import floor


def get_pixel_hue(r, g, b):
    # TODO: fix
    # RuntimeWarning: invalid value encountered in double_scalars
    r /= 256.0
    g /= 256.0
    b /= 256.0
    mini, maxi = min(r, g, b), max(r, g, b)
    hue = 0.0

    if mini != maxi:
        if maxi == r:
            hue = ((g - b) * 60.0) / (maxi - mini)
        elif maxi == g:
            hue = (2 + (b-r) * 60.0) / (maxi - mini)
        elif maxi == b:
            hue = (4 + (r-g) * 60.0) / (maxi - mini)

    if hue > 0:
        return floor(hue)
    else:
        return floor(360 - hue)


def get_pixel_chr(r, g, b):
    return max(r, g, b) - min(r, g, b)    # chroma


def get_pixel_lum(r, g, b):         # luminance
    return r*0.3 + g*0.59 + b*0.11


def get_pixel_bri(r, g, b):         # brightness
    return round(mean([r, g, b]))


def __get_sorted__(temp, mode, rev_status):
    new_rgb_vector = []
    if mode == 'L':
        for i in range(0, np.shape(temp)[0]):
            new_rgb_vector.append(get_pixel_lum(temp[i][0], temp[i][1], temp[i][2]))
    elif mode == 'C':
        for i in range(0, np.shape(temp)[0]):
            new_rgb_vector.append(get_pixel_chr(temp[i][0], temp[i][1], temp[i][2]))
    elif mode == 'H':
        for i in range(0, np.shape(temp)[0]):
            new_rgb_vector.append(get_pixel_hue(temp[i][0], temp[i][1], temp[i][2]))
    elif mode == 'B':
        for i in range(0, np.shape(temp)[0]):
            new_rgb_vector.append(get_pixel_bri(temp[i][0], temp[i][1], temp[i][2]))
    return [rgb for sort_criteria, rgb in sorted(zip(new_rgb_vector, temp), reverse=rev_status)]


class ImageFeatureVector(object):
    """ docstring """
    def __init__(self, img_name, dest_img_path, sort_criteria, sort_mode, direction, reverse):
        self.img_name = img_name
        self.dest_img_path = dest_img_path
        self.sort_criteria = sort_criteria
        self.sort_mode = sort_mode
        self.direction = direction
        self.reverse = reverse
        self.pixel_data = None
        self.criteria_data = None
        self.img = None
        self.COLS = -1
        self.ROWS = -1
        self.r = []
        self.g = []
        self.b = []
        self.__process_img__()

    def get_color_channel(self):
        """ docstring """
        return self.r, self.g, self.b

    def __process_img__(self):
        """
        This is a helper method that is used to read in the data of the source image. This method
        gets all the pixel data of the source image to be edited.
        """
        self.img = cv2.imread(self.img_name)

        # Make sure we have an even number of rows and cols
        if np.shape(self.img)[0] % 2 == 1:
            self.img = np.delete(self.img, 0, axis=0)

        if np.shape(self.img)[1] % 2 == 1:
            self.img = np.delete(self.img, 0, axis=1)

        # If we're doing Vertical, rotate the image by 90 degrees
        if self.direction == 'V':
            self.img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)

        original_image = self.img.copy()
        original_image[:, :, 0] = self.img[:, :, 2]
        original_image[:, :, 2] = self.img[:, :, 0]
        self.COLS = self.get_no_cols()
        self.ROWS = self.get_no_rows()
        self.b, self.g, self.r = cv2.split(self.img)

        if self.sort_mode == 'S':
            for i in range(np.shape(self.b)[0]):
                zipped = list(zip(self.r[i, ...][:self.COLS], self.g[i, ...][:self.COLS], self.b[i, ...][:self.COLS]))
                temp = list(zipped[:])
                sorted_data = __get_sorted__(temp, self.sort_criteria, self.reverse)

                self.r[i, ...][:self.COLS] = np.array([r for r, g, b in sorted_data])
                self.g[i, ...][:self.COLS] = np.array([g for r, g, b in sorted_data])
                self.b[i, ...][:self.COLS] = np.array([b for r, g, b in sorted_data])
        else:
            half_cols = int(self.COLS / 2)
            for i in range(np.shape(self.b)[0]):

                # Pull out the RGB values for the columns we're using (every other column)
                zipped = list(zip(self.r[i, ...][::2], self.g[i, ...][::2], self.b[i, ...][::2]))
                temp = list(zipped[:])

                # Sort the data
                sorted_data = __get_sorted__(temp, self.sort_criteria, self.reverse)

                # Reconstruct the pixels
                self.r[i, ...][:half_cols] = np.array([r for r, g, b in sorted_data])
                self.g[i, ...][:half_cols] = np.array([g for r, g, b in sorted_data])
                self.b[i, ...][:half_cols] = np.array([b for r, g, b in sorted_data])

                # The right hand side is the flip of the left hand side
                self.r[i, ...][half_cols:] = self.r[i, ...][:half_cols][::-1]
                self.g[i, ...][half_cols:] = self.g[i, ...][:half_cols][::-1]
                self.b[i, ...][half_cols:] = self.b[i, ...][:half_cols][::-1]

        # If we're doing Vertical, rotate the image back
        final_image = cv2.merge((self.b, self.g, self.r))

        if self.direction == 'V':
            final_image = cv2.rotate(final_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imwrite(self.dest_img_path, final_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def get_no_rows(self):
        return self.img.shape[0]

    def get_no_cols(self):
        return self.img.shape[1]

    def is_img_clr(self):
        return self.img.shape[2] != 0

    def get_criteria_data(self):
        return self.criteria_data

    def get_pixel_data(self):
        """
        get the data for individual pixels, the return type of this method is a length
        3 turple that has individual pixel RGB data. NOTE: these are actually the original
        pixel data of the image that was read from the source image to be edited
        """
        return self.pixel_data

    def get_img_destination_path(self):
        """
        returns the destination or path of the resulting image, that has been sorted.
        """
        return self.dest_img_path

    def get_image_name(self):
        """
        returns the original image name that was being sorted or that was being edited.
        """
        return self.img_name
