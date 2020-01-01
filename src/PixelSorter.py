''' docstring '''
from ImageFeatureVector import ImageFeatureVector as IFV
from sys import (argv, exit)

class PixelSorter(object):
    ''' docstring '''
    def __init__(self, src_img, dest_img, sort_criteria, sort_mode):
        self.src_img = src_img
        self.dest_img = dest_img
        self.sort_criteria = sort_criteria
        self.sort_mode = sort_mode

        if not self.__validate_args__():
            exit('USAGE: python PixelSorter.py <source-img-path> <dest-img-path> <sort-criteria> <sort-mode>')

    def __validate_args__(self):

        valid = True
        ''' docstring '''
        if not (self.sort_criteria in ['C', 'L', 'H', 'B']):
            valid = False

        if not (self.sort_mode in ['S', 'M']):
            valid = False

        return valid

    def pixel_sorter_middleware(self):
        ''' docstring '''
        IFV(self.src_img, self.dest_img, self.sort_criteria, self.sort_mode)

if __name__ == '__main__':
    if len(argv) == 5:
        window = PixelSorter(argv[1], argv[2], argv[3], argv[4])
        window.pixel_sorter_middleware()
    else:
        exit('USAGE: python PixelSorter.py <source-img-path> <dest-img-path> <sort-criteria> <sort-mode>')
