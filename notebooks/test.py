
import sys
import numpy as np
from scipy import ndimage
from scipy.misc import imsave


im = ndimage.imread(sys.argv[1], mode='L')

if 0:
    sx = ndimage.sobel(im, axis=0, mode='constant')
    sy = ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)

    imsave('sobel.jpg', sob)

if 0:
    hist, bin_edges = np.histogram(im, bins=60)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    print bin_edges
    #binary_img = im > 0.1
    #imsave('binary_img.jpg', binary_img)

if 1:
    label_im, nb_labels = ndimage.label(im)
    print nb_labels
    #sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    #mean_vals = ndimage.sum(im, label_im, range(1, nb_labels + 1))
