# gets information from the camera and gives back line information
#

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.cluster.vq import kmeans
from scipy.signal import argrelextrema
from scipy import ndimage
import time

def im_histogram(im):
    """ calculate histogram for an 8 bit image, returns a 256 numpy array """
    hist, edges = np.histogram(im, bins=np.arange(256))
    return hist

def calculate_colors(im, lpf_size=10, lpf_passes=2, plot=None):
    """get an image and return it quantified using the more frequent colors.
    In order to get the local peaks in color histogram and cluster colors around them.
    Then the image is quantified using only those colors
    """

    t0 = time.time()
    hist = im_histogram(im)
    t1 = time.time()
    print ("histogram time %f" % (t1 - t0))

    t0 = time.time()
    # low pass filter so local peak algorithm works better
    w = np.hanning(lpf_size)
    for x in range(lpf_passes):
        hist = np.convolve(w/w.sum(), hist)

    h = np.column_stack((np.linspace(0, hist.max(), hist.size), hist))
    t1 = time.time()
    print ("lpf time %f" % (t1 - t0))

    # calculate local peaks
    t0 = time.time()
    maxs = argrelextrema(h[:, 1], np.greater, order=10)[0]
    # after getting all the peaks in the histogram lets cluster all colors in two
    # this is a way to remove background
    maxs = kmeans(maxs.astype(np.float), 2)[0].astype(np.int)
    t1 = time.time()
    print ("local peaks time %f" % (t1 - t0))
    if plot:
        plot.figure(figsize=(13,7))
        plot.subplot(121)
        plot.bar(range(0, hist.size), h[:,1])
        plot.bar(maxs, np.ones(maxs.size)*hist.max())
        plot.subplot(122)

    # find the nearest peak color for each color based on max
    def mm(x):
        idx = 0
        m = 25500
        for i, v in enumerate(maxs):
            if abs(v - x) < m: 
                idx = i
                m = abs(v - x)
        return idx

    t0 = time.time()
    # create the lookup table
    lookup = np.vectorize(mm)(range(hist.size))
    # then create a quantified version of the image
    paletted = np.vectorize(lambda x: lookup[x])(im)
    if plot:
        plot.imshow(paletted)
    t1 = time.time()
    print ("vectorize time %f" % (t1 - t0))

    # calcualte pixels sums
    t0 = time.time()
    sums = np.zeros(maxs.size)
    for i, x in enumerate(lookup):
        sums[x] += hist[i]
    t1 = time.time()
    print ("pixels sums time %f" % (t1 - t0))

    # sort peaks by the number of colors (likely higher frequencies can be discarded)
    a = np.column_stack((range(0, maxs.size), maxs , hist[maxs], sums, sums/paletted.size))
    maximums = a[a[:, 3].argsort()]

    return maximums, paletted

def extract_lines_from_image(image, color_index, percent_discard=0.3, plot=None):
    """given an image and the color index to use calculate lines
    """
    x, y = np.where(image == color_index)
    X = np.column_stack((x, y)) 
    # if the number of points is a important part of the image, discard (likely background)
    if X.size == 0 or X.shape[0] > image.size * percent_discard:
        return []
    print ("dbscan data size %d " % X.shape[0])
    try:
        t0 = time.time()
        db = DBSCAN(eps=5, min_samples=10).fit(X)
        t1 = time.time()
        print ("   db scan time %f" % (t1 - t0))
    except ValueError:
        print(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    coords = []
    for c in set(labels):
        # label -1 means "noise"
        if c != -1:
            class_member_mask = (labels == c)
            coords.append(X[class_member_mask])
    # with the clusters look for the polynomial that fits better (well, it's a LR)
    pol = []
    errors = []
    t0 = time.time()
    for x in coords:
      try:
          pfd = np.polyfit(x[:,0], x[:, 1], 1)
          # compute varianze
          weight = float(x.shape[0])/image.size
          # discard small pieces
          if weight > 0.000005:
            xx = np.poly1d(pfd)
            err = (xx(x[:,0]) - x[:,1]).var()
            bbox = [x[:,0].min(), x[:,0].max(),x[:,1].min(), x[:,1].max()]
            # create line segment, like [x0, y0, x1, y1]
            segment = np.array((xx(bbox[2]), bbox[2], xx(bbox[3]), bbox[3]))
            pol.append((x, pfd, err, segment, weight, bbox))
      except ValueError:
          print(x)
      t1 = time.time()


    if plot:
        for c, po, err, segment, weight, bbox in pol:
           xp = np.linspace(bbox[0], bbox[1] , 100)
           p = np.poly1d(po)
           _ = plot.plot(c[:,1], c[:, 0], '.', p(xp), xp, '-')
    return pol



def extract_lines(im, plot=None):
    """extract lines from a image
    it returns a list of segments found in the image, like
    [
        [x0, y0, x1, y1],
        ...
    ]
    """
    t0 = time.time()
    maximums, paletted = calculate_colors(im)
    t1 = time.time()
    print ("calculate colors time %f" % (t1 - t0))
    #ax = None
    lines = []
    ax = plot
    final_image = ndimage.zoom(paletted, 0.3)
    for i, x in enumerate(maximums[:,0]):
        #if plot and not ax:
        #    fig, ax = plot.subplots()
        if ax:
            ax.imshow(final_image)
            ax.autoscale(False)
        t0 = time.time()
        pols = extract_lines_from_image(final_image, x, plot=ax)
        for p in pols:
            # the coordinate system is reversed, coordinate 0 is X and Y is 1, so reverse it so
            # the following operations with this segments don't drive me crazy
            # also normalize by the image size so next calcs can be done without the needing to know the bounds
            lines.append(np.array([
                [p[3][1]/final_image.shape[1], p[3][0]/final_image.shape[0]],
                [p[3][3]/final_image.shape[1], p[3][2]/final_image.shape[0]]
            ]))
        t1 = time.time()
        print ("extract lines from image time %f" % (t1 - t0))
    return lines

def calculate_direction(lines):
    """
    based on the lines extracted from the image this function calculates the real direccion the car needs to turn
    The algortihm is really simple, get the lines (weighting by closeness and line lenght) and calculate the center point
    higher values of Y is closer
    """
    
    # calculate line weight based on distance
    # find the total distance to weight based on distance
    total_length = 0
    for x in lines:
        v = x[1] - x[0]
        length = np.hypot(v[1], v[0])
        total_length += length

    # calculate centroid for lines
    centroid = np.array([0.0, 0.0])
    for x in lines:
        # modulate centroid by lenght and distance to 
        v = x[1] - x[0]
        length = np.hypot(v[1], v[0])
        line_centroid = 0.5 * (x[1] + x[0])
        print("line %f %f -> %f %f" % (x[0][0], x[0][1], x[1][0], x[1][1]))
        print("line centroid %f %f" % (line_centroid[0], line_centroid[1]))
        #distance_weigth = 1.0 - 10**(-3*line_centroid[1])
        distance_weigth = 1.0
        length_weigth = length/total_length
        centroid += line_centroid * distance_weigth * length_weigth
    return centroid
        




