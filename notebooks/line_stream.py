

def lines(image):
    # launch a path from each external pixel on the image
    # each path is a list of pixels that travel
    for x in xrange(image.shape[0]):
        paths.append([x, 0])
    #npaths = 2 * (image.shape[0] + image.shape[1])
    #for x in xrange(image.shape[0]):
        #paths.append([x, 0])
        #paths.append([x, image.shape[1]])
    #for y in xrange(image.shape[2]):
        #paths.append([0, y])
        #paths.append([image.shape[0], y])

    #paths = [[] for x in xrange(npaths)]
    #ITERS = image.shape[0]
    for x in xrange(ITERS):



