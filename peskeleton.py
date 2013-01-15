"""PESkeleton.py - skeletonize a binary image using the Poisson Equation

"""

import numpy as np
import scipy.ndimage as scind

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.cpmath.cpmorphology as morph

class PESkeleton(cpm.CPModule):
    variable_revision_number = 1
    module_name = "PESkeleton"
    category = "Image Processing"
    
    def create_settings(self):
        self.input_image = cps.ImageNameSubscriber("Input image")
        self.output_image = cps.ImageNameProvider("Output image", "PESkeleton")
        
    def settings(self):
        return [self.input_image, self.output_image]
    
    def run(self, workspace):
        image = workspace.image_set.get_image(self.input_image.value,
                                              must_be_binary=True)
        pe = poisson_equation(image.pixel_data, convergence=.1, percentile=75)
        dt = scind.distance_transform_edt(image.pixel_data)
        skel = skeletonize(image.pixel_data, image.mask, pe*dt)
        output = cpi.Image(skel, image.mask, parent_image=image)
        workspace.image_set.add(self.output_image.value, output)
        workspace.display_data.input_image = image.pixel_data
        workspace.display_data.poisson_equation = pe
        workspace.display_data.output_image = skel
        
    def is_interactive(self):
        return False
    
    def display(self, workspace):
        figure = workspace.create_or_find_figure(subplots=(2, 1))
        img = workspace.display_data.input_image
        skel = workspace.display_data.output_image
        pe = workspace.display_data.poisson_equation
        cimg = np.zeros((img.shape[0], img.shape[1], 3))
        cimg[~img, :] = 1
        cimg[skel, 1] = 1
        title = "Skeleton of %s" % self.input_image.value
        ax = figure.subplot_imshow_color(0, 0, cimg, title = title)
        figure.subplot_imshow(1, 0, pe, title="Solution to Poisson Equation",
                              sharex = ax, sharey = ax)
        
        
def skeletonize(image, mask=None, distance=None):
    '''Skeletonize the image
    
    Take the distance transform.
    Order the 1 points by the distance transform.
    Remove a point if it has more than 1 neighbor and if removing it
    does not change the Euler number.
    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    #
    # Lookup table - start with only positive pixels.
    # Keep if # pixels in neighborhood is 2 or less
    # Keep if removing the pixel results in a different connectivity
    #
    table = (morph.make_table(True,np.array([[0,0,0],[0,1,0],[0,0,0]],bool),
                              np.array([[0,0,0],[0,1,0],[0,0,0]],bool)) &
             (np.array([scind.label(morph.pattern_of(index), morph.eight_connect)[1] !=
                        scind.label(morph.pattern_of(index & ~ 2**4),
                                    morph.eight_connect)[1]
                        for index in range(512) ]) |
              np.array([np.sum(morph.pattern_of(index))<3 for index in range(512)])))
    
    if distance is None:
        distance = scind.distance_transform_edt(masked_image)
    #
    # The processing order along the edge is critical to the shape of the
    # resulting skeleton: if you process a corner first, that corner will
    # be eroded and the skeleton will miss the arm from that corner. Pixels
    # with fewer neighbors are more "cornery" and should be processed last.
    #
    cornerness_table = np.array([9-np.sum(morph.pattern_of(index))
                                 for index in range(512)])
    corner_score = morph.table_lookup(masked_image, cornerness_table, False,1)
    i,j = np.mgrid[0:image.shape[0],0:image.shape[1]]
    result=masked_image.copy()
    distance = distance[result]
    i = np.ascontiguousarray(i[result],np.int32)
    j = np.ascontiguousarray(j[result],np.int32)
    result=np.ascontiguousarray(result,np.uint8)
    #
    # We use a random # for tiebreaking. Assign each pixel in the image a
    # predictable, random # so that masking doesn't affect arbitrary choices
    # of skeletons
    #
    r = np.random.RandomState()
    r.seed(0)
    tiebreaker=r.permutation(np.arange(np.product(masked_image.shape)))
    tiebreaker.shape=masked_image.shape
    order = np.lexsort((tiebreaker[masked_image],
                        corner_score[masked_image],
                        distance))
    order = np.ascontiguousarray(order, np.int32)
    table = np.ascontiguousarray(table, np.uint8)
    morph.skeletonize_loop(result, i, j, order, table)
    
    result = result.astype(bool)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

def poisson_equation(image, gradient=1, max_iter=100, convergence=.01, percentile = 90.0):
    '''Estimate the solution to the Poisson Equation
    
    The Poisson Equation is the solution to gradient(x) = h^2/4 and, in this
    context, we use a boundary condition where x is zero for background
    pixels. Also, we set h^2/4 = 1 to indicate that each pixel is a distance
    of 1 from its neighbors.
    
    The estimation exits after max_iter iterations or if the given percentile
    of foreground pixels differ by less than the convergence fraction
    from one pass to the next.
    
    Some ideas taken from Gorelick, "Shape representation and classification
    using the Poisson Equation", IEEE Transactions on Pattern Analysis and
    Machine Intelligence V28, # 12, 2006
    
    image - binary image with foreground as True
    gradient - the target gradient between 4-adjacent pixels
    max_iter - maximum # of iterations at a given level
    convergence - target fractional difference between values from previous 
                  and next pass
    percentile - measure convergence at this percentile
    '''
    # Evaluate the poisson equation with zero-padded boundaries
    pe = np.zeros((image.shape[0]+2, image.shape[1]+2))
    if image.shape[0] > 64 and image.shape[1] > 64:
        #
        # Sub-sample to get seed values
        #
        sub_image = image[::2, ::2]
        sub_pe = poisson_equation(sub_image, 
                                  gradient=gradient*2,
                                  max_iter=max_iter,
                                  convergence=convergence)
        coordinates = np.mgrid[0:(sub_pe.shape[0]*2),
                               0:(sub_pe.shape[1]*2)].astype(float) / 2
        pe[1:(sub_image.shape[0]*2+1), 1:(sub_image.shape[1]*2+1)] = \
            scind.map_coordinates(sub_pe, coordinates)
        pe[~image] = 0
    else:
        pe[1:-1,1:-1] = image
    #
    # evaluate only at i and j within the foreground
    #
    i, j = np.mgrid[0:pe.shape[0], 0:pe.shape[1]]
    mask = (i>0) & (i<pe.shape[0]-1) & (j>0) & (j<pe.shape[1]-1)
    mask[mask] = image[i[mask]-1, j[mask]-1]
    i = i[mask]
    j = j[mask]
    for itr in range(max_iter):
        next_pe = (pe[i+1, j] + pe[i-1, j] + pe[i, j+1] + pe[i, j-1]) / 4 + 1
        difference = np.abs((pe[mask] - next_pe) / next_pe)
        pe[mask] = next_pe
        if np.percentile(difference, percentile) <= convergence:
            break
    return pe[1:-1, 1:-1]

