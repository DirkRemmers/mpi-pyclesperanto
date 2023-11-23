"""This is an example script on we can do some analysis on an example image.

To trigger this script without multiprocessing via the terminal, use the following command:

python example.py
"""

import pyclesperanto_prototype as cle
from skimage.io import imread
import numpy as np
from datetime import datetime

# define some analysis stuff on the gpu
def some_analysis_stuff(image:np.ndarray)->np.ndarray:

    # get the individual channels
    nuclei = image[:,:,2]
    cells = image[:,:,1]

    # do something in the gpu, don't mind the actual quality of the analysis ;) 
    nuclei_gpu = cle.push(nuclei)
    nuclei_labels_gpu = cle.voronoi_otsu_labeling(nuclei_gpu, spot_sigma=10, outline_sigma=2)
    nuclei_labels = cle.pull(nuclei_labels_gpu)

    del nuclei_gpu, nuclei_labels_gpu

    cells_gpu = cle.push(cells)
    cells_labels_gpu = cle.voronoi_otsu_labeling(cells_gpu, spot_sigma=25, outline_sigma=2)
    cells_labels = cle.pull(cells_labels_gpu)

    del cells_gpu, cells_labels_gpu

    return nuclei_labels, cells_labels

# for time tracking purposes
start = datetime.now()

# select a GPU for pyclesperanto
cle.select_device("RTX") # change to the correct GPU type

# load in an image
image = imread("example-image.tiff")

# crop into 4 tiles
new_height = int(image.shape[0]/2)
new_width = int(image.shape[1]/2)
tile_1 = image[0:new_height, 0:new_width, :]
tile_2 = image[0:new_height, new_width:new_width*2, :]
tile_3 = image[new_height:new_height*2, 0:new_width, :]
tile_4 = image[new_height:new_height*2, new_width:new_width*2, :]

# collect in tiles, and prepare the empty tile
tiles = [tile_1, tile_2, tile_3, tile_4]

# process the images in parallel
for tile in tiles:
    nuclei_labels, cells_labels = some_analysis_stuff(image = tile)
    
# for time tracking purposes
end = datetime.now()

# print total time
print(f"Total time spend with a single process = {end - start}")

