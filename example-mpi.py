"""This is an example script on how pyclesperanto-prototype can be used together with MPI4py.

To trigger this script on multiple ranks (3) via the terminal, use the following command:

mpiexec -n 4 python -m mpi4py example-mpi.py
"""

import pyclesperanto_prototype as cle
from mpi4py import MPI
from skimage.io import imread
import numpy as np
from datetime import datetime
from time import sleep

# define some analysis stuff on the gpu
def some_analysis_stuff(image:np.ndarray, extra_wait_time:int)->np.ndarray:

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

    # add some extra waiting time to showcase time differences
    # in reality this will be taken up by more advanced image-analysis functionality
    sleep(extra_wait_time)

    return nuclei_labels, cells_labels


# for time tracking purposes
start = datetime.now()

# select a GPU for pyclesperanto
cle.select_device("MX250") # change to the correct GPU type

# initialize the mpi settings
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.size

# load in an image with the first rank
if rank == 0:
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
else:
    tiles = None

# wait until all ranks are done with their job
comm.barrier()

# give each rank their tile
tiles = comm.bcast(tiles, root=0)
tile = tiles[rank]

# process the images in parallel
nuclei_labels, cells_labels = some_analysis_stuff(image = tile, extra_wait_time=2)

# wait until all ranks are done with their job
comm.barrier()

# for time tracking purposes
end = datetime.now()

# print total time
if rank == 0:
    print(f"Total time spend with MPI = {end - start}")