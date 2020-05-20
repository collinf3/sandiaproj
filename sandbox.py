#%%
globals().clear()
#%%
import os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import imageio
import faulthandler
import skimage

from skimage.filters import threshold_li
from skimage.morphology import erosion,binary_opening,reconstruction
from skimage.exposure import adjust_log
from skimage.restoration import denoise_tv_chambolle
from skimage.color import rgb2gray,gray2rgb
from skimage import img_as_float
from skimage import img_as_uint
from glob import glob
import matplotlib.pyplot as plt
import datetime

## my code
import skimage.data as data
import skimage.segmentation as seg
from skimage import filters
from skimage import draw
from skimage import color
from skimage import exposure
from skimage import io
import imageio
# from scipy.misc import imsave

## new code
# import pygalmesh
import meshio
import pymesh
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.draw import ellipsoid

# %%

image_whole = io.imread('/home/collinf3/Documents/sandia_temp/datasamples/oxi_gray8bit.tif')  # full data set
image = image_whole[0:10, 0:43, 0:199]  # [z,y,x] small section

## show single segmented slice
single1 = image[int(len(image)/2)]  # pull center slice
# single1 = image[8]  # pull specific slice
thresholded_slice = filters.threshold_li(single1)
segmented_bool_slice = single1 > thresholded_slice
single1_li_slice = binary_opening(segmented_bool_slice)

# %%
# todo learn advanced plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable
orig = gray2rgb(img_as_float(np.array(image[0]))).copy()
im_2d = img_as_float(np.array(single1_li_slice[0]))



# %%

## my code
from mpl_toolkits.axes_grid1 import make_axes_locatable

orig = gray2rgb(img_as_float(np.array(Image.open('orig_%d.tif' % id)))).copy()
im_2d = img_as_float(np.array(Image.open('thrsld_%d.tif' % id)))
im_3d = img_as_float(np.array(Image.open('thrsld_alt_0%d.tif' % id)))
red_mult = [1, 0, 0]
ax, grid = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True)
over_1 = orig.copy()
over_1[np.array(im_2d) > 0, :] *= red_mult
grid[0].imshow(over_1)
grid[0].set_title('2d Segmentation')
over_2 = orig.copy()
over_2[np.array(im_3d) > 0, :] *= red_mult
grid[1].imshow(over_2)
grid[1].set_title('3d Segmentation')
im = grid[2].imshow(im_3d - im_2d, cmap='magma')
grid[2].set_title('Diff')
divider = make_axes_locatable(grid[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)



# %%
########################################################################################

# Sequence of filters
def filter_stack(img):
    # Adjust exposure
    img = adjust_log(img, 1)
    # faulthandler.dump_traceback()
    print(str(datetime.datetime.now()) + ' adjust log done')
    # # Denoise
    img = denoise_tv_chambolle(img, weight=0.2, eps=0.001)
    # faulthandler.dump_traceback()
    print(str(datetime.datetime.now()) + ' denoise done')
    # Erode image to reduce roughness/small spots
    img = erosion(img)
    print(str(datetime.datetime.now()) + ' erosion done')
    # Reconstruction using h-level (increases phase contrast, minimizes contrast in parent phase, cuts down beam hardening)
    # Adaptive level choice (increases per-slice variability)
    h = 0.9 * img.min()
    # h = 0.3
    seed = img - h
    dilated = reconstruction(seed, img, method='dilation')
    # faulthandler.dump_traceback()
    print(str(datetime.datetime.now()) + ' reconstruction done')
    img -= dilated
    # Renormalize array
    img *= 1. / img.max()
    print(str(datetime.datetime.now()) + ' normalize done')
    return img


def segment_stack(img):
    # Simple thresholding on erosion of image (could go earlier)
    # thresholded = threshold_li(erosion(img))
    thresholded = filters.threshold_li(img)
    print(str(datetime.datetime.now()) + ' threshold done')
    segmented = img > thresholded
    del thresholded
    # Final opening of semented image
    segmented = binary_opening(segmented)
    print(str(datetime.datetime.now()) + ' opening done')
    return segmented

def save_slices(img, names):
    for slice in range(img.shape[0]):
        label = names + str(stack_ids[slice]).zfill(2)
        # label = names + str(stack_ids[slice])
        # imsave('/projects/cami/tps/NASA_sp_lowres/py_stack/%s.tif' % label, img[slice, ...])
        imageio.imwrite('/home/collinf3/Documents/sandia_temp/datasamples/pystack/%s.tif' % label, img_as_uint(img[slice, ...]))


def compare_seg(source, mask):
    segment_viz = img_as_float(gray2rgb(source.copy()))
    segment_viz[mask, :] *= [1, 0, 0]
    plt.imshow(segment_viz)

# %%

#Use the pre-trimmed images
# orig_images = glob('/projects/cami/tps/NASA_sp_lowres/py_stack/orig_14*.tif')
orig_images = sorted(glob('/home/collinf3/Documents/sandia_temp/datasamples/pystack/oxi_gray8bit00*.tif'))

# %%
def img_id(x):
    # return int(x.split('_')[-1].strip('.tif').strip(''))
    return int(x.split('bit')[-1].strip('.tif').strip(''))

# %%
# stack = img_as_float(np.array([np.array(Image.open(x)) for x in orig_images])) #converts entire stack to float
stack = img_as_float(np.array([np.array(Image.open(x)) for x in orig_images])) #converts entire stack to float
# stack_ids = [img_id(x) for x in orig_images] #produces an array of labels
stack_ids = [img_id(x) for x in orig_images] #produces an array of labels



# %%
print(str(datetime.datetime.now()) + ' INITIATE')
filtered = filter_stack(stack)
print(str(datetime.datetime.now()) + ' filtered done')
# del stack
segments = segment_stack(filtered)
print(str(datetime.datetime.now()) + ' segmentation done')
# %%
save_slices(filtered, 'filtr_alt_')
save_slices(segments, 'thrsld_alt_')
save_slices(stack, 'orig_')
print(str(datetime.datetime.now()) + ' TERMINATE')

# %%

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_comparison(id):
    # orig = gray2rgb(img_as_float(np.array(Image.open('orig_%d.tif' % id)))).copy()
    # orig = img_as_float(np.array(Image.open('orig_%d.tif' % id))).copy()
    # im_2d = img_as_float(np.array(Image.open('thrsld_%d.tif' % id)))
    # im_3d = img_as_float(np.array(Image.open('thrsld_alt_0%d.tif' % id)))
    orig = img_as_float(np.array(Image.open('/home/collinf3/Documents/sandia_temp/datasamples/pystack/orig_%d.tif' % id))).copy()
    im_2d = img_as_float(np.array(Image.open('/home/collinf3/Documents/sandia_temp/datasamples/pystack/filtr_alt_%d.tif' % id)))
    im_3d = img_as_float(np.array(Image.open('/home/collinf3/Documents/sandia_temp/datasamples/pystack/thrsld_alt_0%d.tif' % id)))
    red_mult = [1, 0, 0]
    ax, grid = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True)
    over_1 = orig.copy()
    over_1[np.array(im_2d) > 0, :] *= red_mult
    grid[0].imshow(over_1)
    grid[0].set_title('2d Segmentation')
    over_2 = orig.copy()
    over_2[np.array(im_3d) > 0, :] *= red_mult
    grid[1].imshow(over_2)
    grid[1].set_title('3d Segmentation')
    im = grid[2].imshow(im_3d - im_2d, cmap='magma')
    grid[2].set_title('Diff')
    divider = make_axes_locatable(grid[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

# %%
# plot_comparison(2)
id = 2
orig = img_as_float(np.array(Image.open('/home/collinf3/Documents/sandia_temp/datasamples/pystack/orig_%d.tif' % id))).copy()
im_2d = img_as_float(np.array(Image.open('/home/collinf3/Documents/sandia_temp/datasamples/pystack/filtr_alt_%d.tif' % id)))
im_3d = img_as_float(np.array(Image.open('/home/collinf3/Documents/sandia_temp/datasamples/pystack/thrsld_alt_0%d.tif' % id)))
# red_mult = [1, 0, 0] #for 3D matrix
red_mult = [1, 0]
ax, grid = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True)
over_1 = orig.copy()
# over_1[np.array(im_2d) > 0, :] *= red_mult #performing for a 3D matrix
# over_1[np.array(im_2d) > 0] *= red_mult


grid[0].imshow(over_1)
grid[0].set_title('2d Segmentation')
over_2 = orig.copy()
# over_2[np.array(im_3d) > 0, :] *= red_mult
grid[1].imshow(over_2)
grid[1].set_title('3d Segmentation')
im = grid[2].imshow(im_3d - im_2d, cmap='magma')
grid[2].set_title('Diff')
divider = make_axes_locatable(grid[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

# %%
def compare_seg(source, mask):
    segment_viz = img_as_float(gray2rgb(source.copy()))
    segment_viz[mask, :] *= [1, 0, 0]
    plt.imshow(segment_viz)


# /home/collinf3/Documents/sandia_temp/datasamples/pystack/

# %%
# check out anisotropic diffusion filter for denoising
# see how to keep things on github online 