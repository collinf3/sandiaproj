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
from skimage import img_as_uint

# %%
## new code #todo automate small stack pulling
image_whole = io.imread('/home/collinf3/Documents/sandia_temp/datasamples/oxi_gray8bit.tif')  # full data set
# image = image_whole #entire stack
sf=4 #size factor of center chunk to pull
image = image_whole[0:10, 0:43, 0:199]  # [z,y,x] small section
# image = image_whole[int(len(image_whole)/sf):int(len(image_whole)/sf)*2, int(len(image_whole[0])/sf):int(len(image_whole[0])/sf)*2, int(len(image_whole[0,0])/sf):int(len(image_whole[0,0])/sf)*2]  # [z,y,x] small section

# %%
####################################################

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
    # h = 0.9 * img.min()
    h = 0.3 * img.min()
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
    thresholded = threshold_li(img)
    print(str(datetime.datetime.now()) + ' threshold done')
    segmented = img > thresholded
    del thresholded
    # Final opening of semented image
    segmented = binary_opening(segmented)
    print(str(datetime.datetime.now()) + ' opening done')
    return segmented

def compare_seg(source, mask):
    segment_viz = img_as_float(gray2rgb(source.copy()))
    segment_viz[mask, :] *= [1, 0, 0]
    plt.imshow(segment_viz)

# %%

# todo select segmentation method that optimizes choice, check out "overlay" method in plots from original script

## show single segmented slice
orig1 = image[int(len(image)/2)]  # pull center slice
thresholded_slice = filters.threshold_li(orig1) #select thresholding technique
orig1_li = binary_opening(orig1 > thresholded_slice)

filtered=filter_stack(orig1)

filtered_li = binary_opening(filtered > filters.threshold_li(filtered))
filtered_yen = binary_opening(filtered > filters.threshold_yen(filtered))
filtered_otsu = binary_opening(filtered > filters.threshold_otsu(filtered))
filtered_mean = binary_opening(filtered > filters.threshold_mean(filtered))
filtered_isodata = binary_opening(filtered > filters.threshold_isodata(filtered))
filtered_local = binary_opening(filtered > filters.threshold_local(filtered,3))
filtered_minimum = binary_opening(filtered > filters.threshold_minimum(filtered))
# filtered_multiotsu = binary_opening(filtered > filters.threshold_multiotsu(filtered))
filtered_niblack = binary_opening(filtered > filters.threshold_niblack(filtered))
filtered_sauvola = binary_opening(filtered > filters.threshold_sauvola(filtered))
filtered_triangle = binary_opening(filtered > filters.threshold_triangle(filtered))

plt.figure()
f, ax = plt.subplots(4,3);
ax[0,0].imshow(filtered, cmap=plt.cm.gray);
ax[0,0].set_title('filtered')
ax[0,1].imshow(filtered_li, cmap=plt.cm.gray);
ax[0,1].set_title('li segmentation')
ax[0,2].imshow(filtered_yen, cmap=plt.cm.gray);
ax[0,2].set_title('yen segmentation')
ax[1,0].imshow(filtered_otsu, cmap=plt.cm.gray);
ax[1,0].set_title('otsu segmentation')
ax[1,1].imshow(filtered_mean, cmap=plt.cm.gray);
ax[1,1].set_title('mean segmentation')
ax[1,2].imshow(filtered_isodata, cmap=plt.cm.gray);
ax[1,2].set_title('isodata segmentation')
ax[2,0].imshow(filtered_local, cmap=plt.cm.gray);
ax[2,0].set_title('local segmentation')
ax[2,1].imshow(filtered_minimum, cmap=plt.cm.gray);
ax[2,1].set_title('minimum segmentation')
ax[2,2].imshow(filtered_niblack, cmap=plt.cm.gray);
ax[2,2].set_title('niblack segmentation')
ax[3,0].imshow(filtered_sauvola, cmap=plt.cm.gray);
ax[3,0].set_title('sauvola segmentation')
ax[3,1].imshow(filtered_triangle, cmap=plt.cm.gray);
ax[3,1].set_title('triangle segmentation')
ax[3,1].spines['bottom'].set_linewidth(1)
ax[3,2].imshow(orig1, cmap=plt.cm.gray);
ax[3,2].set_title('original')
plt.subplots_adjust()
# [axi.set_axis_off() for axi in ax.ravel()]
plt.show()


# %%
# todo multiotsu segmentation

meteor1 = io.imread('/home/collinf3/Documents/sandia_temp/datasamples/tamdakht_800C_8bit_brut_slice1.tif')

def filter_stack(img):
    # Adjust exposure
    img = adjust_log(img, 1)
    # faulthandler.dump_traceback()
    print(str(datetime.datetime.now()) + ' adjust log done')
    # # Denoise
    img = denoise_tv_chambolle(img, weight=0.2, eps=0.001)
    # faulthandler.dump_traceback()
    print(str(datetime.datetime.now()) + ' denoise done')
    # # Erode image to reduce roughness/small spots
    # img = erosion(img)
    # print(str(datetime.datetime.now()) + ' erosion done')
    # # Reconstruction using h-level (increases phase contrast, minimizes contrast in parent phase, cuts down beam hardening)
    # # Adaptive level choice (increases per-slice variability)
    # # h = 0.9 * img.min()
    # h = 0.3 * img.min()
    # # h = 0.3
    # seed = img - h
    # dilated = reconstruction(seed, img, method='dilation')
    # # faulthandler.dump_traceback()
    # print(str(datetime.datetime.now()) + ' reconstruction done')
    # img -= dilated
    # # Renormalize array
    # img *= 1. / img.max()
    # print(str(datetime.datetime.now()) + ' normalize done')
    return img

filtered=filter_stack(meteor1)
orig1=meteor1

plt.figure()
f, ax = plt.subplots(2, 1)
ax[0].imshow(color.gray2rgb(orig1))
ax[0].set_title('original')
ax[1].imshow(filtered, cmap=plt.cm.gray)
ax[1].set_title('filtered')
plt.show()

thresholds = filters.threshold_multiotsu(filtered, classes=4)
# Using the threshold values, we generate the three regions.
regions = np.digitize(filtered, bins=thresholds)
fig, ax = plt.subplots(1,3, figsize=(10, 3.5))
# Plotting the original image.
ax[0].imshow(filtered, cmap=plt.cm.gray)
ax[0].set_title('filtered')
ax[0].axis('off')
ax[0].spines['bottom'].set_linewidth(.5)
# Plotting the histogram and the two thresholds obtained from
# multi-Otsu.
ax[1].hist(filtered.ravel(), bins=255)
ax[1].set_title('Histogram')
for thresh in thresholds:
    ax[1].axvline(thresh, color='r')
# Plotting the Multi Otsu result.
ax[2].imshow(regions, cmap='Accent')
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')
plt.subplots_adjust()
plt.show()

# %%
# todo automate "knob turning" in tv & aniso for best segmentation. \
#  perhaps ideal judged from what yields most dramatic and spaced peaks in tv...
from medpy.filter.smoothing import anisotropic_diffusion

tv=denoise_tv_chambolle(meteor1, weight=0.5, eps=0.001)
aniso=anisotropic_diffusion(meteor1, niter=40, kappa=10, gamma=0.1, voxelspacing=None, option=1)

# multiotsu comparison
thresholds_tv = filters.threshold_multiotsu(tv, classes=4)
regions_tv = np.digitize(tv, bins=thresholds_tv)
thresholds_aniso = filters.threshold_multiotsu(aniso, classes=4)
regions_aniso = np.digitize(aniso, bins=thresholds_aniso)
thresholds_orig = filters.threshold_multiotsu(meteor1, classes=4)
regions_orig = np.digitize(meteor1, bins=thresholds_orig)

plt.figure()
f, ax = plt.subplots(3, 3)
ax[0,0].imshow(color.gray2rgb(meteor1))
ax[0,0].set_title('original')
ax[0,0].axis('off')
ax[0,1].hist(meteor1.ravel(), bins=255)
ax[0,1].set_title('Histogram')
for thresh in thresholds_orig:
    ax[0,1].axvline(thresh, color='r')
ax[0,1].axis('off')
ax[0,2].imshow(regions_orig, cmap='Accent')
ax[0,2].set_title('Multi-Otsu result')
ax[0,2].axis('off')

ax[1,0].imshow(color.gray2rgb(tv))
ax[1,0].set_title('tv filtered')
ax[1,0].axis('off')
ax[1,1].hist(tv.ravel(), bins=255)
for thresh in thresholds_tv:
    ax[1,1].axvline(thresh, color='r')
ax[1,1].axis('off')
ax[1,2].imshow(regions_tv, cmap='Accent')
ax[1,2].axis('off')

ax[2,0].imshow(aniso, cmap=plt.cm.gray)
ax[2,0].set_title('aniso filtered')
ax[2,0].axis('off')
ax[2,1].hist(aniso.ravel(), bins=255)
for thresh in thresholds_aniso:
    ax[2,1].axvline(thresh, color='r')
ax[2,1].axis('off')
ax[2,2].imshow(regions_tv, cmap='Accent')
ax[2,2].axis('off')

plt.subplots_adjust()
plt.show()

# %%

plt.figure()
f, ax = plt.subplots(2, 3, figsize=(10, 3.5))
ax[0,0].imshow(color.gray2rgb(tv))
ax[0,0].set_title('filtered original')
ax[0,0].axis('off')
ax[0,1].hist(tv.ravel(), bins=255)
ax[0,1].set_title('Histogram')
for thresh in thresholds_tv:
    ax[0,1].axvline(thresh, color='r')
ax[0,1].axis('off')
ax[0,2].imshow(regions_tv, cmap='Accent')
ax[0,2].set_title('Multi-Otsu result')
ax[0,2].axis('off')

ax[1,0].imshow(regions_tv==3, cmap='Set1')
ax[1,0].set_title('seg 1')
ax[1,0].axis('off')

ax[1,1].imshow(regions_tv==2, cmap='Set1')
ax[1,1].set_title('seg 2')
ax[1,1].axis('off')

ax[1,2].imshow(regions_tv==1, cmap='Set1')
ax[1,2].set_title('seg 3')
ax[1,2].axis('off')

plt.subplots_adjust()
plt.show()

# %%
# todo random walker segementation for noisy data scikit image \
#  https://scikit-image.org/docs/0.9.x/auto_examples/plot_random_walker_segmentation.html
# todo check out everything in scikit image filters as well as medpy
# todo implement Dfly ML to python with multiotsu teaching seeds


