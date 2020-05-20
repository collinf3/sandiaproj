#%%
globals().clear()
#%%
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
from skimage import color #already appears above
from skimage import exposure #already appears above
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

# Generate a level set about zero of two identical ellipsoids in 3D
ellip_base = ellipsoid(10, 10, 10, levelset=True)
ellip_double = np.concatenate((ellip_base[:1, ...],
                               ellip_base[2:, ...]), axis=0)

# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = measure.marching_cubes(ellip_double, 0)

# type(ellip_double)
# print(ellip_double)

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

ax.set_xlabel("x-axis: a = 6 per ellipsoid")
ax.set_ylabel("y-axis: b = 10")
ax.set_zlabel("z-axis: c = 16")

ax.set_xlim(0, 20)  # a = 6 (times two for 2nd ellipsoid)
ax.set_ylim(0, 20)  # b = 10
ax.set_zlim(0, 20)  # c = 16

plt.tight_layout()
plt.show()

#%%
## visualize unmeshed ellipsoid

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');
plt.show()



#%%

fig = plt.figure() #figsize=(10, 10)
ax = fig.add_subplot(111, projection='3d')
x, y, z = np.indices(ellip_double.shape)  # set domain based on tif stack dimensions
# zzz=np.ones(z[0,:,:].shape)
zzz=ellip_double[0,:,:]
xxx=ellip_double[:,0,:]
yyy=ellip_double[:,:,0]
ax.plot_surface(xxx, yyy, zzz)
plt.show()

# # your real data here - some 3d boolean array
# x, y, z = np.indices(ellip_double.shape)  # set domain based on tif stack dimensions
# # voxels = np.invert(li_stack)
# voxels = ellip_double
#
# ax.voxels(voxels)
# # ax.voxels(voxels,edgecolor='k')
# plt.show()

# %%
## new code #todo automate small stack pulling
image_whole = io.imread('/home/collinf3/Documents/sandia_temp/datasamples/oxi_gray8bit.tif')  # full data set
# image = image_whole #entire stack
sf=4 #size factor of center chunk to pull
image = image_whole[0:10, 0:43, 0:199]  # [z,y,x] small section
# image = image_whole[int(len(image_whole)/sf):int(len(image_whole)/sf)*2, int(len(image_whole[0])/sf):int(len(image_whole[0])/sf)*2, int(len(image_whole[0,0])/sf):int(len(image_whole[0,0])/sf)*2]  # [z,y,x] small section

# %% #todo implement automatic filtering scheme

####################################################

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

# %% #todo plot all thresholding methods and user decide which is best

## show single segmented slice
orig1 = image[int(len(image)/2)]  # pull center slice
thresholded_slice = filters.threshold_li(orig1) #select thresholding technique
orig1_li = binary_opening(orig1 > thresholded_slice)

filtered=filter_stack(orig1)
thresholded_slice = filters.threshold_li(filtered) #select thresholding technique
filtered_li = binary_opening(filtered > thresholded_slice)


plt.figure()
f, ax = plt.subplots(4, 1);
ax[0].imshow(color.gray2rgb(orig1));  # filtered image
ax[0].set_title('original image')
ax[1].imshow(orig1_li, cmap=plt.cm.gray);
ax[1].set_title('non-filtered segmentation')
ax[2].imshow(filtered, cmap=plt.cm.gray);
ax[2].set_title('filtered')
ax[3].imshow(filtered_li, cmap=plt.cm.gray);
ax[3].set_title('filtered segmented')
plt.show()


# %%

## show single segmented slice OLD
single1 = image[int(len(image)/2)]  # pull center slice
# single1 = image[8]  # pull specific slice
thresholded_slice = filters.threshold_li(single1)
segmented_bool_slice = single1 > thresholded_slice
single1_li_slice = binary_opening(segmented_bool_slice)

plt.figure()
f, ax = plt.subplots(3, 1);
ax[0].imshow(color.gray2rgb(single1));  # filtered image
ax[0].set_title('filtered image')
ax[1].imshow(single1_li_slice, cmap=plt.cm.gray);
ax[1].set_title('Li segmentation')
plt.show()

# , aspect='auto'
# %%
## segment stack
# stack1 = image  # pull filtered image stack
# thresholded_stack = filters.threshold_li(stack1)
# segmented_bool_stack = stack1 > thresholded_stack
# li_stack = binary_opening(segmented_bool_stack)


filtered_stack=filter_stack(image)
thresholded_slice = filters.threshold_li(filtered) #select thresholding technique
filtered_li_stack = binary_opening(filtered > thresholded_slice)

# plt.figure()
# f, ax = plt.subplots(2,1, figsize=(6, 12));
# ax[0].imshow(color.gray2rgb(single1),aspect='auto'); #filtered image
# ax[0].set_title('filtered image')
# ax[1].imshow(single1_li,cmap=plt.cm.gray,aspect='auto');
# ax[1].set_title('Li segmentation')

# %%
## visualize segmented stack
fig = plt.figure(figsize=(10, 10)) #figsize=(10, 10)
ax = fig.add_subplot(111, projection='3d')
# your real data here - some 3d boolean array
x, y, z = np.indices(image.shape)  # set domain based on tif stack dimensions
# voxels = np.invert(li_stack)
voxels = filtered_li_stack

ax.voxels(voxels)
# ax.voxels(voxels,edgecolor='k')
plt.show()

# %%
## visualize UN-segmented stack
fig = plt.figure() #figsize=(10, 10)
ax = fig.add_subplot(111, projection='3d')
# your real data here - some 3d boolean array
x, y, z = np.indices(image.shape)  # set domain based on tif stack dimensions
voxels = np.invert(image)
# voxels = image

# ax.voxels(voxels)
ax.voxels(voxels,edgecolor='k')
plt.show()

# %%
# todo get multiotsu to seperate without losing fidelity

#import slice
meteor1 = io.imread('/home/collinf3/Documents/sandia_temp/datasamples/tamdakht_800C_8bit_brut_slice1.tif')
tv1=denoise_tv_chambolle(meteor1, weight=0.5, eps=0.001)
thresholds_tv1 = filters.threshold_multiotsu(tv1, classes=4)
regions_tv1 = np.digitize(tv1, bins=thresholds_tv1)

# %%
#import chunk
meteor_chunk = io.imread('/home/collinf3/Documents/sandia_temp/datasamples/tamdakht_800C_8bit_brut_chunk1.tif')
tv_stack=denoise_tv_chambolle(meteor_chunk, weight=0.5, eps=0.001)
thresholds_tv_stack = filters.threshold_multiotsu(tv_stack, classes=4)
regions_tv_stack = np.digitize(tv_stack, bins=thresholds_tv_stack)

b=regions_tv_stack==3 #boolean of specified region
regions_tv_stack_segmented=tv_stack*b

# %% code code code
#save filtered chunk
for i in range(meteor_chunk[0,0].size):
    newimage = tv_stack[:, :, i]
    imageio.imwrite('/home/collinf3/Documents/sandia_temp/datasamples/filter_chunk/filtered_chunk%d.tif' %i, img_as_uint(newimage))

# %%
# Use marching cubes to obtain the surface mesh regions_tv_stack==3
verts, faces, normals, values = measure.marching_cubes(tv_stack, spacing=[1,1,1],
                                                               gradient_direction='descent', step_size=1,
                                                               allow_degenerate=True, method='lewiner')
mesh=verts[faces]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mesh_vert = Poly3DCollection(mesh)
# mesh_vert.set_edgecolor('k')
ax.add_collection3d(mesh_vert)
# mesh.set_edgecolor('k')
# ax.add_collection3d(mesh)
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
axsize = [len(meteor_chunk),len(meteor_chunk[0]),len(meteor_chunk[0,0])]
ax.set_xlim(0, axsize[0])  # a = 6 (times two for 2nd ellipsoid)
ax.set_ylim(0, axsize[1])  # b = 10
ax.set_zlim(0, axsize[2])  # c = 16
plt.tight_layout()
plt.show()

# %%
from stl import mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = verts[f[j],:]

cube.save('/home/collinf3/Documents/sandia_temp/datasamples/mesh2.stl')

# %%
# todo measure surface area using skimage.measure.mesh_surface_area
# todo surface smoothing ? dont really think its necessary as fidelity loss would occur

# %%
# todo learn how to pull DFLY code directly into python, pycharm interfacing with Dfly?

# %% OLD CODE

# Use marching cubes to obtain the surface mesh
verts, faces, normals, values = measure.marching_cubes(image, spacing=[1,1,1],
                                                               gradient_direction='descent', step_size=2,
                                                               allow_degenerate=True, method='lewiner')


# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh_vert = Poly3DCollection(verts[faces])
mesh_vert.set_edgecolor('k')
ax.add_collection3d(mesh_vert)

# mesh = Poly3DCollection(normals[verts])
# mesh.set_edgecolor('k')
# ax.add_collection3d(mesh)

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

axsize = [len(image),len(image[0]),len(image[0,0])]
ax.set_xlim(0, axsize[0])  # a = 6 (times two for 2nd ellipsoid)
ax.set_ylim(0, axsize[1])  # b = 10
ax.set_zlim(0, axsize[2])  # c = 16

plt.tight_layout()
plt.show()

# %%
# # Use marching cubes to obtain the surface mesh of these ellipsoids
# verts, faces, normals, values = measure.marching_cubes_lewiner(ellip_double, 0)

# type(verts)
# print(verts)

# # Display resulting triangular mesh using Matplotlib. This can also be done
# # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Fancy indexing: `verts[faces]` to generate a collection of triangles
# mesh = Poly3DCollection(verts[faces])
# mesh.set_edgecolor('k')
# ax.add_collection3d(mesh)

# ax.set_xlabel("x-axis: a = 6 per ellipsoid")
# ax.set_ylabel("y-axis: b = 10")
# ax.set_zlabel("z-axis: c = 16")

# ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
# ax.set_ylim(0, 20)  # b = 10
# ax.set_zlim(0, 32)  # c = 16

# plt.tight_layout()
# plt.show()

# %%
# fileno error
# faulthandler.enable()
# faulthandler.dump_traceback()
def img_id(x):
    return int(x.split('_')[-1].strip('.tif').strip(''))


# %%
# Sequence of filters
def filter_stack(img):
    # Adjust exposure
    img = adjust_log(img, 1)
    faulthandler.dump_traceback()
    print(str(datetime.datetime.now()) + ' adjust log done')
    # Denoise
    img = denoise_tv_chambolle(img, weight=0.2, eps=0.001)
    faulthandler.dump_traceback()
    print(str(datetime.datetime.now()) + ' denoise done')
    # Erode image to reduce roughness/small spots
    img = erosion(img)
    # Reconstruction using h-level (increases phase contrast, minimizes contrast in parent phase, cuts down beam hardening)
    # Adaptive level choice (increases per-slice variability)
    h = 0.9 * img.min()
    # h = 0.3
    seed = img - h
    dilated = reconstruction(seed, img, method='dilation')
    faulthandler.dump_traceback()
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


def save_slices(img, names):
    for slice in range(img.shape[0]):
        label = names + str(stack_ids[slice]).zfill(5)
        imsave('/projects/cami/tps/NASA_sp_lowres/py_stack/%s.tif' % label, img[slice, ...])


def compare_seg(source, mask):
    segment_viz = img_as_float(gray2rgb(source.copy()))
    segment_viz[mask, :] *= [1, 0, 0]
    plt.imshow(segment_viz)


# %%
# Use the pre-trimmed images
##orig_images = glob('/projects/cami/tps/NASA_sp_lowres/py_stack/orig_14*.tif')
orig_images = glob('/ascldap/users/colfost/Documents/colfost/NASA_sp_lowres/py_stack/orig_14*.tif')

# plt.imshow(color.gray2rgb(orig_images))

# %%
## my code
# image = io.imread_collection('/ascldap/users/colfost/Documents/colfost/NASA_sp_lowres/py_stack/orig_14*.tif') #100 tifs
image = io.imread('/ascldap/users/colfost/Documents/colfost/NASA_sp_lowres/py_stack/orig_1400.tif')  # single slice
plt.imshow(color.gray2rgb(image))  # plot single slice

# %%
## my code
Image.open(orig_images[1]);
# [np.array(Image.open(x)) for x in orig_images]
# np.array([np.array(Image.open(x)) for x in orig_images])

# %%
stack = img_as_float(np.array([np.array(Image.open(x)) for x in orig_images]))  # converts entire stack to float
# stack = np.array([np.array(Image.open(x)) for x in orig_images])
stack_ids = [img_id(x) for x in orig_images]  # produces an array of labels

## my code
# plt.imshow(color.gray2rgb(stack))

# %%
## my code
stack1 = stack
# plt.imshow(color.gray2rgb(stack1[1]));
# print(type(stack1),stack1.size)
stack2 = adjust_log(stack1[0], 1)  # performs logarithmic correction on the input image

plt.figure()
# subplot(r,c) provide the no. of rows and columns
f, ax = plt.subplots(2, 1, figsize=(7, 12))
ax[0].imshow(color.gray2rgb(stack1[0]))
ax[1].imshow(color.gray2rgb(stack2))
print(stack1[0].shape)
print(stack2.shape)

# plt.imshow(color.gray2rgb(stack1[1]));
# plt.imshow(color.gray2rgb(stack1[2]))

# %%
## my code
stack1 = adjust_log(stack[0], 1)
stack2 = denoise_tv_chambolle(stack[0], weight=0.2, eps=0.001)

plt.figure()
f, ax = plt.subplots(3, 1, figsize=(7, 12))
ax[0].imshow(color.gray2rgb(stack[0]), aspect='auto')  # original
ax[1].imshow(color.gray2rgb(stack1), aspect='auto')
ax[2].imshow(color.gray2rgb(stack2), aspect='auto')

# %%
# Erode image to reduce roughness/small spots

## my code
stack1 = denoise_tv_chambolle(adjust_log(stack[0], 1), weight=0.2, eps=0.001)  # apply upstream functions
stack2 = erosion(stack1)

plt.figure()
f, ax = plt.subplots(2, 1, figsize=(7, 12))
ax[0].imshow(color.gray2rgb(stack[0]), aspect='auto')
ax[1].imshow(color.gray2rgb(stack2), aspect='auto')

# %%
# Reconstruction using h-level (increases phase contrast, minimizes contrast in parent phase, cuts down beam hardening)
# Adaptive level choice (increases per-slice variability)

## my code

stack1 = erosion(denoise_tv_chambolle(adjust_log(stack[0], 1), weight=0.2, eps=0.001))  # apply upstream functions
h = 0.9 * stack1.min()  # takes lowest gray scale
seed = stack1 - h
dilated = reconstruction(seed, stack1, method='dilation')
stack2 -= dilated

plt.figure()
f, ax = plt.subplots(2, 1, figsize=(7, 12))
ax[0].imshow(color.gray2rgb(stack[0]), aspect='auto')
ax[1].imshow(color.gray2rgb(stack2), aspect='auto')

# %%
# Renormalize array
stack3 = stack2
stack3 *= 1. / stack2.max()  # equivalent to stack3 = stack3 * 1./stack2.max()

plt.figure()
f, ax = plt.subplots(2, 1, figsize=(7, 12))
ax[0].imshow(color.gray2rgb(stack2), aspect='auto')
ax[1].imshow(color.gray2rgb(stack3), aspect='auto')

# %%

## my code
stack1 = stack3  # pull filtered image
# thresholded1 = np.array[filters.threshold_isodata(stack1,401),filters.threshold_isodata(stack1,401)] #thresholded
thresholded = filters.threshold_li(stack1)
thresholded1 = filters.threshold_yen(stack1)
thresholded2 = filters.threshold_otsu(stack1)
segmented_bool = stack1 > thresholded
segmented_bool1 = stack1 > thresholded1
segmented_bool2 = stack1 > thresholded2
stack2_li = binary_opening(segmented_bool)
stack2_yen = binary_opening(segmented_bool1)
stack2_otsu = binary_opening(segmented_bool2)

plt.figure()
f, ax = plt.subplots(2, 2, figsize=(12, 12));
ax[0, 0].imshow(color.gray2rgb(stack3), aspect='auto');  # filtered image
ax[0, 0].set_title('filtered image')
ax[1, 0].imshow(stack2_li, cmap=plt.cm.gray, aspect='auto');
ax[1, 0].set_title('Li segmentation')
ax[0, 1].imshow(stack2_yen, cmap=plt.cm.gray, aspect='auto');
ax[0, 1].set_title('Yen segementation')
ax[1, 1].imshow(stack2_otsu, cmap=plt.cm.gray, aspect='auto');
ax[1, 1].set_title('Otsu segementation')


# %%

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

    # %%
    ## my code
    # def save_slices(img,names):
    #     for slice in range(img.shape[0]):
    label = str(stack_ids[0]).zfill(5)
    imsave('/projects/cami/tps/NASA_sp_lowres/py_stack/%s.tif' % label, stack[0])


# %%
def save_slices(img, names):
    for slice in range(img.shape[0]):  # defining that variable slice run through 0 to 100
        label = names + str(stack_ids[slice]).zfill(5)  # zfill turns 100 -> 00100
        imsave('/projects/cami/tps/NASA_sp_lowres/py_stack/%s.tif' % label, img[slice, ...])


# %%


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
print(str(datetime.datetime.now()) + ' INITIATE')
filtered = filter_stack(stack)
print(str(datetime.datetime.now()) + ' filtered done')
del stack
segments = segment_stack(filtered)
print(str(datetime.datetime.now()) + ' segmentation done')
save_slices(filtered, 'filtr_alt_')
save_slices(segments, 'thrsld_alt_')
print(str(datetime.datetime.now()) + ' TERMINATE')

from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_comparison(id):
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


