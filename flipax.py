#%%
globals().clear()
#%%
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_uint


# %% import and flip
image_whole = io.imread('/home/collinf3/Documents/sandia_temp/datasamples/sandia_data/CP004a_75_to_1005CP004a_75_to_1005-scale0-5_croppedborders_rotated_chunk.tif')  # full data set
image = np.swapaxes(image_whole,0,1)

# %% check results
plt.figure()
f, ax = plt.subplots(1,2);
ax[0].imshow(image_whole[0], cmap=plt.cm.gray)
ax[0].set_title('old')
ax[1].imshow(image[0], cmap=plt.cm.gray)
ax[1].set_title('new')
plt.show()

# %% export
for i in range(image[0,0].size):
    newimage = image[:, :, i]
    imageio.imwrite('/home/collinf3/Documents/sandia_temp/datasamples/sandia_data/CPhV_axflip/CPhV_axflip%d.tif' %i, img_as_uint(newimage))
