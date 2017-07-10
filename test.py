import numpy as np
from skimage.transform import resize
from maps.helpers.tiffseriesimport import importtiff
from scipy import signal
from pywt import dwt2

imgs = []
for i in range(3, 30):
    imgs.append(importtiff('D:\Scripts\MaPS\Data sets\Stat_DWT', i))

ca, _ = dwt2(imgs[0], 'db1', mode='sym')

da = np.zeros((ca.shape[0], ca.shape[1], 15))

db = np.zeros((ca.shape[0], ca.shape[1], 30))

for i, img in enumerate(imgs[:15]):
    da[:, :, i], _ = dwt2(img, 'db1', mode='sym')
    db[:, :, i], _ = dwt2(img, 'db1', mode='sym')

for i, img in enumerate(imgs[15:]):
    db[:, :, i + 15], _ = dwt2(img, 'db1', mode='sym')

da_resized = resize(da, (da.shape[0], da.shape[1], da.shape[2] * 4), preserve_range=True)

db_resized = resize(db, (db.shape[0], db.shape[1], db.shape[2] * 4), preserve_range=True)

print da.shape
print db.shape
print da_resized.shape
print db_resized.shape

diff = db_resized[:,:,:60] - da_resized

for i in range(diff.shape[2]):
    print i
    print np.where(diff[:,:,i]!=0)

import matplotlib.pyplot as plt

plt.imshow(diff[:,:,58])
plt.show()
