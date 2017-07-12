import numpy as np
from skimage.transform import resize
from maps.helpers.tiffseriesimport import importtiff, writetiff
from scipy import signal
from pywt import dwt2

from maps.settings import setting, reload_current_settings

reload_current_settings()


def chunked_dwt_resize(dwt_array, resize_factor, chunk_size=100):
    max_chunk_size = 100

    chunk_size = min(chunk_size, max_chunk_size)
    overlap = resize_factor - 1

    hght = dwt_array.shape[0]
    wdth = dwt_array.shape[1]
    dpth = dwt_array.shape[2]

    n = int(np.ceil((float(dpth)) / (chunk_size)))

    resized_dwt = np.zeros((hght, wdth, dpth * resize_factor))

    tempres = resize(
        dwt_array[:, :, 0: chunk_size + overlap],
        (hght, wdth, (chunk_size + overlap) * resize_factor),
        preserve_range=True
    )
    resized_dwt[:, :, : chunk_size * resize_factor] = tempres[:, :, :-(overlap * resize_factor)]
    del tempres

    for i in range(1, n - 1):
        start_frame = i * chunk_size - overlap
        end_frame = (i + 1) * chunk_size + overlap

        print start_frame, '->', end_frame
        print i * chunk_size * resize_factor, '->', (i + 1) * chunk_size * resize_factor

        tempres = resize(
            dwt_array[:, :, start_frame:end_frame],
            (hght, wdth, (chunk_size + 2 * overlap) * resize_factor),
            preserve_range=True
        )

        resized_dwt[:, :, i * chunk_size * resize_factor: (i + 1) * chunk_size * resize_factor] = tempres[:, :, overlap * resize_factor:-overlap * resize_factor]
        del tempres

    tempres = resize(
        dwt_array[:, :, -(chunk_size + overlap):],
        (hght, wdth, (chunk_size + overlap) * resize_factor),
        preserve_range=True
    )
    resized_dwt[:, :, -chunk_size*resize_factor:] = tempres[:, :, overlap * resize_factor:]
    del tempres

    return resized_dwt


N = 100
imgs = []
for i in range(3, N * 2):
    imgs.append(importtiff('D:\Scripts\MaPS\Data sets\Stat_DWT', i))

ca, _ = dwt2(imgs[0], 'db1', mode='sym')

da = np.zeros((ca.shape[0], ca.shape[1], N * 2))

db = np.zeros((ca.shape[0], ca.shape[1], N * 2))

for i, img in enumerate(imgs[:2*N]):
    da[:, :, i], _ = dwt2(img, 'db1', mode='sym')
    db[:, :, i], _ = dwt2(img, 'db1', mode='sym')

# for i, img in enumerate(imgs[N:]):
#     db[:, :, i + N], _ = dwt2(img, 'db1', mode='sym')

# da_resized = resize(da, (da.shape[0], da.shape[1], da.shape[2] * 4), preserve_range=True)
da_resized = chunked_dwt_resize(da, resize_factor=4, chunk_size=10)

db_resized = resize(db, (db.shape[0], db.shape[1], db.shape[2] * 4), preserve_range=True)

print da.shape
print db.shape
print da_resized.shape
print db_resized.shape

diff = db_resized - da_resized

for i in range(diff.shape[2]):
    diffloc = np.where(diff[:, :, i] != 0)
    if diffloc[0].any() or diffloc[1].any():
        print i,
        # print diffloc

for i in range(diff.shape[2]):
    writetiff(diff[:, :, i].astype('int16'), 'D:\\Scripts\\MaPS\\Data sets\\DWT_CHUNK_DIFF', i)

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(da_resized[:, :, 119])
# plt.colorbar()
# # plt.figure()
# # plt.imshow(db_resized[:, :, 119])
# # plt.imshow(diff[:, :, -2])
# plt.show()
