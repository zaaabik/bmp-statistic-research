import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np

import bmp


def decimationYCbCr(rgb, n):
    print("decimation ")
    y = rgb.shape[0]
    x = rgb.shape[1]
    y_cb_cr = bmp.convertYCbCr(rgb)
    resG = bmp.decimationByDeletingEven(y_cb_cr[..., 1], n)
    resR = bmp.decimationByDeletingEven(y_cb_cr[..., 2], n)
    resG2 = bmp.recoverDecimationByDeletingEven(resG, n)
    resR2 = bmp.recoverDecimationByDeletingEven(resR, n)
    rgb_recovered = np.zeros((y, x, 3), 'uint8')
    rgb_recovered[..., 0] = y_cb_cr[..., 0]
    rgb_recovered[..., 1] = resG2
    rgb_recovered[..., 2] = resR2
    print("psnr cb", bmp.psnr(y_cb_cr[..., 1], resG2))
    print("psnr cr", bmp.psnr(y_cb_cr[..., 0], resR2))
    rgb_recovered = bmp.inverseConvertYCbCr(rgb_recovered)
    start = datetime.datetime.now()
    print("psnr blue", bmp.psnr(rgb[..., 0], rgb_recovered[..., 0]))
    print("TIME TIME TIME ", datetime.datetime.now() - start)
    print("psnr green", bmp.psnr(rgb[..., 1], rgb_recovered[..., 1]))
    print("psnr red", bmp.psnr(rgb[..., 2], rgb_recovered[..., 2]))
    return rgb_recovered


def avgDecimationYCbCr(rgb, n):
    print("avg decimation ")
    y = rgb.shape[0]
    x = rgb.shape[1]
    y_cb_cr = bmp.convertYCbCr(rgb)
    resG = bmp.decimationByAverageValue(y_cb_cr[..., 1], 2)
    resR = bmp.decimationByAverageValue(y_cb_cr[..., 2], 2)
    resG2 = bmp.recoverDecimationByDeletingEven(resG, 2)
    resR2 = bmp.recoverDecimationByDeletingEven(resR, 2)
    rgb_recovered = np.zeros((y, x, 3), 'uint8')
    rgb_recovered[..., 0] = y_cb_cr[..., 0]
    rgb_recovered[..., 1] = resG2
    rgb_recovered[..., 2] = resR2
    print("psnr cb", bmp.psnr(y_cb_cr[..., 1], resG2))
    print("psnr cr", bmp.psnr(y_cb_cr[..., 0], resR2))
    rgb_recovered = bmp.inverseConvertYCbCr(rgb_recovered)
    print("psnr blue", bmp.psnr(rgb[..., 0], rgb_recovered[..., 0]))
    print("psnr green", bmp.psnr(rgb[..., 1], rgb_recovered[..., 1]))
    print("psnr red", bmp.psnr(rgb[..., 2], rgb_recovered[..., 2]))
    return rgb_recovered


def getAutoCor(array, title):
    t = range(10, array.shape[1] // 4, 5)
    cor = bmp.auto_correlation(array, 5, t)
    cor1 = bmp.auto_correlation(array, -5, t)
    cor2 = bmp.auto_correlation(array, 0, t)
    cor3 = bmp.auto_correlation(array, 10, t)
    cor4 = bmp.auto_correlation(array, -10, t)
    plt.title(title)
    plt.plot(t, cor, color='g')
    plt.plot(t, cor1, color='b')
    plt.plot(t, cor2, color='r')
    plt.plot(t, cor3, color='y')
    plt.plot(t, cor4, color='black')
    plt.show()


in_file = "C:\\Users\\HawkA\\OneDrive\\Pictures\\pyTest\\lena.bmp"
res_file = "C:\\Users\\HawkA\\OneDrive\\Pictures\\pyTest\\lastRes.bmp"
res_file_path = "C:\\Users\\HawkA\\OneDrive\\Pictures\\pyTest\\"

# 1.2

header, rgb = bmp.readBmp(in_file)

r1 = avgDecimationYCbCr(rgb, 4)
r2 = decimationYCbCr(rgb, 4)
bmp.writeBmp(res_file_path + "recoveredAvg4.bmp", header, r1)
bmp.writeBmp(res_file_path + "recovered4.bmp", header, r2)

# 1.3 r g b separately
r = np.copy(rgb)
r[..., [0, 1]] = 0
g = np.copy(rgb)
g[..., (0, 2)] = 0
b = np.copy(rgb)
b[..., (1, 2)] = 0
bmp.writeBmp(res_file_path + "r.bmp", header, r)
bmp.writeBmp(res_file_path + "g.bmp", header, g)
bmp.writeBmp(res_file_path + "b.bmp", header, b)
# 1.4 a
r = r[..., 2]
g = g[..., 1]
b = b[..., 0]

cor_r_g = bmp.correlation(r, g)
cor_g_b = bmp.correlation(g, b)
cor_r_b = bmp.correlation(r, b)

print("correlation")
print("r b", cor_r_b)
print("g b", cor_g_b)
print("r h", cor_r_g)
# 1.4 b
getAutoCor(r, "red")
getAutoCor(g, "green")
getAutoCor(b, "blue")
# 1.5
a = datetime.datetime.now()
y_cb_cr = bmp.convertYCbCr(rgb)
b = datetime.datetime.now()
print("time", b - a)
# 1.6
bmp.writeBmp(res_file_path + "y.bmp", header, y_cb_cr[..., (0, 0, 0)])
bmp.writeBmp(res_file_path + "cb.bmp", header, y_cb_cr[..., (1, 1, 1)])
bmp.writeBmp(res_file_path + "cr.bmp", header, y_cb_cr[..., (2, 2, 2)])

y = y_cb_cr[..., 0]
cb = y_cb_cr[..., 1]
cr = y_cb_cr[..., 2]

cor_y_cb = bmp.correlation(y, cb)
cor_cb_cr = bmp.correlation(cb, cr)
cor_y_cr = bmp.correlation(y, cr)

print("correlation")
print("y cb ", cor_y_cb)
print("y cr ", cor_y_cr)
print("cb cr ", cor_cb_cr)

# 1.7

rgb_recovered = bmp.inverseConvertYCbCr(y_cb_cr)
psnr_red = bmp.psnr(rgb[..., 2], rgb_recovered[..., 2])
psnr_green = bmp.psnr(rgb[..., 1], rgb_recovered[..., 1])
psnr_blue = bmp.psnr(rgb[..., 0], rgb_recovered[..., 0])
bmp.writeBmp(res_file_path + "recovered.bmp", header, rgb_recovered)
print("psnr red ", psnr_red)
print("psnr green", psnr_green)
print("psnr blue", psnr_blue)

# 1.8
