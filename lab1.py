import sys
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import bmp

in_file = "C:\\Users\\HawkA\\OneDrive\\Pictures\\pyTest\\lena.bmp"
res_file = "C:\\Users\\HawkA\\OneDrive\\Pictures\\pyTest\\lastRes.bmp"
res_file_path = "C:\\Users\\HawkA\\OneDrive\\Pictures\\pyTest\\"


def decimationYCbCr(rgb, n):
    print(f'\n \n \n decimation {n}')
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
    print("psnr cr", bmp.psnr(y_cb_cr[..., 2], resR2))
    rgb_recovered = bmp.inverseConvertYCbCr(rgb_recovered)

    print("psnr blue", bmp.psnr(rgb[..., 0], rgb_recovered[..., 0]))

    print("psnr green", bmp.psnr(rgb[..., 1], rgb_recovered[..., 1]))
    print("psnr red", bmp.psnr(rgb[..., 2], rgb_recovered[..., 2]))
    return rgb_recovered


def avgDecimationYCbCr(rgb, n):
    print(f'\n \n \n avg decimation {n}')
    y = rgb.shape[0]
    x = rgb.shape[1]
    y_cb_cr = bmp.convertYCbCr(rgb)
    resG = bmp.decimationByAverageValue(y_cb_cr[..., 1], n)
    resR = bmp.decimationByAverageValue(y_cb_cr[..., 2], n)
    resG2 = bmp.recoverDecimationByDeletingEven(resG, n)
    resR2 = bmp.recoverDecimationByDeletingEven(resR, n)
    rgb_recovered = np.zeros((y, x, 3), 'uint8')
    rgb_recovered[..., 0] = y_cb_cr[..., 0]
    rgb_recovered[..., 1] = resG2
    rgb_recovered[..., 2] = resR2
    print("psnr cb", bmp.psnr(y_cb_cr[..., 1], resG2))
    print("psnr cr", bmp.psnr(y_cb_cr[..., 2], resR2))
    rgb_recovered = bmp.inverseConvertYCbCr(rgb_recovered)
    print("psnr blue", bmp.psnr(rgb[..., 0], rgb_recovered[..., 0]))
    print("psnr green", bmp.psnr(rgb[..., 1], rgb_recovered[..., 1]))
    print("psnr red", bmp.psnr(rgb[..., 2], rgb_recovered[..., 2]))
    return rgb_recovered


def getAutoCor(array, title):
    t = (range(0, array.shape[1] // 4, 5))
    corg = bmp.auto_correlation(array, 5, t)
    corb = bmp.auto_correlation(array, -5, t)
    corr = bmp.auto_correlation(array, 0, t)
    cory = bmp.auto_correlation(array, 10, t)
    corblacl = bmp.auto_correlation(array, -10, t)
    plt.title(title)
    l1 = mpatches.Patch(color='g', label='y = 5')
    l2 = mpatches.Patch(color='b', label='y = -5')
    l3 = mpatches.Patch(color='r', label='y = 0')
    l4 = mpatches.Patch(color='y', label='y = 10')
    l5 = mpatches.Patch(color='black', label='y = -10')
    plt.legend(handles=[l1, l2, l3, l4, l5])
    plt.plot(t, corg, color='g')
    plt.plot(t, corb, color='b')
    plt.plot(t, corr, color='r')
    plt.plot(t, cory, color='y')
    plt.plot(t, corblacl, color='black')
    plt.savefig(f"images\\auto correlation {title}.png")
    plt.clf()


def componentHistogramsAndEntropy(rgb):
    y_cb_cr = bmp.convertYCbCr(rgb)
    x = range(0, 255)
    freq_blue = defaultdict(int)
    freq_green = defaultdict(int)
    freq_red = defaultdict(int)
    freq_cb = defaultdict(int)
    freq_cr = defaultdict(int)
    freq_y = defaultdict(int)
    for b, g, r, cb, cr, y in zip(rgb[..., 0].flatten(), rgb[..., 1].flatten(), rgb[..., 2].flatten(),
                                  y_cb_cr[..., 1].flatten(), y_cb_cr[..., 2].flatten(), y_cb_cr[..., 0].flatten()):
        freq_blue[b] += 1
        freq_green[g] += 1
        freq_red[r] += 1
        freq_cb[cb] += 1
        freq_cr[cr] += 1
        freq_y[y] += 1
    b = []
    g = []
    r = []
    cb = []
    cr = []
    y = []
    for i in x:
        b.append(freq_blue[i])
        g.append(freq_green[i])
        r.append(freq_red[i])
        cb.append(freq_cb[i])
        cr.append(freq_cr[i])
        y.append(freq_y[i])
    size = rgb[..., 0].size

    plt.bar(x, y)
    plt.title("y")
    plt.savefig("images\\y hist.png")
    plt.clf()
    plt.bar(x, b)
    plt.title("blue")
    plt.savefig("images\\blue hist.png")
    plt.clf()
    plt.bar(x, g)
    plt.title("green")
    plt.savefig("images\\green hist.png")
    plt.clf()
    plt.bar(x, r)
    plt.title("red")
    plt.savefig("images\\red hist.png")
    plt.clf()
    plt.bar(x, cb)
    plt.title("cb")
    plt.savefig("images\\cb hist.png")
    plt.clf()
    plt.bar(x, cr)
    plt.title("cr")
    plt.savefig("images\\cr hist.png")
    plt.savefig("images\\cr hist.png")
    plt.clf()

    entropy_b = -np.sum(b * np.ma.log2(b).filled(0))
    entropy_g = -np.sum(g * np.ma.log2(g).filled(0))
    entropy_r = -np.sum(r * np.ma.log2(r).filled(0))
    entropy_cb = -np.sum(cb * np.ma.log2(cb).filled(0))
    entropy_cr = -np.sum(cr * np.ma.log2(cr).filled(0))
    entropy_y = -np.sum(y * np.ma.log2(y).filled(0))
    print("\n \n \n")
    print("entropy b ", entropy_b)
    print("entropy g ", entropy_g)
    print("entropy r ", entropy_r)
    print("entropy cb ", entropy_cb)
    print("entropy cr ", entropy_cr)
    print("entropy y ", entropy_y)


def DPCM(rgb, func_type):
    y_cb_cr = bmp.convertYCbCr(rgb)
    r = rgb[..., 2]
    g = rgb[..., 1]
    b = rgb[..., 0]
    y = y_cb_cr[..., 0]
    cb = y_cb_cr[..., 1]
    cr = y_cb_cr[..., 2]
    hist_d_red = r[1:, 1:].astype(np.int32) - bmp.DPCM(r, func_type)
    hist_d_green = g[1:, 1:].astype(np.int32) - bmp.DPCM(g, func_type)
    hist_d_blue = r[1:, 1:].astype(np.int32) - bmp.DPCM(b, func_type)
    hist_d_y = r[1:, 1:].astype(np.int32) - bmp.DPCM(y, func_type)
    hist_d_cb = r[1:, 1:].astype(np.int32) - bmp.DPCM(cb, func_type)
    hist_d_cr = r[1:, 1:].astype(np.int32) - bmp.DPCM(cr, func_type)
    freq_blue = defaultdict(int)
    freq_green = defaultdict(int)
    freq_red = defaultdict(int)
    freq_cb = defaultdict(int)
    size = rgb[..., 0].size
    freq_cr = defaultdict(int)
    freq_y = defaultdict(int)
    for b, g, r, cb, cr, y in zip(hist_d_blue.flatten(), hist_d_green.flatten(), hist_d_red.flatten(),
                                  hist_d_cb.flatten(), hist_d_cr.flatten(), hist_d_y.flatten()):
        freq_blue[b] += 1
        freq_green[g] += 1
        freq_red[r] += 1
        freq_cb[cb] += 1
        freq_cr[cr] += 1
        freq_y[y] += 1
    b = []
    g = []
    r = []
    cb = []
    cr = []
    y = []
    x = range(-255, 255)
    for i in x:
        b.append(freq_blue[i])
        g.append(freq_green[i])
        r.append(freq_red[i])
        cb.append(freq_cb[i])
        cr.append(freq_cr[i])
        y.append(freq_y[i])
    for i in range(-255, 255):
        b[i] = b[i] / size
        g[i] = g[i] / size
        r[i] = r[i] / size
        cb[i] = cb[i] / size
        cr[i] = cr[i] / size
        y[i] = y[i] / size
    plt.bar(x, b)
    plt.title(f"BLUE DPCM type {func_type}")
    plt.savefig(f"images\\BLUE DPCM type {func_type}.png")
    plt.clf()
    plt.bar(x, g)
    plt.title(f"GREEN DPCM type {func_type}")
    plt.savefig(f"images\\GREEN DPCM type {func_type}.png")
    plt.clf()
    plt.bar(x, r)
    plt.title(f"RED DPCM type {func_type}")
    plt.savefig(f"images\\RED DPCM type {func_type}.png")
    plt.clf()
    plt.bar(x, cb)
    plt.title(f"CB DPCM type {func_type}")
    plt.savefig(f"images\\CB DPCM type {func_type}.png")
    plt.clf()
    plt.bar(x, cr)
    plt.title(f"CR DPCM type {func_type}")
    plt.savefig(f"images\\CR DPCM type {func_type}.png")
    plt.clf()

    entropy_b = -np.sum(b * np.ma.log2(b).filled(0))
    entropy_g = -np.sum(g * np.ma.log2(g).filled(0))
    entropy_r = -np.sum(r * np.ma.log2(r).filled(0))
    entropy_cb = -np.sum(cb * np.ma.log2(cb).filled(0))
    entropy_cr = -np.sum(cr * np.ma.log2(cr).filled(0))
    entropy_y = -np.sum(y * np.ma.log2(y).filled(0))
    print("\n \n \n")
    print(f"entropy type {func_type} of b ", entropy_b)
    print(f"entropy type {func_type} of g ", entropy_g)
    print(f"entropy type {func_type} of r ", entropy_r)
    print(f"entropy type {func_type} of cb ", entropy_cb)
    print(f"entropy type {func_type} of cr ", entropy_cr)
    print(f"entropy type {func_type} of y ", entropy_y)


header, rgb = bmp.readBmp(in_file)

a = rgb[...][...][0] - rgb[...][...][1]
print(np.max(a))
sys.exit(0)
y_cb_cr = bmp.convertYCbCr(rgb)
cbDecimated = bmp.decimationByDeletingEven(y_cb_cr[..., 1], 4)
crDecimated = bmp.decimationByDeletingEven(y_cb_cr[..., 2], 4)
cbRecovered = bmp.recoverDecimationByDeletingEven(cbDecimated, 4)
crRecovered = bmp.recoverDecimationByDeletingEven(crDecimated, 4)
y_cb_cr[..., 1] = cbRecovered
y_cb_cr[..., 2] = crRecovered
rgb2 = bmp.inverseConvertYCbCr(y_cb_cr)
bmp.writeBmp(res_file_path + "testRec.bmp", header, rgb2)
print(bmp.psnr(rgb[..., 0], rgb2[..., 0]))
print(bmp.psnr(rgb[..., 1], rgb2[..., 1]))
print(bmp.psnr(rgb[..., 2], rgb2[..., 2]))
sys.stdout = open('console.txt', 'w')

b = rgb[..., 0]
g = rgb[..., 1]
r = rgb[..., 2]
# [4] b
getAutoCor(r, "red")
getAutoCor(g, "green")
getAutoCor(b, "blue")

# 7 [y_cb_cr]
y_cb_cr = bmp.convertYCbCr(rgb)
rgbRec = bmp.inverseConvertYCbCr(y_cb_cr)
print("\n \n \n")
print("convert to ycbcr and reverse")
print("psnr blue ", bmp.psnr(b, rgbRec[..., 0]))
print("psnr green ", bmp.psnr(g, rgbRec[..., 1]))
print("psnr red ", bmp.psnr(r, rgbRec[..., 2]))

# [8 9 10] decimation and recovering

recovered = decimationYCbCr(rgb, 2)
recoveredAvg = avgDecimationYCbCr(rgb, 2)
bmp.writeBmp(res_file_path + "dec2.bmp", header, recovered)
bmp.writeBmp(res_file_path + "decAvg2.bmp", header, recoveredAvg)

# [11] decimation and recovering

recovered = decimationYCbCr(rgb, 4)
recoveredAvg = avgDecimationYCbCr(rgb, 4)
bmp.writeBmp(res_file_path + "dec4.bmp", header, recovered)
bmp.writeBmp(res_file_path + "decAvg4.bmp", header, recoveredAvg)

# [12 13] histograms and entropy

componentHistogramsAndEntropy(rgb)

# [14 15]

DPCM(rgb, 1)
DPCM(rgb, 2)
DPCM(rgb, 3)
DPCM(rgb, 4)
