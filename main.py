import datetime

import numpy as np

import bmp

in_file = "C:\\Users\\HawkA\\OneDrive\\Pictures\\pyTest\\mini.bmp"
res_file = "C:\\Users\\HawkA\\OneDrive\\Pictures\\pyTest\\lastRes.bmp"
res_file_path = "C:\\Users\\HawkA\\OneDrive\\Pictures\\pyTest\\"

# 1.2
header, rgb = bmp.readBmp(in_file)

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
r = r[:-5, :-5]
g = g[5:, 5:]
a = datetime.datetime.now()
cor_r_g = bmp.correlation(r, g)
b = datetime.datetime.now()
print("time", b - a)
print("correlation")
print("r h", cor_r_g)
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
print("psnr red ", psnr_red)
print("psnr green", psnr_green)
print("psnr blue", psnr_blue)

# 1.8

