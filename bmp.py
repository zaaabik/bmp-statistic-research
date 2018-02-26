import struct

import numpy as np
from PIL import Image


def readBmp(path):
    bmp = open(path, "rb")
    header = {'typeOfPic': bmp.read(2).decode(),
              'size': struct.unpack('I', bmp.read(4)),
              'res1': struct.unpack('H', bmp.read(2)),
              'res2': struct.unpack('H', bmp.read(2)),
              'offset': struct.unpack('I', bmp.read(4)),
              'headerSize': struct.unpack('I', bmp.read(4)),
              'width': struct.unpack('I', bmp.read(4)),
              'height': struct.unpack('I', bmp.read(4)),
              'ColourPlanes': struct.unpack('H', bmp.read(2)),
              'bitsPerPixel': struct.unpack('H', bmp.read(2)),
              'compressionMethod': struct.unpack('I', bmp.read(4)),
              'rawImageSize': struct.unpack('I', bmp.read(4)),
              'horizontalResolution': struct.unpack('I', bmp.read(4)),
              'verticalResolution': struct.unpack('I', bmp.read(4)),
              'numberofColours': struct.unpack('I', bmp.read(4)),
              'importantColours': struct.unpack('I', bmp.read(4))
              }

    ba = np.fromfile(bmp, dtype='uint8')
    ar = np.array(ba)
    size = ar.size
    padding = size - header['rawImageSize'][0]
    ar = np.delete(ar, range(size - 1 - padding, size - 1))
    width = header['width'][0]
    padding_size = (width * 3) % 4
    row_len = padding_size + width * 3
    ar = ar.reshape(-1, row_len)
    ar = np.delete(ar, range(row_len - padding_size - 1, row_len - 1), 1)
    ar = ar.reshape(-1, row_len // 3, 3)
    return header, ar


def writeBmp(path, header, rgb_array):
    packed_header = {
        'typeOfPic': header['typeOfPic'].encode(),
        'size': struct.pack('I', header['size'][0]),
        'res1': struct.pack('H', header['res1'][0]),
        'res2': struct.pack('H', header['res2'][0]),
        'offset': struct.pack('I', header['offset'][0]),
        'headerSize': struct.pack('I', header['headerSize'][0]),
        'width': struct.pack('I', header['width'][0]),
        'height': struct.pack('I', header['height'][0]),
        'ColourPlanes': struct.pack('H', header['ColourPlanes'][0]),
        'bitsPerPixel': struct.pack('H', header['bitsPerPixel'][0]),
        'compressionMethod': struct.pack('I', header['compressionMethod'][0]),
        'rawImageSize': struct.pack('I', header['rawImageSize'][0]),
        'horizontalResolution': struct.pack('I', header['horizontalResolution'][0]),
        'verticalResolution': struct.pack('I', header['verticalResolution'][0]),
        'numberofColours': struct.pack('I', header['numberofColours'][0]),
        'importantColours': struct.pack('I', header['importantColours'][0])
    }
    padding_len = rgb_array.shape[1] % 4
    rgb_array = rgb_array.reshape(rgb_array.shape[0], rgb_array.shape[1] * 3)
    rgb_array = np.hstack((rgb_array, np.zeros((rgb_array.shape[0], padding_len), dtype='uint8')))

    out = open(path, "wb")
    for key, val in packed_header.items():
        out.write(val)
    out.write(rgb_array.tobytes())


def getRgb(src_path):
    im = Image.open(src_path)
    return np.array(im)


def saveYCbCr(matrix, full_path):
    x = matrix.shape[0]
    y = matrix.shape[1]
    res_array = np.zeros((x, y, 3), 'uint8')
    res_array[..., 0] = matrix
    res_array[..., 1] = matrix
    res_array[..., 2] = matrix

    image = Image.fromarray(res_array, mode="RGB")
    image.save(full_path)


def correlation(a, b):
    exp_val_a = expectedValue(a)
    exp_val_b = expectedValue(b)

    res_top = 0
    res_bot1 = 0
    res_bot2 = 0
    for x, y in zip(np.nditer(a), np.nditer(b)):
        res_top += (x - exp_val_a) * (y - exp_val_b)
        res_bot1 += ((x - exp_val_a) ** 2)
        res_bot2 += ((y - exp_val_b) ** 2)

    return res_top / np.sqrt(res_bot1 * res_bot2)


def expectedValue(value):
    h = value.shape[0]
    w = value.shape[1]
    result = 0.0
    for j in range(0, h):
        for i in range(0, w):
            result += value[j][i]
    return result / (w * h)


def standardDeviation(value):
    h = value.shape[0]
    w = value.shape[1]
    exp_val = expectedValue(value)

    result = 0.0
    for j in range(0, h):
        for i in range(0, w):
            result += ((value[j][i] - exp_val) ** 2)

    result = np.sqrt(result / (w * h - 1))
    return result


def separateColors(src_path, dst_path):
    im = Image.open(src_path)
    r, g, b = im.split()
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    x = r.shape[0]
    y = r.shape[1]
    red_map = np.zeros((x, y, 3), 'uint8')
    red_map[..., 0] = r
    blue_map = np.zeros((x, y, 3), 'uint8')
    blue_map[..., 2] = b
    green_map = np.zeros((x, y, 3), 'uint8')
    green_map[..., 1] = g
    im_red = Image.fromarray(red_map, mode="RGB")
    im_blue = Image.fromarray(blue_map, mode="RGB")
    im_green = Image.fromarray(green_map, mode="RGB")
    im_red.save(dst_path + "\\r.bmp")
    im_blue.save(dst_path + "\\b.bmp")
    im_green.save(dst_path + "\\g.bmp")


def getColorArray(path, color_name):
    colors_number = {'r': 0, 'g': 1, 'b': 2}
    im = Image.open(path)
    p = np.array(im)
    return p[..., colors_number[color_name]]


def convertYCbCr(value):
    h = value.shape[0]
    w = value.shape[1]

    result = np.zeros((h, w, 3), 'uint8')

    for i in np.arange(0, h):
        for j in np.arange(0, w):
            pixel = value[i][j]
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            result[i][j][0] = 0.299 * r + 0.587 * g + 0.114 * b
            tmp = 0.5643 * (b - float(result[i][j][0])) + 128
            if tmp > 255:
                result[i][j][1] = 255
            else:
                result[i][j][1] = tmp
            tmp2 = 0.7132 * (r - float(result[i][j][0])) + 128
            if tmp2 > 255:
                result[i][j][2] = 255
            else:
                result[i][j][2] = tmp2
    return result


def inverseConvertYCbCr(value):
    h = value.shape[0]
    w = value.shape[1]

    result = np.zeros((h, w, 3), 'uint8')

    for i in range(0, h):
        for j in range(0, w):
            pixel = value[i][j]
            y = pixel[0]
            cb = pixel[1]
            cr = pixel[2]
            result[i][j][0] = y + 1.402 * (cr - 128)
            result[i][j][1] = y - 0.714 * (cr - 128) - 0.334 * (cb - 128)
            result[i][j][2] = y + 1.772 * (cb - 128)

    return result


def psnr(src, transformed):
    h = src.shape[0]
    w = src.shape[1]
    l = 8
    bottom = 0
    for i in range(0, h):
        for j in range(0, w):
            bottom += ((float(src[i][j]) - transformed[i][j]) ** 2)
    top = w * h * ((2 ** l) - 1) ** 2
    return 10 * np.log10(top / bottom)
