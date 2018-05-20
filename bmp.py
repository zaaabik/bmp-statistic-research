import struct

import numpy as np


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
    if padding > 3:
        padding = 0
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


def correlation(a, b):
    exp_val_a = expectedValue(a)
    exp_val_b = expectedValue(b)

    res_top = np.sum((a - exp_val_a) * (b - exp_val_b))
    res_bot = np.sum(((a - exp_val_a) ** 2)) * np.sum((b - exp_val_b) ** 2)

    return res_top / np.sqrt(res_bot)


def expectedValue(value):
    return np.average(value)


def standardDeviation(value):
    exp_val = expectedValue(value)
    result = np.sum((value - exp_val) ** 2)
    result = np.sqrt(result / (value.size - 1))
    return result


def convertYCbCr(value):
    value = value.astype(np.int32)
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = value.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    np.putmask(ycbcr, ycbcr > 256, 255)
    return np.uint8(ycbcr)


def inverseConvertYCbCr(value):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = value.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb < 0, 0)
    np.putmask(rgb, rgb > 255, 255)
    return np.uint8(rgb)


def psnr(src, transformed):
    h = src.shape[0]
    w = src.shape[1]
    mse = np.mean((src - transformed) ** 2)
    if mse == 0:
        return 100
    MAX = 255.0
    return 20 * np.log10(MAX / np.sqrt(mse))


def decimationByDeletingEven(value, n):
    result = np.copy(value[::n, ::n])
    return result


def decimationByAverageValue(value, n):
    y = value.shape[0]
    x = value.shape[1]
    res = np.zeros((y, x), 'uint8')
    for i in range(0, y, n):
        for j in range(0, x, n):
            w = 0
            avg = 0
            if i + 1 < y:
                w += 1
                avg += value[i + 1][j]
            if i - 1 > 0:
                w += 1
                avg += value[i - 1][j]
            if j + 1 < x:
                w += 1
                avg += value[i][j + 1]
            if j - 1 > 0:
                w += 1
                avg += value[i][j - 1]
            res[i][j] = avg // w
    return decimationByDeletingEven(res, n)


def recoverDecimationByDeletingEven(value, n):
    y = value.shape[0]
    x = value.shape[1]
    res = np.zeros((y * n, x))
    for i in range(0, y):
        j = i * n
        for z in range(0, n):
            res[j + z, ...] = value[i, ...]
    res = np.hstack((res, np.zeros((res.shape[0], x * (n - 1)), dtype='uint8')))
    value = np.copy(res)
    for i in range(0, x):
        j = i * n
        for z in range(0, n):
            res[..., j + z] = value[..., i]
    return res


def auto_correlation(array, y, t):
    if y == 0:
        a = array
        b = array
    else:
        if y > 0:
            a = array[y:, ...]
            b = array[:-y, ...]
        else:
            a = array[:y, ...]
            b = array[-y:, ...]
    crl = []
    for i in t:
        if i != 0:
            crl.append((correlation(a[..., :-i], b[..., i:])))
        else:
            crl.append((correlation(a, b)))
    return crl


def DPCM(rgb, mode):
    # rgb = rgb[..., 0]
    y = rgb.shape[0]
    x = rgb.shape[1]
    if mode == 1:
        return rgb[1:, :-1]
    if mode == 2:
        return rgb[:-1, 1:]
    if mode == 3:
        return rgb[:-1, :-1]
    if mode == 4:
        res = np.ndarray((y - 1, x - 1), int)
        for i in range(1, y):
            for j in range(1, x):
                res[i - 1][j - 1] = (float(rgb[i][j - 1]) + float(rgb[i - 1][j]) + float(rgb[i - 1][j - 1])) / 3
        return res
