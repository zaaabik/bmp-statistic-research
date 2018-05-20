import numpy as np
from scipy import interpolate
from scipy import signal


class BmpFilters:
    @staticmethod
    def __set_kernel(array, kernel):
        res = signal.convolve2d(array.astype(np.uint8), kernel)
        dif_y = res.shape[0] - array.shape[0]
        dif_x = res.shape[1] - array.shape[1]
        dif_y //= 2
        dif_x //= 2
        return res[dif_y:-dif_y, dif_x:-dif_x]

    @staticmethod
    def gauss_filter(rgb, r, d):
        y = rgb.shape[0]
        x = rgb.shape[1]
        diam = r * 2 + 1

        kernel = np.zeros((diam, diam))
        center = r
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                kernel[center + i, center + j] = np.exp(-(i ** 2 + j ** 2) / 2 * d ** 2)
        kernel_sum = np.sum(kernel)
        b = rgb[..., 0]
        g = rgb[..., 1]
        r = rgb[..., 2]
        r = BmpFilters.__set_kernel(r, kernel)
        g = BmpFilters.__set_kernel(g, kernel)
        b = BmpFilters.__set_kernel(b, kernel)
        res_rgb = np.zeros((y, x, 3), dtype=np.uint8)
        res_rgb[..., 0] = b // kernel_sum
        res_rgb[..., 1] = g // kernel_sum
        res_rgb[..., 2] = r // kernel_sum
        np.putmask(res_rgb, res_rgb > 255, 255)
        np.putmask(res_rgb, res_rgb < 0, 0)
        return res_rgb

    @staticmethod
    def avg(rgb, r):
        y = rgb.shape[0]
        x = rgb.shape[1]
        r = r * 2 + 1
        kernel = np.ones((r, r))
        kernel_sum = np.sum(kernel)
        b = rgb[..., 0]
        g = rgb[..., 1]
        r = rgb[..., 2]
        r = BmpFilters.__set_kernel(r, kernel)
        g = BmpFilters.__set_kernel(g, kernel)
        b = BmpFilters.__set_kernel(b, kernel)
        res_rgb = np.zeros((y, x, 3), dtype='uint8')
        res_rgb[..., 0] = b / kernel_sum
        res_rgb[..., 1] = g / kernel_sum
        res_rgb[..., 2] = r / kernel_sum
        return res_rgb

    @staticmethod
    def __add_impulse(array, pa, pb):
        rand = np.random.choice([255, 0, 128], (array.shape[0], array.shape[1]), p=[pa, pb, 1 - pa - pb])
        res = np.empty_like(rand)

        for idx, a in np.ndenumerate(rand):
            if a == 255:
                res[idx] = 255
            if a == 0:
                res[idx] = 0
            if a == 128:
                res[idx] = array[idx]

        return res

    @staticmethod
    def add_impulse_noise(rgb, pa, pb):
        b = rgb[..., 0]
        g = rgb[..., 1]
        r = rgb[..., 2]
        r = BmpFilters.__add_impulse(r, pa, pb)
        g = BmpFilters.__add_impulse(g, pa, pb)
        b = BmpFilters.__add_impulse(b, pa, pb)
        res = np.empty_like(rgb)
        res[..., 0] = b
        res[..., 1] = g
        res[..., 2] = r
        return res

    @staticmethod
    def __add_gauss(array, d, m):
        s = (array.shape[0], array.shape[1])
        g = np.random.normal(loc=m, scale=d, size=s)
        res = np.zeros(s, dtype='float')
        res += array
        res += g
        np.putmask(res, res < 0, 0)
        np.putmask(res, res > 255, 255)
        return np.uint8(res)

    @staticmethod
    def add_gauss_noise(rgb, d, m):
        b = rgb[..., 0]
        g = rgb[..., 1]
        r = rgb[..., 2]
        r = BmpFilters.__add_gauss(r, d, m)
        g = BmpFilters.__add_gauss(g, d, m)
        b = BmpFilters.__add_gauss(b, d, m)
        res = np.empty_like(rgb)
        res[..., 0] = b
        res[..., 1] = g
        res[..., 2] = r
        return res

    @staticmethod
    def median_filter_test(rgb, rad):
        b = rgb[..., 0]
        g = rgb[..., 1]
        r = rgb[..., 2]
        resB = BmpFilters.__median_filter(b, rad)
        resG = BmpFilters.__median_filter(g, rad)
        resR = BmpFilters.__median_filter(r, rad)
        res = np.empty_like(rgb)
        res[..., 0] = resB
        res[..., 1] = resG
        res[..., 2] = resR
        return res

    @staticmethod
    def __get_neighbors_matrix(m, r, x, y):
        return m[y - r + r:y + r + 1 + r, x - r + r:x + r + 1 + r]

    @staticmethod
    def __median_filter(m, r):
        x_size = m.shape[1]
        y_size = m.shape[0]
        res = np.zeros_like(m)
        yRng = range(0, y_size)
        xRng = range(0, x_size)
        m = np.vstack([np.zeros((r, x_size)), m])
        m = np.vstack([m, np.zeros((r, x_size))])
        m = np.hstack([np.zeros((y_size + 2 * r, r)), m])
        m = np.hstack([m, np.zeros((y_size + 2 * r, r))])
        for i in yRng:
            for j in xRng:
                res[j, i] = np.median(BmpFilters.__get_neighbors_matrix(m, r, i, j))
        return res

    @staticmethod
    def median_filter(rgb, rad):
        b = rgb[..., 0]
        g = rgb[..., 1]
        r = rgb[..., 2]
        r = signal.medfilt2d(r, rad * 2 + 1)
        g = signal.medfilt2d(g, rad * 2 + 1)
        b = signal.medfilt2d(b, rad * 2 + 1)
        res = np.empty_like(rgb)
        res[..., 0] = b
        res[..., 1] = g
        res[..., 2] = r
        return res

    @staticmethod
    def laplas_operator(rgb, a):
        kernel = np.array(
            [[0, -1, 0],
             [-1, a + 4, -1],
             [0, -1, 0]]
        )
        b = rgb[..., 0]
        g = rgb[..., 1]
        r = rgb[..., 2]
        r = BmpFilters.__set_kernel(r, kernel)
        g = BmpFilters.__set_kernel(g, kernel)
        b = BmpFilters.__set_kernel(b, kernel)
        res = np.zeros_like(rgb, dtype=np.float64)
        res[..., 0] = b
        res[..., 1] = g
        res[..., 2] = r
        np.putmask(res, res > 255, 255)
        np.putmask(res, res < 0, 0)
        return res.astype(np.uint8)

    @staticmethod
    def max_freq(rgb):
        laplas = BmpFilters.laplas_operator(rgb, 0)
        res = np.zeros_like(rgb, dtype='float')
        res += rgb
        res += laplas
        np.putmask(res, res < 0, 0)
        np.putmask(res, res > 255, 255)
        return res.astype(np.uint8)

    @staticmethod
    def sobel(rgb, thr):
        kernel_horiz = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ])
        kernel_vertical = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ])
        b = rgb[..., 0]
        g = rgb[..., 1]
        r = rgb[..., 2]
        r_h = BmpFilters.__set_kernel(r, kernel_horiz)
        g_h = BmpFilters.__set_kernel(g, kernel_horiz)
        b_h = BmpFilters.__set_kernel(b, kernel_horiz)
        r_v = BmpFilters.__set_kernel(r, kernel_vertical)
        g_v = BmpFilters.__set_kernel(g, kernel_vertical)
        b_v = BmpFilters.__set_kernel(b, kernel_vertical)
        h = np.empty_like(rgb, dtype='float')
        v = np.empty_like(rgb, dtype='float')
        h[..., 0] = b_h
        h[..., 1] = g_h
        h[..., 2] = r_h
        v[..., 0] = b_v
        v[..., 1] = g_v
        v[..., 2] = r_v
        res = np.zeros_like(rgb, dtype='float')
        res += np.sqrt(h ** 2 + v ** 2)
        np.putmask(res, res < thr, 0)
        np.putmask(res, res >= thr, 255)
        angles = np.degrees(np.arctan2(v, h))
        quad = np.vectorize(make_quad)
        a = quad(angles)
        return res.astype(np.uint8), make_rgb_map(a, res[..., 0], 127)

    @staticmethod
    def add_bright(rgb, c):
        res = np.array(rgb) * c
        np.putmask(res, res > 255, 255)
        return res.astype(np.uint8)

    @staticmethod
    def two_dot(rgb, func):
        f = np.vectorize(func)
        return f(rgb).astype(np.uint8)

    @staticmethod
    def gamma(rgb, c, y):
        func = lambda x: pow(c * x, y)
        f = np.vectorize(func)
        rgb = np.array(rgb)
        rgb = rgb / 255
        res = f(rgb)
        res *= 255
        return res.astype(np.uint8)

    @staticmethod
    def hist(table, rgb):
        f = lambda x: table[x]
        func = np.vectorize(f)
        return func(rgb).astype(np.uint8)

    @staticmethod
    def create_look_up(histogram):
        table = np.zeros(2 ** 8, np.int32)
        sum = 0
        for i in range(0, 2 ** 8):
            sum += histogram[i] * 2 ** 8
            table[i] = sum
        np.putmask(table, table > 2 ** 8, 2 ** 8)
        return table.astype(np.uint8)


def two_dot_func(a, b):
    return interpolate.interp1d([0, a[1], b[1], 255], [0, a[0], b[0], 255])


def make_quad(x):
    if -180 <= x <= -90:
        return 1
    if -90 < x <= 0:
        return 2
    if 0 < x <= 90:
        return 3
    if 90 < x <= 180:
        return 4


def make_rgb_map(angles, grad_len, thr):
    res = np.zeros_like(angles, dtype=np.uint8)
    angles = angles[..., 0]
    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            if grad_len[i, j] < 255:
                continue
            if angles[i, j] == 1:
                res[i, j, 0] = 255
            if angles[i, j] == 2:
                res[i, j, 1] = 255
            if angles[i, j] == 3:
                res[i, j, 2] = 255
            if angles[i, j] == 4:
                res[i, j, 0] = 255
                res[i, j, 1] = 255
                res[i, j, 2] = 255
    return res
