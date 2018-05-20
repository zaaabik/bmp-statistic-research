from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import BmpFilters
import bmp

orig_filename = 'lena.bmp'
median_folder = 'median'
original_file_y_cb_cr = 'lena y_cb_cr.bmp'
gauss_noise_filename = "gauss/gauss d =100.bmp"
moving_average_folder = "moving average"
gauss_noise_psnr_filename = "gauss/psnr.png"
impulse_noise_filename = "impulse/impulse.bmp"
gauss_filter_filename = "gauss filter"
laplas_folder = "laplas"
impule_noise_filename = 'impulse pa = 0.1 pb = 0.2.png'

_, orig_pic = bmp.readBmp(orig_filename)


def avg():
    file_name = gauss_noise_filename
    header, rgb = bmp.readBmp(file_name)

    psnr = []
    rng = range(1, 25)
    y_cb_cr = bmp.convertYCbCr(rgb)
    orinal_y_cb_cr = bmp.convertYCbCr(orig_pic)
    for i in rng:
        res = BmpFilters.BmpFilters.avg(y_cb_cr, i)
        psnr.append(bmp.psnr(orinal_y_cb_cr[..., 0], res[..., 0]))
        res[..., 1] = res[..., 0]
        res[..., 2] = res[..., 0]
    # bmp.writeBmp(moving_average_folder + f"/avg {i}.bmp", header, res)

    x = np.array(list(rng), dtype='uint')
    y = np.array(psnr)
    np.savetxt(moving_average_folder + "/psnr.csv", (x, y), delimiter=',', fmt='%.2f')
    plt.plot(list(rng), psnr)
    plt.savefig(moving_average_folder + "/psnr.png")


def impulse_noise():
    # hist = defaultdict(list)
    # x = defaultdict(list)
    # hist[0.1].append(1)
    # hist[0.1].append(2)
    # x[0.1].append(3)
    # x[0.1].append(6)
    # x[0.1].append(6)
    # for (k, v), (k2, v2) in zip(hist.items(), x.items()):
    #     print(k, v, k2, v2)
    #     print('\n')
    #
    # return
    small = 0.00001
    file_name = orig_filename
    header, rgb = bmp.readBmp(file_name)
    y_cb_cr = bmp.convertYCbCr(rgb)
    hist = defaultdict(list)
    x = defaultdict(list)
    rng = np.arange(0, 1.1, 0.1)
    for i in rng:
        for j in rng:
            if i + j >= 1:
                continue
            if i == j == 0:
                i = small
                j = small

            res = BmpFilters.BmpFilters.add_impulse_noise(y_cb_cr, i, j)
            p = bmp.psnr(y_cb_cr[..., 0], res[..., 0])
            if i == j == small:
                i = 0
                j = 0

            x[i].append(j)
            hist[i].append(p)
            res[..., 1] = res[..., 0]
            res[..., 2] = res[..., 0]
            bmp.writeBmp('impulse/impulse pa = {0:.1f} pb = {1:.1f}.png'.format(float(i), float(j)), header, res)
    l = []
    for (k, v), (k1, v1) in zip(x.items(), hist.items()):
        leg, = plt.plot(list(v), list(v1), label='pa = {0:.1}'.format(float(k)))
        l.append(leg)

    plt.legend(handles=l)
    plt.savefig('impulse/psnr.png')


def gauss_noise():
    header, rgb = bmp.readBmp(orig_filename)
    y_cb_cr = bmp.convertYCbCr(rgb)
    psnr = []
    rng = range(0, 250, 10)

    for i in rng:
        if i == 0:
            i = 0.0000001
        res = BmpFilters.BmpFilters.add_gauss_noise(y_cb_cr, d=i, m=0)
        res[..., 1] = res[..., 0]
        res[..., 2] = res[..., 0]
        p = bmp.psnr(y_cb_cr[..., 0], res[..., 0])
        psnr.append(p)
        if i == 0.0000001:
            i = 0
        bmp.writeBmp(f"gauss/gauss d ={i}.bmp", header, res)
    plt.plot(list(rng), psnr)
    plt.savefig(gauss_noise_psnr_filename)


def gauss_filter():
    header, y_cb_cr = bmp.readBmp(gauss_noise_filename)
    h, y_cb_cr_o = bmp.readBmp(original_file_y_cb_cr)
    sigmaRange = np.arange(0.0001, 1.1, 0.1)
    psnr = defaultdict(list)
    rRange = range(1, 10)
    legends = []
    out_psnr = np.zeros((len(list(rRange)) + 1, (len(list(sigmaRange)) + 1)))
    out_psnr[1:, 0] = list(rRange)
    out_psnr[0, 1:] = list(sigmaRange)
    print('orig ', bmp.psnr(y_cb_cr_o[..., 0], y_cb_cr[..., 0]))
    for r in rRange:
        psnr = []
        for i in sigmaRange:
            res = BmpFilters.BmpFilters.gauss_filter(y_cb_cr, r, i)
            ps = bmp.psnr(y_cb_cr_o[..., 0], res[..., 0])
            res[..., 1] = res[..., 0]
            res[..., 2] = res[..., 0]
            bmp.writeBmp(gauss_filter_filename + "/gauss r = {0:.2f} d = {1:.2f}.bmp".format(r, i), header, res)
            psnr.append(ps)
        p = np.array(psnr)
        out_psnr[r, 1:] = p
        l, = plt.plot(list(sigmaRange), psnr, label=f'R = {r}')
        legends.append(l)
    plt.legend(handles=legends)
    plt.savefig(gauss_filter_filename + f"/psnr total.png")
    np.savetxt(gauss_filter_filename + '/psnr.csv', out_psnr, delimiter=',', fmt='%.2f')


def median():
    header, y_cb_cr = bmp.readBmp(gauss_noise_filename)
    header_o, y_cb_cr_o = bmp.readBmp(original_file_y_cb_cr)
    psnr = []
    print('orig', bmp.psnr(y_cb_cr_o[..., 0], y_cb_cr[..., 0]))
    x = range(1, 10)
    for r in x:
        res = BmpFilters.BmpFilters.median_filter(y_cb_cr, r * 2 + 1)
        res[..., 1] = res[..., 0]
        res[..., 2] = res[..., 0]
        psnr.append(bmp.psnr(y_cb_cr_o[..., 0], res[..., 0]))
        bmp.writeBmp(median_folder + f"/median r = {r}.bmp", header, res)
    np.savetxt(median_folder + "/psnr.csv", np.array(psnr).T, delimiter=',', fmt='%.2f')
    plt.plot(list(x), psnr)
    plt.savefig(median_folder + "/psnr.png")


def laplas():
    header, rgb = bmp.readBmp(orig_filename)
    y_cb_cr = bmp.convertYCbCr(rgb)
    y_cb_cr_res = BmpFilters.BmpFilters.laplas_operator(y_cb_cr, 0)
    y_cb_cr_res += 128
    np.putmask(y_cb_cr_res, y_cb_cr_res < 0, 0)
    np.putmask(y_cb_cr_res, y_cb_cr_res > 255, 255)
    y_cb_cr_res[..., 1] = y_cb_cr_res[..., 0]
    y_cb_cr_res[..., 2] = y_cb_cr_res[..., 0]
    y_cb_cr[..., 1] = y_cb_cr[..., 0]
    y_cb_cr[..., 2] = y_cb_cr[..., 0]
    bmp.writeBmp(laplas_folder + "/laplas.bmp", header, y_cb_cr_res)
    bmp.writeBmp(laplas_folder + "/orig.bmp", header, y_cb_cr)


def max_freq():
    header, y_cb_cr = bmp.readBmp(original_file_y_cb_cr)
    y_cb_cr_max = BmpFilters.BmpFilters.max_freq(y_cb_cr)
    y_cb_cr_max[..., 1] = y_cb_cr_max[..., 0]
    y_cb_cr_max[..., 2] = y_cb_cr_max[..., 0]
    y, x = make_hist(y_cb_cr[..., 0])
    y2, x2 = make_hist(y_cb_cr_max[..., 0])
    plt.bar(x, y)
    plt.show()
    plt.bar(x2, y2)
    plt.show()
    bmp.writeBmp(laplas_folder + "/max.bmp", header, y_cb_cr_max)


def lapslas_with_aplha():
    header, y_cb_cr = bmp.readBmp(original_file_y_cb_cr)
    header, max_y_cb_cr = bmp.readBmp(laplas_folder + "/max.bmp")
    y, x = make_hist(max_y_cb_cr[..., 0])
    plt.bar(x, y)
    plt.savefig(laplas_folder + "/max freq .png")
    plt.clf()
    return
    psnr = []
    bright = []
    y, x = make_hist(y_cb_cr[..., 0])
    bright.append(np.average(y_cb_cr))
    # plt.bar(x, y)
    # plt.savefig(laplas_folder + "/original_hist.png")
    # plt.clf()
    x = np.arange(0, 1.6, 0.1)

    for i in x:
        y_cb_cr_max = BmpFilters.BmpFilters.laplas_operator(y_cb_cr, i)
        bright.append(np.average(y_cb_cr_max))
        y_cb_cr_max[..., 1] = y_cb_cr_max[..., 0]
        y_cb_cr_max[..., 2] = y_cb_cr_max[..., 0]
        y, x = make_hist(y_cb_cr_max[..., 0])
        # plt.bar(x, y)
        # plt.savefig(laplas_folder + "/hist alpha {:.1f}.png".format(i))
        # plt.clf()

        # psnr.append(bmp.psnr(y_cb_cr_max[..., 0], max_y_cb_cr[..., 0]))
        # bmp.writeBmp(laplas_folder + "/alpha {:.1f}.bmp".format(i), header, y_cb_cr_max)
    x = np.arange(0, 1.6, 0.1)
    x = list(x)
    x.insert(0, -1)
    x = np.array(x, dtype=np.float)
    y = np.array(bright)

    print(len(x))
    print(len(y))
    np.savetxt(laplas_folder + "/bright average.csv",
               (x.T, y.T), delimiter=',', fmt='%.2f')
    # plt.plot(list(x), psnr)
    # plt.savefig(laplas_folder + "/psnr max vs alpha.png")


def make_hist(y):
    hist = defaultdict(int)
    size = len(y.flatten())
    for i in range(0, 256):
        hist[i] = 0
    for i in y.flatten():
        hist[i] += 1

    return np.array(list(hist.values())) / size, list(hist.keys())


def sobel():
    folder = "sobel/"
    # file = "test.bmp"
    header, y_cb_cr = bmp.readBmp(original_file_y_cb_cr)
    x = np.arange(0, 255, 5)
    # for i in x:
    y_cb_cr_max, angle_map = BmpFilters.BmpFilters.sobel(y_cb_cr, 127)
    y_cb_cr_max[..., 1] = y_cb_cr_max[..., 0]
    y_cb_cr_max[..., 2] = y_cb_cr_max[..., 0]
    # bmp.writeBmp(folder + "sobel thr = {:d}.bmp".format(127), header, y_cb_cr_max)
    bmp.writeBmp(folder + "angle.bmp", header, angle_map)


def impulse_with_persent():
    folder = 'impulse2/'
    header, y_cb_cr = bmp.readBmp(original_file_y_cb_cr)
    pa = [0.02, 0.05, 0.2, 0.3]
    pb = [0.03, 0.05, 0.05, 0.2]
    psnr = []
    x = []
    for a, b in zip(pa, pb):
        res = BmpFilters.BmpFilters.add_impulse_noise(y_cb_cr, a, b)
        psnr.append(bmp.psnr(y_cb_cr[..., 0], res[..., 0]))
        x.append((a + b) * 100)
        res[..., 1] = res[..., 0]
        res[..., 2] = res[..., 0]
        bmp.writeBmp(folder + f'impulse % = {(a + b) * 100}.bmp', header, res)
    plt.plot(x, psnr)
    plt.savefig(folder + 'psnr.png')


def median_for_impulse():
    header, y_orig = bmp.readBmp(original_file_y_cb_cr)
    folder = 'impulse2/'
    folder_res = 'impulse2/res/'
    pa = [0.02, 0.05, 0.2, 0.3]
    pb = [0.03, 0.05, 0.05, 0.2]
    for a, b in zip(pa, pb):
        percent = (a + b) * 100
        h, y = bmp.readBmp(folder + f'impulse % = {percent}.bmp')
        psnr = []
        x = []
        labels = []
        print(f'original {percent}', bmp.psnr(y_orig, y))
        x.append(0)
        psnr.append(bmp.psnr(y_orig, y))
        for i in range(1, 10):
            res = BmpFilters.BmpFilters.median_filter(y, i)
            p = bmp.psnr(y_orig[..., 0], res[..., 0])

            x.append(i)
            psnr.append(p)

            res[..., 1] = res[..., 0]
            res[..., 2] = res[..., 0]
            bmp.writeBmp(folder_res + f' % {percent} r = {i}.bmp', header, res)
        labels.append(plt.plot(x, psnr, label=f'r = {i}'))
    plt.savefig(folder + 'psnr.png', handlers=labels)


def white_and_black_generation():
    folder = 'C:\\Users\\HawkA\\PycharmProjects\\bmp-stats\\bright\\'
    high_bright_file = 'lenaHigh.bmp'
    low_bright_file = 'lenaLow.bmp'
    high_coeff = 2.0
    low_coeff = 0.2

    header, high_bright = bmp.readBmp(folder + high_bright_file)
    y_high, x_high = make_hist(high_bright[..., 0])
    plt.bar(x_high, y_high)
    plt.savefig(folder + 'original_high_hist.png')
    plt.clf()

    header, low_bright = bmp.readBmp(folder + low_bright_file)
    y_low, x_low = make_hist(low_bright[..., 0])
    plt.bar(x_low, y_low)
    plt.savefig(folder + 'original_low_hist.png')
    plt.clf()

    dot_a = (40, 65)
    dot_b = (210, 180)
    f = BmpFilters.two_dot_func(dot_a, dot_b)

    r = range(0, 255)
    y_grad = list(f(r))
    x = list(r)
    plt.plot(x, y_grad)
    plt.savefig(folder + 'grad.png')
    plt.clf()

    low_fnc = list(map(lambda x: x * low_coeff, list(range(0, 255))))
    x = list(r)
    axes = plt.gca()
    axes.set_xlim([0, 260])
    axes.set_ylim([0, 260])
    plt.plot(x, low_fnc)
    plt.savefig(folder + 'low fnc.png')
    plt.clf()

    high_fnc = np.array(list(map(lambda x: x * high_coeff, list(range(0, 255)))))
    x = list(r)
    np.putmask(high_fnc, high_fnc > 255, 255)
    axes = plt.gca()
    axes.set_xlim([0, 260])
    axes.set_ylim([0, 260])
    plt.plot(x, high_fnc)
    plt.savefig(folder + 'high fnc.png')
    plt.clf()

    high_grad = f(np.array(high_fnc))
    x = list(r)
    np.putmask(high_grad, high_grad > 255, 255)
    axes = plt.gca()
    axes.set_xlim([0, 260])
    axes.set_ylim([0, 260])
    plt.plot(x, high_grad)
    plt.savefig(folder + 'high fnc  grad.png')
    plt.clf()

    low_grad = f(np.array(low_fnc))
    x = list(r)
    np.putmask(low_grad, low_grad > 255, 255)
    axes = plt.gca()
    axes.set_xlim([0, 260])
    axes.set_ylim([0, 260])
    plt.plot(x, low_grad)
    plt.savefig(folder + 'low fnc  grad.png')
    plt.clf()

    low_bright_grad = BmpFilters.BmpFilters.two_dot(low_bright, f)
    y, x = make_hist(low_bright_grad[..., 0])
    plt.bar(x, y)
    plt.savefig(folder + 'low_grad_hist.png')
    plt.clf()
    high_bright_grad = BmpFilters.BmpFilters.two_dot(high_bright, f)

    y, x = make_hist(high_bright_grad[..., 0])
    plt.bar(x, y)
    plt.savefig(folder + 'high_grad_hist.png')
    plt.clf()

    bmp.writeBmp(folder + 'low grad.bmp', header, low_bright_grad)
    bmp.writeBmp(folder + 'high grad.bmp', header, high_bright_grad)


def gamma():
    folder = 'C:\\Users\\HawkA\\PycharmProjects\\bmp-stats\\gamma\\'
    lena_low = 'lenaLow.bmp'
    lena_high = 'lenaHigh.bmp'
    lena = 'lena.bmp'
    mid_folder = 'mid\\'
    low_folder = 'low\\'
    high_folder = 'high\\'

    h, y_high = bmp.readBmp(folder + lena_high)
    h, y_low = bmp.readBmp(folder + lena_low)
    h, y_orig = bmp.readBmp(folder + lena)

    y, x = make_hist(y_high[..., 0])
    plt.bar(x, y)
    plt.savefig(folder + 'original high.png')
    plt.clf()

    y, x = make_hist(y_low[..., 0])
    plt.bar(x, y)
    plt.savefig(folder + 'original low.png')
    plt.clf()

    y, x = make_hist(y_orig[..., 0])
    plt.bar(x, y)
    plt.savefig(folder + 'original.png')
    plt.clf()

    # d = np.array(list(range(0,255)))
    # plt.plot(d, BmpFilters.BmpFilters.gamma(d,1,25))
    # plt.show()

    y_rng = np.arange(1, 25, 0.5)
    for i in y_rng:
        res_high = BmpFilters.BmpFilters.gamma(y_high, 1, i)
        res_low = BmpFilters.BmpFilters.gamma(y_low, 1, i)
        orig = BmpFilters.BmpFilters.gamma(y_orig, 1, i)

        bmp.writeBmp(folder + high_folder + f'lena y = {i:.2f}.bmp', h, res_high)
        bmp.writeBmp(folder + low_folder + f'lena y = {i:.2f}.bmp', h, res_low)
        bmp.writeBmp(folder + mid_folder + f'lena y = {i:.2f}.bmp', h, orig)

        y, x = make_hist(res_high[..., 0])
        plt.bar(x, y)
        plt.savefig(folder + high_folder + f"high {i:.2f}.png")
        plt.clf()

        y, x = make_hist(res_low[..., 0])
        plt.bar(x, y)
        plt.savefig(folder + low_folder + f"low {i:.2f}.png")
        plt.clf()

        y, x = make_hist(orig[..., 0])
        plt.bar(x, y)
        plt.savefig(folder + mid_folder + f"orig {i:.2f}.png")
        plt.clf()


def hist():
    folder = 'C:\\Users\\HawkA\\PycharmProjects\\bmp-stats\\hist\\'
    lena_low = 'lenaLow.bmp'
    lena_high = 'lenaHigh.bmp'
    lena = 'lena.bmp'

    h, y_high = bmp.readBmp(folder + lena_high)
    h, y_low = bmp.readBmp(folder + lena_low)
    h, y_orig = bmp.readBmp(folder + lena)

    y_high_hist, x = make_hist(y_high[..., 0])
    plt.bar(x, y_high_hist)
    plt.savefig(folder + 'original high.png')
    plt.clf()

    y_low_hist, x = make_hist(y_low[..., 0])
    plt.bar(x, y_low_hist)
    plt.savefig(folder + 'original low.png')
    plt.clf()

    y_mid_hist, x = make_hist(y_orig[..., 0])
    plt.bar(x, y_mid_hist)
    plt.savefig(folder + 'original.png')
    plt.clf()

    look_up_table_mid = BmpFilters.BmpFilters.create_look_up(y_mid_hist)
    look_up_table_low = BmpFilters.BmpFilters.create_look_up(y_low_hist)
    look_up_table_high = BmpFilters.BmpFilters.create_look_up(y_high_hist)

    res_orig_hist = BmpFilters.BmpFilters.hist(look_up_table_mid, y_orig)
    res_high_hist = BmpFilters.BmpFilters.hist(look_up_table_high, y_high)
    res_low_hist = BmpFilters.BmpFilters.hist(look_up_table_low, y_low)

    y_mid_hist, x = make_hist(res_orig_hist[..., 0])
    plt.bar(x, y_mid_hist)
    plt.savefig(folder + 'mid2 hist.png')
    plt.clf()

    y_mid_hist, x = make_hist(res_high_hist[..., 0])
    plt.bar(x, y_mid_hist)
    plt.savefig(folder + 'high2 hist.png')
    plt.clf()

    y_mid_hist, x = make_hist(res_low_hist[..., 0])
    plt.bar(x, y_mid_hist)
    plt.savefig(folder + 'low2 hist.png')
    plt.clf()

    bmp.writeBmp(folder + "org2.bmp", h, res_orig_hist)
    bmp.writeBmp(folder + "low2.bmp", h, res_low_hist)
    bmp.writeBmp(folder + "high2.bmp", h, res_high_hist)


def main():
    white_and_black_generation()


if __name__ == "__main__":
    main()
