from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
import scipy.spatial as sp


gb_palette = [[15, 56, 15], [48, 98, 48], [ 139, 172, 15], [155, 188, 15]]
c64_palette = [[0, 0, 0	], [255, 255, 255], [136, 0, 0], [170, 255, 238], [204, 68, 204], [0, 204, 85], [0, 0, 170],
               [238, 238, 119], [221, 136, 85], [102, 68, 0], [255, 119, 119], [51, 51, 51], [119, 119, 119], [170, 255, 102],
               [0, 136, 255], [187, 187, 187]]


def get_nearest_web_safe_color(rgbcolor):

    r = int(round((rgbcolor[0] / 255.0) * 5) * 51)
    g = int(round((rgbcolor[1] / 255.0) * 5) * 51)
    b = int(round((rgbcolor[2] / 255.0) * 5) * 51)

    if r > 255:
        r = 255
    if g > 255:

        g = 255
    if b > 255:
        b = 255

    return r, g, b


def get_nearest_color_from_palette(rgbcolor, palette):

    tree = sp.KDTree(palette)  # creating k-d tree from web-save colors
    distance, result = tree.query(rgbcolor)  # get Euclidean distance and index of web-save color in tree/list

    return palette[result]


def ordered_dithering_color(imgarray, palette="WEB"):

    bayerMatrix = np.array([[0, 8, 2, 10],
                            [12, 4, 14, 6],
                            [3, 11, 1, 9],
                            [15, 7, 13, 5]])
    matrix = bayerMatrix / 16

    threshold = np.array([255 / 4, 255 / 4, 255 / 4])

    for i in range(int(np.floor(imgarray.shape[0] / matrix.shape[0])) + 1):
        for j in range(int(np.floor(imgarray.shape[1] / matrix.shape[1])) + 1):
            for ii in range(matrix.shape[0]):
                for jj in range(matrix.shape[1]):
                    if i * matrix.shape[0] + ii < imgarray.shape[0] and j * matrix.shape[1] + jj < imgarray.shape[1]:
                        temp = imgarray[i * matrix.shape[0] + ii, j * matrix.shape[1] + jj, :] + ( np.array([matrix[ii, jj], matrix[ii, jj], matrix[ii, jj]]) * threshold)
                        if palette == "WEB":
                            imgarray[i * matrix.shape[0] + ii, j * matrix.shape[1] + jj, :] = get_nearest_web_safe_color(temp)
                        else:
                            imgarray[i * matrix.shape[0] + ii, j * matrix.shape[1] + jj, :] = get_nearest_color_from_palette(temp, palette)

    return imgarray


def no_dither_color(imgarray, palette="WEB"):

    for y in range(imgarray.shape[0]):
        for x in range(imgarray.shape[1]):
            if palette == "WEB":
                imgarray[y, x, :] = get_nearest_web_safe_color(imgarray[y, x, :])
            else:
                imgarray[y, x, :] = get_nearest_color_from_palette(imgarray[y, x, :], palette)

    return imgarray


def ordered_dithering_bw(imgarray, matrix=None):
    if matrix is not None:
        if matrix == "BAYER":
            bayerMatrix = np.array([[0, 8, 2, 10],
                                     [12, 4, 14, 6],
                                     [3, 11, 1, 9],
                                     [15, 7, 13, 5]])
            tempMatrix = bayerMatrix / 16 * 255
        elif matrix == "HALFTONE":
            cdotMatrix = np.array([[0, 0, 1, 1, 1, 1, 0, 0],
                                   [0, 1, 2, 2, 2, 2, 1, 0],
                                   [1, 2, 2, 3, 3, 2, 2, 1],
                                   [1, 2, 3, 4, 4, 3, 2, 1],
                                   [1, 2, 3, 4, 4, 3, 2, 1],
                                   [1, 2, 2, 3, 3, 2, 2, 1],
                                   [0, 1, 2, 2, 2, 2, 1, 0],
                                   [0, 0, 1, 1, 1, 1, 0, 0]
                                   ])
            tempMatrix = cdotMatrix / 5 * 255
        else:
            tempMatrix = matrix
    else:
        bayerMatrix = np.array([[0, 8, 2, 10],
                                [12, 4, 14, 6],
                                [3, 11, 1, 9],
                                [15, 7, 13, 5]])
        tempMatrix = bayerMatrix / 16 * 255

    matrix = tempMatrix

    for i in range(int(np.floor(imgarray.shape[0] / matrix.shape[0])) + 1):
        for j in range(int(np.floor(imgarray.shape[1] / matrix.shape[1])) + 1):
            for ii in range(matrix.shape[0]):
                for jj in range(matrix.shape[1]):
                    if i * matrix.shape[0] + ii < imgarray.shape[0] and j * matrix.shape[1] + jj < imgarray.shape[1]:
                        if imgarray[i * matrix.shape[0] + ii, j * matrix.shape[1] + jj] > matrix[ii, jj]:
                            imgarray[i * matrix.shape[0] + ii, j * matrix.shape[1] + jj] = 1
                        else:
                            imgarray[i * matrix.shape[0] + ii, j * matrix.shape[1] + jj] = 0

    return imgarray


def threshold_bw(imgarray, thresh=128, random=False):

    for i in range(imgarray.shape[0]):
        for j in range(imgarray.shape[1]):
            if random:
                thresh = randrange(256)
            if imgarray[i, j] > thresh:
                imgarray[i, j] = 1
            else:
                imgarray[i, j] = 0
    return imgarray


def error_diffusion_bw(imgarray, algo="FLOYDSTEINBERG"):

    if algo == "NORM":
        imgarray = np.float32(imgarray.copy())
        imgarray *= 1 / 256
        distribution = np.array([7, 3, 5, 1], dtype=float) / 16
        u = np.array([0, 1, 1, 1])

        v = np.array([1, -1, 0, 1])

        for y in range(imgarray.shape[0] - 1):
            for x in range(imgarray.shape[1] - 1):
                value = np.round(imgarray[y, x])
                error = imgarray[y, x] - value
                imgarray[y, x] = value
                imgarray[y + u, x + v] += error * distribution

        imgarray[:, -1] = 1
        imgarray[-1, :] = 1
        return imgarray

    if algo == "FLOYDSTEINBERG":
        imgarray = np.float32(imgarray.copy())
        imgarray *= 1/256
        distribution = np.array([7, 3, 5, 1], dtype=float) / 16
        u = np.array([0, 1, 1, 1])
        v = np.array([1, -1, 0, 1])

        for y in range(imgarray.shape[0] - 1):
            for x in range(imgarray.shape[1] - 1):
                value = np.round(imgarray[y, x])
                error = imgarray[y, x] - value
                imgarray[y, x] = value
                imgarray[y + u, x + v] += error * distribution

        imgarray[:, -1] = 1
        imgarray[-1, :] = 1
        return imgarray

    elif algo == "JJN":
        imgarray = np.float32(imgarray.copy())
        imgarray *= 1 / 256
        distribution = np.array([7, 5, 3, 5, 7, 5, 3, 1, 3, 5, 3, 1], dtype=float) / 48
        u = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        v = np.array([1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2])

        for y in range(imgarray.shape[0] - 2):
            for x in range(imgarray.shape[1] - 2):
                value = np.round(imgarray[y, x])
                error = imgarray[y, x] - value
                imgarray[y, x] = value
                imgarray[y + u, x + v] += error * distribution

        imgarray[:, -1] = 1
        imgarray[-1, :] = 1
        return imgarray

    else:
        return imgarray
