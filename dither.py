from PIL import Image
import numpy as np
from random import randrange
import scipy.spatial as sp


gb_palette = [[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]]

c64_palette = [[0, 0, 0	], [255, 255, 255], [136, 0, 0], [170, 255, 238], [204, 68, 204], [0, 204, 85], [0, 0, 170],
               [238, 238, 119], [221, 136, 85], [102, 68, 0], [255, 119, 119], [51, 51, 51], [119, 119, 119],
               [170, 255, 102], [0, 136, 255], [187, 187, 187]]

grayscale_2bit = [[0, 0, 0], [64, 64, 64], [128, 128, 128], [192, 192, 192], [255, 255, 255]]

RGB_3bit = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]


def get_palette(photo, colors):

    new_photo = photo.convert("RGB").quantize(colors=colors)
    new_palette = []
    palette = new_photo.convert('RGB').getcolors()

    for each in palette:
        new_palette.append(each[1])

    return new_palette


def get_nearest_web_safe_color(rgb_color):

    """

    :param rgb_color:  rgb color triplet (0-255)
    :return: rgb color triplet (0-255) from web safe color palette nearest to input color
    """

    r = int(round((rgb_color[0] / 255.0) * 5) * 51)
    g = int(round((rgb_color[1] / 255.0) * 5) * 51)
    b = int(round((rgb_color[2] / 255.0) * 5) * 51)

    if r > 255:
        r = 255
    if g > 255:
        g = 255
    if b > 255:
        b = 255

    return r, g, b


def get_nearest_color_from_palette(rgb_color, palette='WEB'):

    """

    :param rgb_color: rgb color triplet (0-255)
    :param palette: color palette, list of rgb color triplets (0-255)
    :return: rgb color triplet (0-255) from color palette nearest to input color
    """
    if palette == "WEB":
        return get_nearest_web_safe_color(rgb_color)
    else:
        tree = sp.KDTree(palette)
        return palette[tree.query(rgb_color)[1]]


def ordered_dithering_color(image_array, palette="WEB", matrix=None):

    """

    :param image_array: np array size: width x height x 3 (0-255)
    :param palette: Color palette, list of rgb color triplets (0-255).
    :param matrix: Ordered dithering matrix. Only BAYER and HALFTONE supported.
    :return: Dithered np array.
    """
    if matrix is not None:
        if matrix == "BAYER":
            bayer_matrix = np.array([[0, 8, 2, 10],
                                     [12, 4, 14, 6],
                                     [3, 11, 1, 9],
                                     [15, 7, 13, 5]])
            temp_matrix = bayer_matrix / 16
        elif matrix == "HALFTONE":
            halftone_matrix = np.array([[0, 0, 1, 1, 1, 1, 0, 0],
                                        [0, 1, 2, 2, 2, 2, 1, 0],
                                        [1, 2, 2, 3, 3, 2, 2, 1],
                                        [1, 2, 3, 4, 4, 3, 2, 1],
                                        [1, 2, 3, 4, 4, 3, 2, 1],
                                        [1, 2, 2, 3, 3, 2, 2, 1],
                                        [0, 1, 2, 2, 2, 2, 1, 0],
                                        [0, 0, 1, 1, 1, 1, 0, 0]
                                        ])
            temp_matrix = halftone_matrix / 5
        elif type(matrix) == 'string':
            return -1
        else:
            temp_matrix = matrix
    else:
        bayer_matrix = np.array([[0, 8, 2, 10],
                                 [12, 4, 14, 6],
                                 [3, 11, 1, 9],
                                 [15, 7, 13, 5]])
        temp_matrix = bayer_matrix / 16

    matrix = temp_matrix

    threshold = np.array([255 / 4, 255 / 4, 255 / 4])

    for i in range(int(np.floor(image_array.shape[0] / matrix.shape[0])) + 1):
        for j in range(int(np.floor(image_array.shape[1] / matrix.shape[1])) + 1):
            for ii in range(matrix.shape[0]):
                for jj in range(matrix.shape[1]):
                    if i * matrix.shape[0] + ii < image_array.shape[0] \
                            and j * matrix.shape[1] + jj < image_array.shape[1]:
                        temp = image_array[i * matrix.shape[0] + ii, j * matrix.shape[1] + jj, :] + \
                            (np.array([matrix[ii, jj], matrix[ii, jj], matrix[ii, jj]]) * threshold)
                        image_array[i * matrix.shape[0] + ii, j * matrix.shape[1] + jj, :] = \
                            get_nearest_color_from_palette(temp, palette)

    return image_array


def no_dither_color(image, palette="WEB"):

    image_array = np.array(image)

    for y in range(image_array.shape[0]):
        for x in range(image_array.shape[1]):
            image_array[y, x, :] = get_nearest_color_from_palette(image_array[y, x, :], palette)

    return Image.fromarray(np.uint8(image_array)).convert("RGB")


def undither(image):

    return None


def ordered_dithering_bw(image_array, matrix=None):
    if matrix is not None:
        if matrix == "BAYER":
            bayer_matrix = np.array([[0, 8, 2, 10],
                                     [12, 4, 14, 6],
                                     [3, 11, 1, 9],
                                     [15, 7, 13, 5]])
            temp_matrix = bayer_matrix / 16 * 255
        elif matrix == "HALFTONE":
            halftone_matrix = np.array([[0, 0, 1, 1, 1, 1, 0, 0],
                                       [0, 1, 2, 2, 2, 2, 1, 0],
                                       [1, 2, 2, 3, 3, 2, 2, 1],
                                       [1, 2, 3, 4, 4, 3, 2, 1],
                                       [1, 2, 3, 4, 4, 3, 2, 1],
                                       [1, 2, 2, 3, 3, 2, 2, 1],
                                       [0, 1, 2, 2, 2, 2, 1, 0],
                                       [0, 0, 1, 1, 1, 1, 0, 0]
                                        ])
            temp_matrix = halftone_matrix / 5 * 255
        elif type(matrix) == 'string':
            return -1
        else:
            temp_matrix = matrix
    else:
        bayer_matrix = np.array([[0, 8, 2, 10],
                                [12, 4, 14, 6],
                                [3, 11, 1, 9],
                                [15, 7, 13, 5]])
        temp_matrix = bayer_matrix / 16 * 255

    matrix = temp_matrix

    for i in range(int(np.floor(image_array.shape[0] / matrix.shape[0])) + 1):
        for j in range(int(np.floor(image_array.shape[1] / matrix.shape[1])) + 1):
            for ii in range(matrix.shape[0]):
                for jj in range(matrix.shape[1]):
                    if i * matrix.shape[0] + ii < image_array.shape[0] \
                            and j * matrix.shape[1] + jj < image_array.shape[1]:
                        if image_array[i * matrix.shape[0] + ii, j * matrix.shape[1] + jj] > matrix[ii, jj]:
                            image_array[i * matrix.shape[0] + ii, j * matrix.shape[1] + jj] = 1
                        else:
                            image_array[i * matrix.shape[0] + ii, j * matrix.shape[1] + jj] = 0

    return image_array


def threshold_bw(image_array, thresh=128, random=False):

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            if random:
                thresh = randrange(256)
            if image_array[i, j] > thresh:
                image_array[i, j] = 1
            else:
                image_array[i, j] = 0
    return image_array


def error_diffusion_bw(image_array, algo="FLOYDSTEINBERG"):

    if algo == "FLOYDSTEINBERG":
        image_array = np.float32(image_array.copy())
        image_array *= 1/256
        distribution = np.array([7, 3, 5, 1], dtype=float) / 16
        u = np.array([0, 1, 1, 1])
        v = np.array([1, -1, 0, 1])

        for y in range(image_array.shape[0] - 1):
            for x in range(image_array.shape[1] - 1):
                value = np.round(image_array[y, x])
                error = image_array[y, x] - value
                image_array[y, x] = value
                image_array[y + u, x + v] += error * distribution

        image_array[:, -1] = 1
        image_array[-1, :] = 1
        return image_array

    elif algo == "JJN":
        image_array = np.float32(image_array.copy())
        image_array *= 1 / 256
        distribution = np.array([7, 5, 3, 5, 7, 5, 3, 1, 3, 5, 3, 1], dtype=float) / 48
        u = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        v = np.array([1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2])

        for y in range(image_array.shape[0] - 2):
            for x in range(image_array.shape[1] - 2):
                value = np.round(image_array[y, x])
                error = image_array[y, x] - value
                image_array[y, x] = value
                image_array[y + u, x + v] += error * distribution

        image_array[:, -1] = 1
        image_array[-1, :] = 1
        return image_array

    else:
        return image_array


def error_diffusion_color(image_array, algo="FLOYDSTEINBERG", palette="WEB"):

    if algo == "FLOYDSTEINBERG":
        image_array = np.float32(image_array.copy())
        distribution = np.array([7, 3, 5, 1], dtype=float) / 16
        u = np.array([0, 1, 1, 1])
        v = np.array([1, -1, 0, 1])

        for y in range(image_array.shape[0] - 1):
            for x in range(image_array.shape[1] - 1):
                value = get_nearest_color_from_palette(image_array[y, x], palette)
                error = image_array[y, x] - value
                image_array[y, x] = value
                image_array[y + u, x + v, :] += error * distribution.reshape(4, 1)

        image_array[:, -1] = 1
        image_array[-1, :] = 1
        return image_array

    elif algo == "JJN":
        image_array = np.float32(image_array.copy())
        distribution = np.array([7, 5, 3, 5, 7, 5, 3, 1, 3, 5, 3, 1], dtype=float) / 48
        u = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        v = np.array([1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2])

        for y in range(image_array.shape[0] - 2):
            for x in range(image_array.shape[1] - 2):
                value = get_nearest_color_from_palette(image_array[y, x], palette)
                error = image_array[y, x] - value
                image_array[y, x] = value
                image_array[y + u, x + v] += error * distribution.reshape(12, 1)

        image_array[:, -1] = 1
        image_array[-1, :] = 1
        return image_array

    else:
        return image_array


def ordered_dithering(image, palette="WEB", matrix=None, color='RGB', colors=8):

    if palette == "ADAPTIVE":
        palette = get_palette(image, colors)

    if color == 'BW':
        image = image.convert("L")

    img_array = np.array(image)

    print(palette)
    if len(img_array.shape) == 3:
        img_array = ordered_dithering_color(img_array, palette=palette, matrix=matrix)
    else:
        img_array = ordered_dithering_bw(img_array, matrix=matrix)

    return Image.fromarray(np.uint8(img_array*255))


def error_diffusion(image, algo="FLOYDSTEINBERG", color="RGB", palette="WEB", colors=8):

    img_array = np.array(image)

    if palette == "ADAPTIVE":
        palette = get_palette(image, colors)

    if len(img_array.shape) == 3 and color == "BW":
        image = image.convert("L")
        img_array = np.array(image)
        img_array = error_diffusion_bw(img_array, algo=algo)
    elif len(img_array.shape) == 2:
        img_array = error_diffusion_bw(img_array, algo=algo)
    elif len(img_array.shape) == 3 and color == "RGB":
        img_array = error_diffusion_color(img_array, palette=palette, algo=algo)

    return Image.fromarray(np.uint8(img_array)).convert("RGB")


def open_image(filename, size=None, conversion="RGB"):

    photo = Image.open(filename)

    if size is None:
        size = photo.size
    else:
        size = [size, size]

    photo.thumbnail(size, Image.ANTIALIAS)

    if conversion == "RGB":
        photo = photo.convert("RGB")
    elif conversion == "BW":
        photo = photo.convert("L")
    elif type(conversion) == int:
        photo = photo.convert("RGB").quantize(colors=conversion)
        photo = photo.convert("RGB")

    else:
        photo = photo.convert("RGB")

    return photo


def upscale(image, scale):

    new_image = image.resize(np.array(image.size)*scale)

    return new_image
