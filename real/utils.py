"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import numpy as np
import scipy as sp
from imageio import imread, imsave
import cv2
from time import gmtime, strftime
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def save_images(images, output_dir, image_name, nrow=8, padding=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, image_name)
    ndarr = make_grid(images, nrow=nrow, padding=padding)
    imsave(filename, ndarr)


def make_grid(tensor, nrow=8, padding=1):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    batch_size = tensor.shape[0]
    xmaps = min(nrow, batch_size)
    ymaps = batch_size // xmaps
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + padding, width * xmaps + padding, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= batch_size:
                break
            h, h_width = y * height + padding, height - padding
            w, w_width = x * width + padding, width - padding
            grid[h:h + h_width, w:w + w_width] = tensor[k]
            k += 1
    return grid


def save_kde_plot(z, output_dir, image_name, bbox=[-5, 5, -5, 5]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x = z[:, 0]
    y = z[:, 1]
    values = np.vstack([x, y])
    kernel = sp.stats.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis(bbox)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:100j, bbox[2]:bbox[3]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    cset = ax.contour(xx, yy, f, colors='k')
    # ax.plot(z[:, 0], z[:, 1], 'x')

    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
    plt.close(fig)

def save_heat_map(f, output_dir, image_name, samples=None, bbox=[-5, 5, -5, 5]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    N, M = f.shape
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis(bbox)

    # ax.imshow(f, cmap='hot',
    #     origin='lower',
    #     extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
    xx, yy = np.mgrid[bbox[0]:bbox[1]:(N*1j), bbox[2]:bbox[3]:(M*1j)]
    cfset = ax.contourf(xx, yy, f, cmap='Reds')
    cset = ax.contour(xx, yy, f, colors='k')

    if samples is not None:
        ax.plot(samples[:, 0], samples[:, 1], 'x')

    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
    plt.close(fig)


def save_z_plot(z, zlabels, output_dir, image_name, bbox=[-5, 5, -5, 5]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, ax = plt.subplots()

    ax.set_autoscale_on(False)
    for label in set(zlabels):
        z_i = z[zlabels==label]
        ax.plot(z_i[:, 0], z_i[:, 1], 'x')

    ax.set_aspect("equal")
    ax.axis(bbox)

    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
    plt.close(fig)


def imread(path, is_grayscale = False):
    if (is_grayscale):
        return imread(path, flatten=True).astype(np.float)
    else:
        return imread(path).astype(np.float)


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return cv2.resize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    # return np.array(cropped_image)/127.5 - 1.
    return  np.array(cropped_image)/255.


def to_nested_dict(d):
    nested_d = defaultdict(dict)
    for (k1, k2), v in d.items():
        nested_d[k1][k2] = v
    return nested_d

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict
