

from skimage.transform import resize
from skimage.draw import ellipse, line, polygon
import tensorflow as tf
import numpy as np
import cv2



@tf.function
def infer(model, img, upscale=True, flip_test=True):
    is_single = len(img.shape) == 3
    if is_single:
        img = img[np.newaxis, ...]
    
    heatmap = model(img, training=False)
    if flip_test:
        flipped = model(img[:, :, ::-1, :], training=False)[:, :, ::-1, :]
        flipped = tf.gather(flipped, axis=-1, indices=[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
        heatmap = 0.5 * (heatmap + flipped)

    heatmap = tf.clip_by_value(heatmap, 0, 255) / 255
    if upscale:
        heatmap = tf.image.resize(heatmap, img.shape[1:3], antialias=True)
    return heatmap[0] if is_single else heatmap


def extract_skeleton(heatmap):
    skeleton = []
    for k in range(0, heatmap.shape[-1]):
        hm = heatmap[:, :, k]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        skeleton.append((y, x, 100 * hm[y, x]))

    return np.array(skeleton)

def draw_skeleton(skeleton, connectivity, image, cutoff=40):
    skeleton = skeleton.astype(np.int)

    for j1, j2 in connectivity:
        j1, j2 = skeleton[j1], skeleton[j2]
        if j1[2] < cutoff:
            continue
        if j2[2] < cutoff:
            continue
        
        yy, xx = line(j1[0]-1, j1[1], j2[0]-1, j2[1])
        image[yy, xx, :] = 0.5
        yy, xx = line(j1[0]+1, j1[1], j2[0]+1, j2[1])
        image[yy, xx, :] = 0.5
        yy, xx = line(j1[0], j1[1]-1, j2[0], j2[1]-1)
        image[yy, xx, :] = 0.5
        yy, xx = line(j1[0], j1[1]+1, j2[0], j2[1]+1)
        image[yy, xx, :] = 0.5

        yy, xx = line(j1[0],   j1[1],   j2[0],   j2[1])
        image[yy, xx, :] = 1

    for (y, x, c) in [kp for kp in skeleton if kp[2] >= cutoff]:
        yy, xx = ellipse(y, x, 3, 3, shape=image.shape, rotation=0)
        image[yy, xx, :] = 1

    return image


def sample(X, y, yhat, connectivity, prune=False):
    X, y, yhat = np.array(X[0:32]), np.array(y[0:32]), np.array(yhat[0:32])
    for j, (_X, _t, _y), in enumerate(zip(X, y, yhat)):
        _X = np.array(_X) / 255
        _t = np.array(_t) / 255
        _y = np.array(_y) if j > 7 else _t
        _v = np.sum(_t, axis=(0, 1), keepdims=True) > 4
        _y = _y * _v

        _y = resize(_y, (_X.shape[0], _X.shape[1]))
        _X = draw_skeleton(extract_skeleton(_y), connectivity, _X)
        _X[:, 0, :] = 0
        _X[:, -1, :] = 0
        _X[0, :, :] = 0
        _X[-1, :, :] = 0
        X[j] = _X
    
    sample = np.vstack([np.hstack(X[i:i+8]) for i in range(0, 32, 8)])
    sample[212:236, :, :] = 0
    return sample


def evaluate(X, y, yhat):
    metrics = []
    for j, (_X, _t, _y), in enumerate(zip(X, y, yhat)):
        _t = np.array(_t)
        _y = np.array(_y) * 255
        _v = np.sum(_t, axis=(0, 1), keepdims=True) > 4
        _t = _t * _v
        _y = _y * _v

        if np.sum(_v) > 0:
            sy = extract_skeleton(_t)[_v[0, 0]][:, 0:2]
            syhat = extract_skeleton(_y)[_v[0, 0]][:, 0:2]
            joints_mse = np.mean(np.square(sy - syhat))
            heatmap_mse = np.mean(np.square(_t - _y))
            metrics.append((joints_mse, heatmap_mse))
    return np.mean(np.array(metrics), axis=0)

def evaluate_all(dataset, model, flip_test=True, count=None):
    metrics = []
    for j, (_X, _y), in enumerate(dataset):
        _yhat = infer(model, _X, upscale=False, flip_test=flip_test)
        metrics.append(evaluate(_X, _y, _yhat))
        if count != None and j >= count:
            break
    return np.mean(np.array(metrics), axis=0)
