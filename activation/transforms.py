import numpy as np
import tensorflow as tf

# transforms

def padded_jitter(img, d_t, d_f):
    if d_t//2 != d_t/2:
        t_corr = 1
    else:
        t_corr = 0

    if d_f//2 != d_f/2:
        f_corr = 1
    else:
        f_corr = 0

    crop_shape = (img.shape[0],img.shape[1]-d_f, img.shape[2]-d_t, img.shape[3])
    cropped = tf.image.random_crop(img, crop_shape)
    paddings = ([0,0], [d_f//2, d_f//2+f_corr], [d_t//2,d_t//2+t_corr], [0,0])
    padded = tf.pad(cropped, paddings, "SYMMETRIC")
    return padded

