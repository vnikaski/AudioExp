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


def get_random_slice(img, t_slice, f_slice):
    t_begin = np.random.randint(0, img.shape[2]-t_slice)
    f_begin = np.random.randint(0, img.shape[1]-f_slice)
    imsize = img.shape
    if len(imsize) != 4:
        raise ValueError('Expected image in format (batch, freq, time, chan)')
    fragment = img[:,f_begin:f_begin+f_slice, t_begin:t_begin+t_slice]
    multi = np.zeros(imsize)
    for i in range(imsize[1]//f_slice):
        for j in range(imsize[2]//t_slice):
            multi[:, i*f_slice:(i+1)*f_slice, j*t_slice:(j+1)*t_slice] = fragment
    f_remain = imsize[1]%f_slice
    t_remain = imsize[2]%t_slice
    for i in range(imsize[1]//f_slice):
        multi[:, i*f_slice:(i+1)*f_slice, (imsize[2]//t_slice)*t_slice:] = fragment[:,:, :t_remain]
    for j in range(imsize[2]//t_slice):
        multi[:, (imsize[1]//f_slice)*f_slice:, j*t_slice:(j+1)*t_slice] = fragment[:,:f_remain, :]
    multi[:, (imsize[1]//f_slice)*f_slice:, (imsize[2]//t_slice)*t_slice:] = fragment[:,:f_remain, :t_remain]

    padded = np.zeros(imsize)
    padded[:, f_begin: f_begin+fragment.shape[1], t_begin: t_begin+fragment.shape[2]] = fragment
    return multi, padded
