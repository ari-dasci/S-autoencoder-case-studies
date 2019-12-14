import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def rgb_to_ycbcr(im):
    xform = np.array(
        [[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr


def ycbcr_to_rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def preprocessing(dat):
    return rgb_to_ycbcr(dat) / 255.


def conv_ae(input_shape=(32, 32, 3)):
    params = {"padding": "same", "activation": keras.layers.PReLU}

    enc_i = keras.layers.Input(shape=input_shape)
    enc = keras.layers.Conv2D(6, 3, **params)(enc_i)
    enc = keras.layers.Conv2D(6, 3, strides=2, **params)(enc)
    enc = keras.layers.Conv2D(6, 3, **params)(enc)
    enc = keras.layers.Conv2D(6, 3, strides=2, **params)(enc)
    enc = keras.layers.Conv2D(6, 3, **params)(enc)
    enc = keras.layers.Conv2D(6, 3, strides=2, **params)(enc)

    encoder = keras.Model(enc_i, enc)

    dec_i = keras.layers.Input(shape=enc.shape[1:])
    dec = keras.layers.Conv2DTranspose(6, 3, **params)(dec_i)
    dec = keras.layers.Conv2DTranspose(6, 4, strides=2, **params)(dec)
    dec = keras.layers.Conv2DTranspose(6, 3, **params)(dec)
    dec = keras.layers.Conv2DTranspose(6, 4, strides=2, **params)(dec)
    dec = keras.layers.Conv2DTranspose(6, 3, **params)(dec)
    dec = keras.layers.Conv2DTranspose(3, 4, strides=2, **params)(dec)

    decoder = keras.Model(dec_i, dec)

    nsy_i = keras.layers.Input(shape=enc.shape[1:])
    nsy = keras.layers.GaussianNoise(0.05)(nsy_i)

    noiser = keras.Model(nsy_i, nsy)

    autoencoder = keras.Model(enc_i, decoder(noiser(encoder.output)))

    return encoder, decoder, autoencoder


def dup_gen(single_gen):
    while True:
        next_x = next(single_gen)

        yield (next_x, next_x)

# Experiment with CIFAR10 dataset


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=rgb_to_ycbcr, rescale=1./255).flow(x_train)
gen = dup_gen(gen)

encoder, decoder, autoencoder = conv_ae()
autoencoder.compile(loss="binary_crossentropy", optimizer="rmsprop")
autoencoder.fit_generator(gen, steps_per_epoch=32, epochs=50)

testgen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=rgb_to_ycbcr, rescale=1./255).flow(x_test)
preds = autoencoder.predict_generator(testgen, steps=32)

for i in range(preds.shape[0]):
    keras.preprocessing.image.save_img(
        f"pred{i}.png", ycbcr_to_rgb(preds[i, :, :, :] * 255))

# Experiment with DIV2K dataset:

resize = (256, 256, 3)

gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=rgb_to_ycbcr, rescale=1./255).flow_from_directory("DIV2K_train_HR/", target_size=resize[:2], class_mode="input")

encoder, decoder, autoencoder = conv_ae(input_shape=resize)
autoencoder.compile(loss="binary_crossentropy", optimizer="rmsprop")
autoencoder.fit_generator(gen, steps_per_epoch=32, epochs=50)

testgen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=rgb_to_ycbcr, rescale=1./255).flow_from_directory("DIV2K_valid_HR/", target_size=resize[:2], class_mode=None)
preds = autoencoder.predict_generator(testgen, steps=32)

for i in range(preds.shape[0]):
    keras.preprocessing.image.save_img(
        f"pred{i}.png", ycbcr_to_rgb(preds[i, :, :, :] * 255))
