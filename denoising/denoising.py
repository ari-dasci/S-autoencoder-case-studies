from tensorflow import keras
import numpy as np
from scipy.stats import entropy
import tensorflow as tf
from tensorflow.keras.layers import GaussianNoise, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Lambda
import os

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
keras.backend.set_session(tf.Session(config=config))


def noisy(sd):
    def f(im):
        return np.minimum(im/255., np.float32(np.greater(np.random.uniform(size=im.shape), sd)))
        # return np.clip(im/255. + np.random.normal(size = im.shape, scale=sd), 0, 1)
    return f

def noisy_tf(sd):
    def f(im):
        return tf.minimum(im, tf.dtypes.cast(tf.greater(tf.random.uniform(im.shape[1:]), sd), tf.float32))
        # return tf.clip_by_value(im + tf.random.normal(shape=im.shape, stddev=sd), 0, 1)
    return f

np.random.seed(12345)
side = 96
images = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
noisy_images = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=noisy(.1))

generator = images.flow_from_directory("../stl10/", class_mode="input", target_size=(side, side))
noisy_generator = noisy_images.flow_from_directory("../stl10_test/", class_mode=None, save_to_dir="noisy", target_size=(side, side))

inputs = keras.Input(shape = (side, side, 3))

noise_layer = Lambda(noisy_tf(.1), input_shape=(side, side, 3), output_shape=(side, side, 3))

encoder = keras.Sequential([
    Conv2D(filters=64, kernel_size=5,
                activation="relu", padding="same", input_shape=(side, side, 3)),
    Conv2D(filters=128, kernel_size=1,
                activation="relu", padding="same"),
    MaxPooling2D()
])

decoder = keras.Sequential([
    Conv2DTranspose(filters=64, kernel_size=5, activation="relu",
                            padding="same", input_shape=encoder.get_output_shape_at(0)[1:]),
    UpSampling2D(),
    Conv2DTranspose(filters=3, kernel_size=3,
                            activation="sigmoid", padding="same")
])

model = keras.Model(inputs, decoder(encoder(noise_layer(inputs))))
predictor = keras.Model(inputs, decoder(encoder(inputs)))

model.compile(loss="binary_crossentropy", optimizer="adam")
# predictor.compile(loss="binary_crossentropy", optimizer="adam")
model.fit_generator(generator, 128, 10)

import itertools

os.makedirs("noisy", exist_ok=True)
test_batch = next(noisy_generator)
predictions = predictor.predict(test_batch)
os.makedirs("predictions", exist_ok=True)

concat = np.concatenate([test_batch, predictions], axis = 1)
concat = np.concatenate(concat, axis = 1)

keras.preprocessing.image.save_img(f"predictions.png", concat)

import glob
noise = noisy(.1)
imgfiles = glob.glob("../stl10_test/*/*.png")

mse_noisy = np.zeros((len(imgfiles),))
mse_rec = np.zeros((len(imgfiles),))

for i, imgname in enumerate(imgfiles):
    img = np.array(keras.preprocessing.image.load_img(imgname))
    noisy_img = noise(img)
    reconstruction = predictor.predict(noisy_img[np.newaxis,:,:,:])

    mse_noisy[i] = np.mean(np.square(img - noisy_img*255.))
    mse_rec[i] = np.mean(np.square(img - reconstruction*255.))

reduction = (1 - mse_rec/mse_noisy) * 100
print(f"MSE noisy: {np.mean(mse_noisy)} +- {np.std(mse_noisy)}")
print(f"MSE rec: {np.mean(mse_rec)} +- {np.std(mse_rec)}")
print(f"Noise reduction: {np.mean(reduction)} +- {np.std(reduction)}")
