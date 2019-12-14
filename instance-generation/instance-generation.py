from sklearn.datasets import fetch_olivetti_faces
from tensorflow import keras
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def sampling(args):
    z_mean, z_log_var = args
    # extracting the shape with keras.backend is the only way to make this work:
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    # random_normal has mean = 0 and std = 1.0
    epsilon = tf.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2 * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis
    )

def interpolate(a, b, n=10):
    return list(map(lambda t: (1-t/10) * a + t/10 * b, range(n)))

faces = fetch_olivetti_faces()
x_train = faces.images[:, :, :, np.newaxis]
input_shape = x_train[0].shape
latent_dim = 32#32
dec_start_dim = input_shape[0] // 8
batch_size = 8
epochs = 2000
seed = 4242

training_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, rotation_range=5, zoom_range=.1, width_shift_range=.1)
training_data = training_datagen.flow(x_train, y=x_train[:, 0, 0, 0], batch_size=batch_size, seed=seed)
# training_data = training_datagen.flow_from_directory("../../celeba")

#---------------------- Basic components ------------------------------
infer = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=input_shape), # 64x64x1
    keras.layers.Conv2D(8, 3, strides=(2, 2), activation='selu', kernel_initializer="lecun_normal"), # 32x32x8
    keras.layers.Conv2D(16, 3, strides=(2, 2), activation='selu', kernel_initializer="lecun_normal"), # 16x16x16
    keras.layers.Conv2D(32, 3, strides=(2, 2), activation='selu', kernel_initializer="lecun_normal"), # 8x8x32
    keras.layers.Flatten()
])

decode = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(latent_dim,)), # 32
    keras.layers.Dense(dec_start_dim * dec_start_dim * 8, activation='selu', kernel_initializer="lecun_normal"),
    keras.layers.Reshape(target_shape=(dec_start_dim, dec_start_dim, 8)), # 8x8x8
    keras.layers.Conv2DTranspose(32, 3, strides=(2, 2), padding='same', activation='selu', kernel_initializer="lecun_normal"), # 16x16x32
    keras.layers.Conv2DTranspose(16, 3, strides=(2, 2), padding='same', activation='selu', kernel_initializer="lecun_normal"), # 32x32x16
    # keras.layers.UpSampling2D(),
    keras.layers.Conv2DTranspose(8, 3, strides=(2, 2), padding='same', activation='selu', kernel_initializer="lecun_normal"), # 64x64x8
    # keras.layers.UpSampling2D(),
    keras.layers.Conv2DTranspose(1, 3, padding='same') # 64x64x1
])

#--------------- Loss function definition ----------------------
def calculate_loss(args):
    predictions, inputs, z, mean, logvar = args
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=predictions, labels=inputs)
    # cross_ent = keras.losses.mean_squared_error(inputs, predictions)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
    logpz = log_normal_pdf(z, 0., 1.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    variational_loss = -tf.reduce_mean(10*logpx_z + logpz - logqz_x)

    return variational_loss


#------------- Model architecture ----------------------------
inputs = keras.layers.Input(shape=input_shape)
inferred = infer(inputs)
mean = keras.layers.Dense(latent_dim)(inferred)
logvar = keras.layers.Dense(latent_dim)(inferred)
sample = keras.layers.Lambda(sampling, output_shape=(latent_dim,))
z = sample([mean, logvar])
autoencoder_dec = decode(z)

mean_input = keras.layers.Input(shape=mean.shape[1:])
logvar_input = keras.layers.Input(shape=logvar.shape[1:])

#---------------------- Models --------------------------------
infer_mean = keras.Model(inputs, mean)
infer_logvar = keras.Model(inputs, logvar)
variational_loss = keras.layers.Lambda(calculate_loss, output_shape=(1,), name="loss")([autoencoder_dec, inputs, z, mean, logvar])
model = keras.models.Model(inputs, variational_loss)
generate = keras.Model([mean_input, logvar_input], tf.sigmoid(decode(sample([mean_input, logvar_input]))))

# decode_sigmoid = tf.sigmoid(dec)


#------------------- Model training ----------------------------
# model.add_loss(variational_loss)
optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer, loss={'loss': lambda y_true, y_pred: y_pred})
hist = model.fit_generator(training_data, steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, verbose=2)

#------------------- Image generation --------------------------
preds = infer_mean.predict(x_train)
preds_lv = infer_logvar.predict(x_train)

originals = (20, 70, 80, 90)
interpolations = np.zeros((4, 10, *preds[0].shape))
interp_lv = np.zeros((4, 10, *preds_lv[0].shape))

mean_matrix = np.zeros((100, *preds[0].shape))
logvar_matrix = np.zeros((100, *preds_lv[0].shape))

for i in range(4):
    interpolations[i] = interpolate(preds[originals[i]], preds[originals[(i+1)%4]])
    interp_lv[i] = interpolate(preds_lv[originals[i]], preds_lv[originals[(i+1)%4]])

mean_matrix[0:10] = interpolations[0]
mean_matrix[np.array(range(9,100,10))] = interpolations[1]
mean_matrix[np.array(range(99,89,-1))] = interpolations[2]
mean_matrix[np.array(range(90,-1,-10))] = interpolations[3]
logvar_matrix[0:10] = interp_lv[0]
logvar_matrix[np.array(range(9,100,10))] = interp_lv[1]
logvar_matrix[np.array(range(99,89,-1))] = interp_lv[2]
logvar_matrix[np.array(range(90,-1,-10))] = interp_lv[3]

for i in range(10, 90, 10):
    # fill each row of the matrix
    mean_matrix[i:(i+10)] = interpolate(mean_matrix[i], mean_matrix[i+9])
    logvar_matrix[i:(i+10)] = interpolate(logvar_matrix[i], logvar_matrix[i+9])


generated = generate.predict([mean_matrix, logvar_matrix])

concatenated = list(map(lambda i: np.concatenate(generated[i:(i+10)], axis=1), range(0, 100, 10)))
concatenated = np.concatenate(concatenated, axis=0)
# concatenated = np.concatenate([x_train[0], concatenated, x_train[20]], axis=1)


keras.preprocessing.image.save_img("generated0.png", concatenated)
