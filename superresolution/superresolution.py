import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator, save_img, img_to_array

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def conv_autoencoder(input_shape = (256, 256, 3)):
    encoder_input = Input(shape = input_shape)
    encoder = Conv2D(16, 3, padding = "same", activation="relu")(encoder_input)
    encoder = MaxPooling2D(padding = "same")(encoder)
    encoder = Conv2D(64, 3, padding = "same", activation="relu")(encoder)
    encoder = MaxPooling2D(padding = "same")(encoder)
    encoder_model = keras.Model(encoder_input, encoder)

    decoder_input = Input(shape = encoder_model.output_shape[1:])
    decoder = Conv2D(64, 3, padding = "same", activation="relu")(decoder_input)
    decoder = UpSampling2D()(decoder)
    decoder = Conv2D(16, 3, padding = "same", activation="relu")(decoder)
    decoder = UpSampling2D()(decoder)
    decoder = Conv2D(3, 3, padding="same", activation="sigmoid")(decoder)
    decoder_model = keras.Model(decoder_input, decoder)

    full_model = keras.Model(encoder_input, decoder_model(encoder))

    return encoder_model, decoder_model, full_model

## Low resolution autoencoder
lr_encoder, lr_decoder, lr_ae = conv_autoencoder()

lr_gen = ImageDataGenerator(rescale=1./255).flow_from_directory("DIV2K_train_LR_x8/", class_mode = "input")
lr_ae.compile(optimizer = "rmsprop", loss = "binary_crossentropy")
lr_ae.fit_generator(lr_gen, steps_per_epoch=32, epochs = 20)

## High resolution autoencoder
hr_encoder, hr_decoder, hr_ae = conv_autoencoder()

hr_gen = ImageDataGenerator(rescale=1./255).flow_from_directory("DIV2K_train_HR/", class_mode = "input")
hr_ae.compile(optimizer = "rmsprop", loss = "binary_crossentropy")
hr_ae.fit_generator(hr_gen, steps_per_epoch=32, epochs = 20)

def super_generator(batch_size = 32):
    lr_gen = ImageDataGenerator(rescale=1./255).flow_from_directory("DIV2K_train_LR_x8/", class_mode = "input", shuffle = False, batch_size = batch_size)
    hr_gen = ImageDataGenerator(rescale=1./255).flow_from_directory("DIV2K_train_HR/", class_mode = "input", shuffle = False, batch_size = batch_size)
    while True:
        next_x = next(lr_gen)
        next_y = next(hr_gen)

        yield (next_x[0], next_y[0])

super_ae = keras.Model(lr_encoder.input, hr_decoder(lr_encoder.output))
super_ae.compile(optimizer = "rmsprop", loss = "binary_crossentropy")
super_ae.fit_generator(super_generator(), steps_per_epoch = 32, epochs = 20)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory("DIV2K_valid_LR_x8/", class_mode = None, shuffle=False)
test_preds = super_ae.predict(test_gen)
save_img("test0.png", img_to_array(test_gen[0][0]))
save_img("pred0.png", img_to_array(test_preds[0]))
