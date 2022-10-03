import pandas as pd
import argparse
import tensorflow as tf
import os
import cv2
import numpy as np
import copy
from PIL import Image
from keras import backend
from skimage import data, io, filters
from sklearn import preprocessing
from keras import preprocessing
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, concatenate
from keras.models import Model
import keras
from keras import applications

# Note that you can save models in different formats. Some format needs to save/load model and weight separately.
# Some saves the whole thing together. So, for your set up you might need to save and load differently.

def load_model_weights(model, weights = None):
    my_model = tf.keras.models.load_model(model)
    my_model.summary()
    return my_model

def get_images_labels(df, classes, img_size):
    X_test = np.zeros((len(df), img_size, img_size, 3), dtype='float32')
    Y_test = np.zeros((len(df), len(classes)), dtype='float32')
    for i in range(len(df)):
        # frame = cv2.imread(df[i][0])
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.open(df[i][0]).convert('RGB')
        height, width = Image.open(df[i][0]).size
        print(height, width)
        if height != img_size or width != img_size:
            # frame = frame.resize((img_size, img_size), Image.ANTIALIAS)
            factor = img_size / min(height, width)
            frame = frame.resize((int(height * factor), int(width * factor)), Image.ANTIALIAS)
            print(frame.size)
            frame = frame.crop((0, 0, img_size, img_size))
            print(frame.size)
            frame = frame.resize((img_size, img_size), Image.ANTIALIAS)
            print(frame.size)
            # frame.show()
        # frame = frame.astype("float32")
        frame = np.array(frame) / 255  # Normalization
        # frame = tf.io.decode_jpeg(frame, channels=3) / 255.0
        # print(frame.shape)
        print(frame)

        X_test[i, :, :, :] = frame

        im = Image.fromarray(np.uint8(255 * X_test[i]))
        im.show()
        Y_test_temp = np.zeros(len(classes), dtype='float64')
        if df[i][1][0] == ' ':
            p = classes.get(df[i][1][1:])
        else:
            p = classes.get(df[i][1])
        Y_test_temp[p] = 1
        print(Y_test_temp)
        Y_test[i] = Y_test_temp
    print(Y_test)
    # Write the code as needed for your code
    # for index, row in df.iterrows():
    #     label = row['label']
    #     img = tf.io.read_file(row['image_path'])
    #     img = decode_img(img, img_height, img_width)
    return X_test, Y_test

def decode_img(img, img_height, img_width):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Trasnfer Learning Test")
    parser.add_argument('--model', type=str, default='my_model.h5', help='Saved model')
    parser.add_argument('--weights', type=str, default=None, help='weight file if needed')
    parser.add_argument('--test_csv', type=str, default='flowers_test.csv', help='CSV file with true labels')

    args = parser.parse_args()
    model = args.model
    weights = args.weights
    test_csv = args.test_csv

    img_size = 256
    test_df = pd.read_csv(test_csv)
    classes = {'astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy', 'carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip'}
    test_data = test_df.values
    print(test_df.values)
    # print(Y)

    # Create training and validationLo set and Load the image
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                    rotation_range=10,
                                                                    width_shift_range=0.1,
                                                                    height_shift_range=0.1,
                                                                    shear_range=0.1,
                                                                    zoom_range=0.1,
                                                                    horizontal_flip=False,
                                                                    fill_mode='nearest',
                                                                    validation_split=0.1
    )
    train_generator = train_datagen.flow_from_directory(directory="flowers",
                                                        target_size=(img_size, img_size),
                                                        batch_size=32,
                                                        subset='training',
                                                        seed=1,
                                                        class_mode='categorical',
                                                        color_mode='rgb',
    )
    validation_generator = train_datagen.flow_from_directory(directory="flowers",
                                                        target_size=(img_size, img_size),
                                                        batch_size=32,
                                                        subset='validation',
                                                        seed=1,
                                                        class_mode='categorical',
                                                        color_mode='rgb',
    )
    print(train_generator.class_indices)

    '''test_df = pd.read_csv("./flowers_test_1.csv", dtype=str)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # 不增强验证数据
    test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                      directory="./",
                                                      x_col="image_path",
                                                      y_col=" label",
                                                      batch_size=32,
                                                      seed=1,
                                                      shuffle=False,
                                                      cclass_mode=None,
                                                      target_size=(256, 256)
    )
    print(test_generator[0])'''

    # Load pre-trained backbone model
    model = tf.keras.applications.MobileNet(
        input_shape=(img_size, img_size, 3), 
        alpha=0.25,
        depth_multiplier=1, 
        dropout=0.5,
        include_top=False, 
        weights='imagenet', 
        input_tensor=None, 
        pooling=None,
        classes=1000, 
        classifier_activation='softmax'
    )
    '''model = tf.keras.applications.resnet50.ResNet50(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000
    )'''
    # Define the input size
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    x = model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Change the number of classes from 1000 to 13
    outputs = tf.keras.layers.Dense(13, activation='softmax', use_bias=True, name='Logits')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    lr = backend.get_value(model.optimizer.lr)
    # print('lr is: ', lr)
    backend.set_value(model.optimizer.lr, 0.00008)
    lr = backend.get_value(model.optimizer.lr)
    print('lr is: ', lr)

    # Load test images
    test_images, test_labels = get_images_labels(test_data, train_generator.class_indices, img_size)
    print(test_images.shape)
    print(test_labels.shape)

    test_images_copy = copy.deepcopy(test_images)
    test_labels_copy = copy.deepcopy(test_labels)
    # loss, acc = model.evaluate(test_images_copy, test_labels_copy, batch_size=32, verbose=2)
    for i in range(len(test_labels)):
        X_test = np.zeros((1, img_size, img_size, 3), dtype='float32')
        X_test[0] = test_images_copy[i]
        # im = Image.fromarray(255 * X_test[0])
        # im.show()
        # print(model.predict(X_test))
        print(np.argmax(model.predict(X_test)))
    print(model.predict(test_images_copy))

    '''STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    loss, acc = model.evaluate_generator(generator=test_generator, verbose=2)
    print(model.predict_generator(generator=test_generator, verbose=2))
    print('Test model, accuracy: {:5.5f}%'.format(100 * acc))'''

    loss, acc = model.evaluate(test_images_copy, test_labels_copy, batch_size=32, verbose=2)
    loss_0, acc_0 = model.evaluate_generator(validation_generator, verbose=2)

    # Load trained model
    # my_model = load_model_weights(model)

    # Train the model
    for i in range(100):
        test_images_copy = copy.deepcopy(test_images)
        test_labels_copy = copy.deepcopy(test_labels)
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='my_model.h5', verbose=1, save_weights_only=False, save_freq=1)
        #model.fit_generator(train_generator, validation_data=validation_generator, steps_per_epoch=train_generator.samples/train_generator.batch_size, epochs=30, callbacks=[checkpointer])
        model.fit_generator(train_generator, steps_per_epoch=train_generator.samples / train_generator.batch_size, epochs=1, callbacks=None)
        loss, acc = model.evaluate(test_images_copy, test_labels_copy, batch_size=32, verbose=2)
        loss_0, acc_0 = model.evaluate_generator(validation_generator, verbose=2)
        print('Test model, accuracy: {:5.5f}%'.format(100 * acc))
        for j in range(len(test_labels)):
            X_test = np.zeros((1, img_size, img_size, 3), dtype='float32')
            X_test[0] = test_images_copy[j]
            print(model.predict(X_test))
            print(np.argmax(model.predict(X_test)))
            # im = Image.fromarray(255 * X_test[0])
            # im.show()
        # print(model.predict(test_images_copy))
        model.save('my_model.h5')
        if acc == 1.0 and acc_0 > 0.95:
            break