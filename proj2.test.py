import pandas as pd
import argparse
import tensorflow as tf
import os
import cv2
import numpy as np
import copy
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
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

def load_model_weights(model, weights=None):
    my_model = tf.keras.models.load_model(model)
    my_model.summary()
    return my_model


def get_images_labels(df, classes, img_size):
    X_test = np.zeros((len(df), img_size, img_size, 3), dtype='float64')
    Y_test = np.zeros((len(df), len(classes)), dtype='float64')
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
            frame.show()
        # frame = frame.astype("float64")
        frame = np.array(frame) / 255
        # frame = tf.io.decode_jpeg(frame, channels=3) / 255.0
        # print(frame.shape)
        print(frame)

        X_test[i, :, :, :] = frame

        # im = Image.fromarray(np.uint8(255 * X_test[i]))
        # im.show()
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


if __name__ == "__main__":
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
    classes = {'astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy', 'carnation',
               'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip'}
    test_data = test_df.values
    print(test_df.values)
    # print(Y)

    # Load the image
    '''train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
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

    print(train_generator.class_indices)'''


    # Load test images
    classes_dict = {'astilbe': 0, 'bellflower': 1, 'black-eyed susan': 2, 'calendula': 3, 'california poppy': 4, 'carnation': 5, 'common daisy': 6, 'coreopsis': 7, 'dandelion': 8, 'iris': 9, 'rose': 10, 'sunflower': 11, 'tulip': 12}

    test_images, test_labels = get_images_labels(test_data, classes_dict, img_size)
    print(test_images.shape)
    print(test_labels.shape)

    # Load trained model
    my_model = load_model(model)

    test_images_copy = copy.deepcopy(test_images)
    test_labels_copy = copy.deepcopy(test_labels)
    loss, acc = my_model.evaluate(test_images_copy, test_labels_copy, batch_size=32, verbose=2)
    for i in range(len(test_labels)):
        X_test = np.zeros((1, img_size, img_size, 3), dtype='float64')
        X_test[0] = test_images_copy[i]
        # im = Image.fromarray(255 * X_test[0])
        # im.show()
        print(my_model.predict(X_test))
        print(np.argmax(my_model.predict(X_test)))
    results = my_model.predict(test_images_copy)
    print(results)
    for i in range(len(results)):
        print(list(classes_dict.keys())[list(classes_dict.values()).index(np.argmax(results[i]))])
    print('Test model, accuracy: {:5.5f}%'.format(100 * acc))

    # loss_0, acc_0 = my_model.evaluate_generator(validation_generator, verbose=2)
    '''train_acc = [0.8172, 0.9047, 0.9229, 0.9337, 0.9427, 0.9475, 0.9529, 0.9569, 0.9595, 0.9582, 0.9658, 0.9693, 0.9713, 0.9691, 0.9685, 0.9703, 0.9704, 0.9757, 0.9742, 0.9793, 0.9778, 0.9773, 0.9798, 0.9774, 0.9831, 0.9842, 0.9800, 0.9866, 0.9799, 0.9837, 0.9874, 0.9874, 0.9882, 0.9895, 0.9868, 0.9908]
    validation_acc = [0.7592, 0.7467, 0.7709, 0.8593, 0.8272, 0.8444, 0.8780, 0.8303, 0.8772, 0.7850, 0.7435, 0.8264, 0.8874, 0.8303, 0.8796, 0.8475, 0.8890, 0.8296, 0.8772, 0.9062, 0.8866, 0.8608, 0.8874, 0.8944, 0.8835, 0.8874, 0.9030, 0.8772, 0.8921, 0.9077, 0.9124, 0.9124, 0.9163, 0.9171, 0.9195, 0.9226]
    test_acc = [0.75, 0.75, 0.75, 1.00, 0.75, 1.00, 0.75, 0.75, 0.50, 0.50, 0.75, 0.75, 0.75, 1.00, 0.50, 0.50, 0.75, 0.50, 0.75, 0.75, 0.75, 0.75, 0.75, 1.00, 1.00, 0.75, 1.00, 1.00, 1.00, 1.00, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00]
    t = []
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('The training_acc, val_acc and test_acc after each epoch')
    for i in range(36):
        t.append(i + 1)
    plt.plot(t, train_acc, t, validation_acc, t, test_acc)
    plt.show()'''