import numpy
import PIL

from mat73 import loadmat

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import np_utils
from keras.callbacks import TensorBoard, ModelCheckpoint

import utils

# Устанавливаем seed для повторяемости результатов
numpy.random.seed(42)

class Neural:
    conf_filename = 'base_neural.h5'
    data_path = 'data/'

    def prepare(self):
        try:
            self.model = load_model(self.conf_filename)
        except OSError:
            self.model = self.train_new_model()
            self.model.save(self.conf_filename)

    def train_new_model(self):
        return Sequential()

class SignsNeural(Neural):
    conf_filename = f'signs_neural.h5'

    def __init__(self):# Создаем последовательную модель
        self.sign_list = list(range(10)) + [
            '<', '>', '='
        ]

    def train_new_model(self):
        model = Sequential()
        # Размер изображения
        img_rows, img_cols = 28, 28

        # Загружаем данные
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Преобразование размерности изображений
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        # Нормализация данных
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        # Преобразуем метки в категории
        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)

        # Преобразуем метки в категории
        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)

        # Добавляем уровни сети
        model.add(Conv2D(75, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(100, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='sigmoid'))

        # Компилируем модель
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        print(model.summary())

        # Обучаем сеть
        model.fit(
            X_train, Y_train,
            batch_size=500,
            epochs=2,
            validation_split=0.2,
            verbose=2,
        )

        # Оцениваем качество обучения сети на тестовых данных
        scores = model.evaluate(X_test, Y_test, verbose=0)
        print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
        print("Max %.2f%% " % (max(scores)*100))

        return model

    def get_sign(self, image: PIL.Image):
        image = image.resize((28, 28), PIL.Image.Resampling.BILINEAR)
        image_data = numpy.asarray(image)
        gray_image_data = utils.gray_scale(image_data).reshape(1, 28, 28, 1)

        predicts = self.model(gray_image_data)
        return predicts


class SeekNeural(Neural):
    """
    Works with dataset - http://ufldl.stanford.edu/housenumbers/
    """
    conf_filename = 'seek_neural.h5'
    standart_width = 500
    standart_height = 500

    def __init__(self):
        self.data_path += 'street_numbers/'

    def _get_dataset_from_file(self, foldername: str):
        full_path = self.data_path + foldername
        mat_data = loadmat(full_path + 'digitStruct.mat')

        images = mat_data['digitStruct']['name']
        boxes = mat_data['digitStruct']['bbox']
        length = len(images)
        X = numpy.zeros((length, self.standart_width, self.standart_height, 1), 'uint')
        y = numpy.zeros((length, self.standart_width, self.standart_height, 1), 'float')
        for index, (img_name, box) in enumerate(zip(images, boxes)):
            image = PIL.Image.open(full_path + img_name)

        return X, y

    def get_full_dataset(self):
        return (
            self._get_dataset_from_file('train/'),
            self._get_dataset_from_file('test/')
        )

    def train_new_model(self):
        return
        model = Sequential()
        # Размер изображения
        img_rows, img_cols = 28, 28

        # Загружаем данные
        (X_train, y_train), (X_test, y_test) = self.get_full_dataset()

        # Преобразование размерности изображений
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        # Нормализация данных
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        # Преобразуем метки в категории
        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)

        # Преобразуем метки в категории
        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)

        # Добавляем уровни сети
        model.add(Conv2D(75, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(100, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        # Компилируем модель
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        print(model.summary())

        # Обучаем сеть
        model.fit(
            X_train, Y_train,
            batch_size=500,
            epochs=2,
            validation_split=0.2,
            verbose=2,
        )

        # Оцениваем качество обучения сети на тестовых данных
        scores = model.evaluate(X_test, Y_test, verbose=0)
        print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
        print("Max %.2f%% " % (max(scores)*100))

        return model

    def get_crop_data(self, image: PIL.Image) -> tuple[int, int, int, int]:
        return
        image_data = numpy.asarray(image)
        gray_image_data = utils.gray_scale(image_data).reshape(1, 28, 28, 1)

        predicts = self.model(gray_image_data)
        print(predicts)
        return self.sign_list[numpy.argmax(predicts)]
