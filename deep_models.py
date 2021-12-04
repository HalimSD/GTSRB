import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, BatchNormalization, Input, Dense, GlobalAvgPool2D
from tensorflow.keras import Model


def street_sign_model(nm_classes):
    model_input = Input(shape=(60,60,3))
    x = Conv2D(32, (3, 3), activation='relu')(model_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(nm_classes, activation='softmax')(x)

    return Model(inputs=model_input, outputs= x)

# This is only to test the model here right away before implementing it on the project data
if __name__ == '__main__':
    model = street_sign_model(10)
    model.summary()