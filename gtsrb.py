import tensorflow as tf
from deep_models import street_sign_model
from utils import split_data, order_test_set, create_generators
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


if __name__ == "__main__":
    
    if False:
        path_to_data = "./archive/Train"
        path_to_save_train = "./archive/training_data/train"
        path_to_save_val = "./archive/training_data/val"
        split_data(path_to_data, path_to_save_train=path_to_save_train,
                   path_to_save_val=path_to_save_val)

    if False:
        path_to_imgs = "./archive/Test"
        pathe_to_csv = "./archive/Test.csv"
        order_test_set(path_to_imgs, pathe_to_csv)

    path_to_train = "./archive/training_data/train"
    path_to_val = "./archive/training_data/val"
    path_to_test = "./archive/Test"
    batch_size = 64
    epochs = 15

    train_generator, val_generator, test_generator = create_generators(
        batch_size, path_to_train, path_to_val, path_to_test)
    nmb_calsses = train_generator.num_classes

    TRAIN = False
    TEST = True

    if TRAIN:

        path_to_save_model = './ModelsData'
        chpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=10,

        )

        model = street_sign_model(nmb_calsses)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_generator, epochs=epochs, batch_size=batch_size,
                  validation_data=val_generator, callbacks=[chpt_saver, early_stop])

    if TEST:

        model = tf.keras.models.load_model("./ModelsData")
        model.summary()

        print("evaluating the model on the validation set: ")
        model.evaluate(val_generator)
        print("evaluating the model on the test set: ")
        model.evaluate(test_generator)
