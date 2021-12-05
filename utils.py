import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):
    folders = os.listdir(path_to_data)

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)
        imgs_paths = glob.glob(os.path.join(full_path, '*.png'))
        x_train, x_val = train_test_split(imgs_paths, test_size=split_size)

        for x in x_train:

            path_to_folder = os.path.join(path_to_save_train, folder)
            if not os.path.isdir(path_to_folder):
                os.mkdir(path_to_folder)
            shutil.copy(x, path_to_folder)

        for x in x_val:

            path_to_folder = os.path.join(path_to_save_val, folder)
            if not os.path.isdir(path_to_folder):
                os.mkdir(path_to_folder)
            shutil.copy(x, path_to_folder)


def order_test_set(path_to_imgs, path_to_csv):
    try:
        with open(path_to_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):

                if i == 0:
                    continue

                img_name = row[-1].replace('Test/', '')
                label = row[-2]

                path_to_folder = os.path.join(path_to_imgs, label)
                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_imgs, img_name)
                shutil.move(img_full_path, path_to_folder)

    except:
        print('[INFO] : Error reading the csv file')


def create_generators(batch_size, train_data_path, val_data_path, test_data_path):
    perprocessor = ImageDataGenerator(
        rescale=1/255.
    )

    train_generator = perprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(60, 60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = perprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(60, 60),
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )

    test_generator = perprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(60, 60),
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator
