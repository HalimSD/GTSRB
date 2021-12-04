from utils import split_data, order_test_set


if __name__ == "__main__":
    if False:
        path_to_data = "/Users/halim/PersonalProjects/GTSRB/archive/Train"
        path_to_save_train = "/Users/halim/PersonalProjects/GTSRB/archive/training_data/train"
        path_to_save_val = "/Users/halim/PersonalProjects/GTSRB/archive/training_data/val"
        split_data(path_to_data, path_to_save_train=path_to_save_train,
                   path_to_save_val=path_to_save_val)

    path_to_imgs = "/Users/halim/PersonalProjects/GTSRB/archive/Test"
    pathe_to_csv = "/Users/halim/PersonalProjects/GTSRB/archive/Test.csv"
    order_test_set(path_to_imgs, pathe_to_csv)
