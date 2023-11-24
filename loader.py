from glob import glob


def label_data(path):
    cats_images = glob(f"{path}/cats/*.*")
    dogs_images = glob(f"{path}/dogs/*.*")
    cats_labels = [[1.0, 0.0]] * len(cats_images)
    dogs_labels = [[0.0, 1.0]] * len(dogs_images)

    return cats_images + dogs_images, cats_labels + dogs_labels


def load(path_to_data):
    train_path = f"{path_to_data}/train"
    valid_path = f"{path_to_data}/test"
    train_images, train_labels = label_data(train_path)
    valid_images, valid_labels = label_data(valid_path)

    return train_images, train_labels, valid_images, valid_labels


if __name__ == "__main__":
    data_path = "/home/vadym/Downloads/archive"
    train_images1, train_labels1, valid_images1, valid_labels1 = load(data_path)

    print()



