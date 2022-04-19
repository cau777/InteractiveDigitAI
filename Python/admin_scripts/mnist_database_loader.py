import argparse
import os
from typing import BinaryIO
from codebase.persistence.training_data_pb2 import ClassificationDataSet
from codebase.persistence import save_compressed


def read_int(stream: BinaryIO):
    return int.from_bytes(stream.read(4), "big")


def read_pixel(stream: BinaryIO):
    return int.from_bytes(stream.read(1), "big") / 255 * 0.99


def read_byte(stream: BinaryIO):
    return int.from_bytes(stream.read(1), "big")


def load_images(path: str):
    print(os.getcwd())
    with open(path, "rb") as f:
        magic_number = read_int(f)
        count = read_int(f)
        rows = read_int(f)
        cols = read_int(f)

        if magic_number != 2051:
            raise ValueError("Invalid file")

        image_size = rows * cols
        values = [[0.0]] * count
        for i in range(count):
            values[i] = [read_pixel(f) for _ in range(image_size)]
        return rows, cols, values


def load_labels(path: str):
    with open(path, "rb") as f:
        magic_number = read_int(f)
        count = read_int(f)

        if magic_number != 2049:
            raise ValueError("Invalid file")
        values = [0] * count
        for i in range(count):
            values[i] = read_byte(f)
        return values


def save(name: str, out_path: str, data):
    with open(os.path.join(out_path, f"{name}.dat"), "wb") as f:
        compressed = save_compressed(data)
        f.write(compressed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="path to the images file downloaded from http://yann.lecun.com/exdb/mnist/")
    parser.add_argument("labels", help="path to the labels file downloaded from http://yann.lecun.com/exdb/mnist/")
    parser.add_argument("out", help="path to the output directory")

    args = parser.parse_args()
    (rows, cols, images_data) = load_images(args.images)
    labels_data = load_labels(args.labels)

    train_data = ClassificationDataSet()
    train_data.classes = 10

    test_data = ClassificationDataSet()
    test_data.classes = 10

    for index, (image, label) in enumerate(zip(images_data, labels_data)):
        data = test_data if index < 1_000 else train_data
        example = data.values.add()
        example.inputs.values.extend(image)
        example.inputs.shape.extend((rows, cols))
        example.cls = label

    save("mnist_train", args.out, train_data)
    save("mnist_test", args.out, test_data)


if __name__ == '__main__':
    main()
