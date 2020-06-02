import pickle as plt
from PIL import Image
from data.image_folder import make_dataset
import os
from sklearn.model_selection import train_test_split


dirs = ["../datasets/horse2zebra/testA",
        "../datasets/horse2zebra/testB"]

for dir in dirs:
    print(dir)
    paths = sorted(make_dataset(dir))
    data = [(os.path.basename(path), Image.open(path).convert('RGB')) for path in paths]

    with open("{}.plt".format(dir), 'wb') as f:
        plt.dump(data, f)

dirs = ["../datasets/horse2zebra/trainA",
        "../datasets/horse2zebra/trainB"]

for dir in dirs:
    print(dir)
    paths = sorted(make_dataset(dir))
    data = [(os.path.basename(path), Image.open(path).convert('RGB')) for path in paths]

    train, valid = train_test_split(data, test_size=0.1)

    with open("{}.plt".format(dir), 'wb') as f:
        plt.dump(train, f)
    path, name = os.path.split(dir)

    with open("{}.plt".format(os.path.join(path, "valid{}".format(name[-1]))), 'wb') as f:
        plt.dump(valid, f)