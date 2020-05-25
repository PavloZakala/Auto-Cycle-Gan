import pickle as plt
from PIL import Image
from data.image_folder import make_dataset

dirs = ["../datasets/horse2zebra/trainA",
       "../datasets/horse2zebra/trainB",
       "../datasets/horse2zebra/testA",
       "../datasets/horse2zebra/testB"]

for dir in dirs:
    print(dir)
    paths = sorted(make_dataset(dir))
    data = [Image.open(path).convert('RGB') for path in paths]

    with open("{}.plt".format(dir), 'wb') as f:
        plt.dump(data, f)