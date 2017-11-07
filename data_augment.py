import chainer

import argparse
import os
import numpy as np
from PIL import Image
from chainercv import transforms as T

def main():
    parser = argparse.ArgumentParser(description='One Practice: Tokyo sight')
    parser.add_argument('--dataset', '-d', default=['image/test'], nargs="*",
                        help='Directory for train mini_cifar')
    parser.add_argument('--output', '-o', default='image/train_aug_cv', help="Directory for augmented images")
    args = parser.parse_args()

    _paths = []
    for top in args.dataset:
        for (root, dirs, files) in os.walk(top):  # 再帰的に探索
            for file in files:  # ファイル名だけ取得
                target = os.path.join(root, file).replace("\\", "/")  # フルパス取得
                if os.path.isfile(target):  # ファイルかどうか判別
                    print(target)
                    _paths.append(target)

    # To augment each file
    for path in _paths:
        # image = np.array(Image.open(path).convert('RGB').resize((224, 224)), np.float32)
        image = np.array(Image.open(path).convert('RGB').resize((256, 256)), np.float32)

        # generate just resized image
        exportResized224Image(image, path)



def exportResized224Image(image, path):
    pilImg = Image.fromarray(np.uint8(image))
    new_image = pilImg.convert('RGB').resize((224, 224))

    filename = culculateFilename(path, "_rs224")
    print(filename)
    if filename is not None:
        new_image.save(filename)

def culculateFilename(path, qual):
    splited_path = path.split('/')
    splited_path[-2] = splited_path[-2] + qual
    splited_filename = splited_path[-1].split('.')
    splited_filename[-2] = splited_filename[-2] + qual
    # print(splited_filename)
    splited_path[-1] = '.'.join(splited_filename)
    # print(splited_path)
    new_file_path = '/'.join(splited_path)
    # print(new_file_path)
    new_dir_name = os.path.dirname(new_file_path)
    if not os.path.isdir(new_dir_name):
        os.makedirs(new_dir_name)
    if os.path.isfile(new_file_path):
        print(new_file_path, "already exists.")
        return None
    else:
        return new_file_path



if __name__ == '__main__':
    main()