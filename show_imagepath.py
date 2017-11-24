import chainer

import argparse
import os
import numpy as np
from PIL import Image
from chainercv import transforms as T

def main():
    parser = argparse.ArgumentParser(description='One Practice: Tokyo sight')
    ## multiple
    # parser.add_argument('--dataset', '-d', default=['image/test'], nargs="*", help='Directory for train mini_cifar')
    ## single
    parser.add_argument('--dataset', '-d', default='image/test',  help='Directory for train mini_cifar')

    args = parser.parse_args()

    _paths = []
    for (root, dirs, files) in os.walk(args.dataset): # 再帰的に探索
        for file in files:  # ファイル名だけ取得
             target = os.path.join(root, file).replace("\\", "/")
             if os.path.isfile(target):  # ファイルかどうか判別
                 # print(target)
                 _paths.append(target)

    for path in _paths:
        print(path)
        image = Image.open(path)

if __name__ == '__main__':
    main()