import chainer

import argparse
import os
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





if __name__ == '__main__':
    main()