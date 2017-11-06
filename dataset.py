import chainer
from chainer import functions as F
from PIL import Image
import numpy as np
from glob import glob
import os
import imghdr

class MyImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, train_dirs):
        # DONE !
        # TODO self._pathsに全ての画像のpathをリスト状に保存する
        self._paths = []
        # added
        #files = os.listdir(train_dir)
        #self._paths = [os.path.join(train_dir, f) for f in files if os.path.isdir(os.path.join(train_dir, f))]

        print(train_dirs)
        print(type(train_dirs))

        for top in train_dirs:
            for (root, dirs, files) in os.walk(top):  # 再帰的に探索
                for file in files:  # ファイル名だけ取得
                    target = os.path.join(root, file).replace("\\", "/")  # フルパス取得
                    if os.path.isfile(target):  # ファイルかどうか判別
                        print(target)
                        self._paths.append(target)


        # self._paths = glob(train_dir+"/*/*/*.jpg")
        # self._paths += glob(train_dir + "/*/*/*.png")
        # self._paths += glob(train_dir + "/*/*/*.JPG")
        # self._paths += glob(train_dir + "/*/*/*.PNG")

         # added end

        self._labels = {
            'asakusa': 0,
            'meiji': 1,
            'sky_tree': 2,
            'tds': 3,
            'tokyo_tower': 4,
        }
        self._labels_inv = {v:k for k, v in self._labels.items()}
        # print(self._labels_inv)

    def __len__(self):
        """ データセットの数を返す関数 """
        # print(len(self._paths))
        return len(self._paths)

    def get_example(self, i):
        """    chainerに入力するための画像ファイルと，そのラベルを出力する．"""

        # !DONE
        # TODO self._pathsからi番目のpathを取得して画像として読み込む
        path = self._paths[i]
        # print(path)
        image = np.array(Image.open(path).convert('RGB').resize((224,224)), np.float32)
        # print(image.shape)

        # print("path:", path)
        # print("len of image", len(image))


        # 画像データは3次元配列で入っており，その軸は(幅，高さ，チャンネル)の順である
        # !DONE
        # TODO chainerは(チャンネル，幅，高さ)で入力することを求めるのでtransposeを行なう
        # image_for_chainer = image.transpose(('?', '?', '?'))
        image_for_chainer = image.transpose((2, 0, 1))

        # pathから画像のラベルを取得することが出来る．
        # ラベルはairplaneのような文字列型ではなく，0のような整数である必要がある．
        label = self.get_label_from_path(path)
        return image_for_chainer, label

    def get_label_from_path(self, path):
        # !DONE
        # TODO pathからlabel(0~9)を推測して返す関数を実装せよ
        str = path.split('/')[-3]
        label = self._labels[str]
        return label
        # raise NotImplementedError

    def get_key_from_label(self, label):
        return self._labels_inv[label]

    def get_paths(self):
        return self._paths

class MyTowerDataset(MyImageDataset):
    def __init__(self, train_dirs):
        # DONE !
        # TODO self._pathsに全ての画像のpathをリスト状に保存する
        self._paths = []
        # added
        #files = os.listdir(train_dir)
        #self._paths = [os.path.join(train_dir, f) for f in files if os.path.isdir(os.path.join(train_dir, f))]

        for top in train_dirs:
            list = []
            # self._paths = glob(train_dir+"/*/*/*.jpg")
            # self._paths += glob(train_dir + "/*/*/*.png")
            # self._paths += glob(train_dir + "/*/*/*.JPG")
            # self._paths += glob(train_dir + "/*/*/*.PNG")
            list.append(glob(top+"/sky_tree/*/*.jpg"))
            list.append(glob(top + "/tokyo_tower/*/*.jpg"))
            len_list = [len(l) for l in list]
            print(len_list)
            min_len = np.min(len_list)

            for i, l in enumerate(list):
                list[i] = np.random.choice(l, min_len, replace=False)
                self._paths.extend(list[i])

         # added end
        print(len(self._paths))

        self._labels = {
            'sky_tree': 0,
            'tokyo_tower': 1,
        }
        self._labels_inv = {v:k for k, v in self._labels.items()}
        # print(self._labels_inv)