import chainer
from chainer import functions as F
from PIL import Image
import numpy as np
from glob import glob
import os
import imghdr


class MyImageDataset(chainer.dataset.DatasetMixin):
    def __init__ (self, train_dirs, style="train"):
        # DONE !
        # TODO self._pathsに全ての画像のpathをリスト状に保存する
        self._paths = []
        # added
        # files = os.listdir(train_dir)
        # self._paths = [os.path.join(train_dir, f) for f in files if os.path.isdir(os.path.join(train_dir, f))]

        print(train_dirs)
        print(type(train_dirs))

        for top in train_dirs:
            # for (root, dirs, files) in os.walk(top):  # 再帰的に探索
            #     for file in files:  # ファイル名だけ取得
            #         target = os.path.join(root, file).replace("\\", "/")  # フルパス取得
            #         if os.path.isfile(target):  # ファイルかどうか判別
            #             print(target)
            #             self._paths.append(target)
            list = []
            list.append(glob(top + "/asakusa/*/*"))
            list.append(glob(top + "/meiji/*/*"))
            list.append(glob(top + "/sky_tree/*/*"))
            list.append(glob(top + "/tds/*/*"))
            list.append(glob(top + "/tokyo_tower/*/*"))
            len_list = [len(l) for l in list]
            print(len_list)
            if style == "train":
                min_len = np.min(len_list)

            for i, l in enumerate(list):
                if style == "train":
                    list[i] = np.random.choice(l, min_len, replace=False)
                self._paths.extend(list[i])

                # added end
        print(len(self._paths))

        # added end

        self._labels = {
            'asakusa': 0,
            'meiji': 1,
            'sky_tree': 2,
            'tds': 3,
            'tokyo_tower': 4,
        }
        self._labels_inv = {v: k for k, v in self._labels.items()}
        # print(self._labels_inv)

    def __len__ (self):
        """ データセットの数を返す関数 """
        # print(len(self._paths))
        return len(self._paths)

    def get_example (self, i):
        """    chainerに入力するための画像ファイルと，そのラベルを出力する．"""

        # !DONE
        # TODO self._pathsからi番目のpathを取得して画像として読み込む
        path = self._paths[i]
        # print(path)
        image = np.array(Image.open(path).convert('RGB').resize((224, 224)), np.float32)
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

    def get_label_from_path (self, path):
        # !DONE
        # TODO pathからlabel(0~9)を推測して返す関数を実装せよ
        str = path.split('/')[-3]
        label = self._labels[str]
        return label
        # raise NotImplementedError

    def get_key_from_label (self, label):
        return self._labels_inv[int(label)]

    def get_paths (self):
        return self._paths


class MyTowerDataset(MyImageDataset):
    def __init__ (self, train_dirs, style="train"):
        # DONE !
        # TODO self._pathsに全ての画像のpathをリスト状に保存する
        self._paths = []
        # added
        # files = os.listdir(train_dir)
        # self._paths = [os.path.join(train_dir, f) for f in files if os.path.isdir(os.path.join(train_dir, f))]

        for top in train_dirs:
            list = []
            list.append(glob(top + "/sky_tree/*/*.jpg"))
            list.append(glob(top + "/tokyo_tower/*/*.jpg"))
            len_list = [len(l) for l in list]
            print(len_list)
            if style == "train":
                min_len = np.min(len_list)

            for i, l in enumerate(list):
                if style == "train":
                    list[i] = np.random.choice(l, min_len, replace=False)
                self._paths.extend(list[i])

                # added end
        print(len(self._paths))

        self._labels = {
            'sky_tree': 0,
            'tokyo_tower': 1,
        }
        self._labels_inv = {v: k for k, v in self._labels.items()}
        # print(self._labels_inv)


class MyImage4Dataset(MyImageDataset):
    def __init__ (self, train_dirs, style="train"):
        # DONE !
        # TODO self._pathsに全ての画像のpathをリスト状に保存する
        self._paths = []
        # added

        print(train_dirs)
        for top in train_dirs:
            list = []
            list.append(glob(top + "/asakusa/*/*.jpg"))
            list.append(glob(top + "/sky_tree/*/*.jpg"))
            list.append(glob(top + "/tds/*/*.jpg"))
            list.append(glob(top + "/tokyo_tower/*/*.jpg"))
            len_list = [len(l) for l in list]
            print(len_list)
            if style == "train":
                min_len = np.min(len_list)

            for i, l in enumerate(list):
                if style == "train":
                    list[i] = np.random.choice(l, min_len, replace=False)
                self._paths.extend(list[i])

                # added end
        print(len(self._paths))

        self._labels = {
            'asakusa': 0,
            'sky_tree': 1,
            'tds': 2,
            'tokyo_tower': 3,
        }
        self._labels_inv = {v: k for k, v in self._labels.items()}
        # print(self._labels_inv)


class MyImageNewDataset(MyImageDataset):
    # No tds
    def __init__ (self, train_dirs, style="train"):
        # DONE !
        # TODO self._pathsに全ての画像のpathをリスト状に保存する
        self._paths = []
        # addedclass MyImageNewDataset(MyImageDataset):
        # files = os.listdir(train_dir)
        # self._paths = [os.path.join(train_dir, f) for f in files if os.path.isdir(os.path.join(train_dir, f))]

        print(train_dirs)
        # print(type(train_dirs))

        for i, top in enumerate(train_dirs):
            # for (root, dirs, files) in os.walk(top):  # 再帰的に探索
            #     for file in files:  # ファイル名だけ取得
            #         target = os.path.join(root, file).replace("\\", "/")  # フルパス取得
            #         if os.path.isfile(target):  # ファイルかどうか判別
            #             print(target)
            #             self._paths.append(target)
            if i == 0:
                list = []
                list.append(glob(top + "/asakusa/*/*"))
                list.append(glob(top + "/meiji/*/*"))
                list.append(glob(top + "/sky_tree/*/*"))
                # list.append(glob(top + "/tds/*/*.jpg"))
                list.append(glob(top + "/tokyo_tower/*/*"))
            else:
                list[0].extend(glob(top + "/asakusa/*/*"))
                list[1].extend(glob(top + "/meiji/*/*"))
                list[2].extend(glob(top + "/sky_tree/*/*"))
                # list[3].extend(glob(top + "/tds/*/*.jpg"))
                list[3].extend(glob(top + "/tokyo_tower/*/*"))
            len_list = [len(l) for l in list]
            print(len_list)


        if style == "train":
            min_len = np.min(len_list)

        for i, l in enumerate(list):
            if style == "train":
                list[i] = np.random.choice(l, min_len, replace=False)
            self._paths.extend(list[i])

            # added end

        print(len(self._paths))

        # added end

        self._labels = {
            'asakusa': 0,
            'meiji': 1,
            'sky_tree': 2,
            'tokyo_tower': 3,
        }
        self._labels_inv = {v: k for k, v in self._labels.items()}

        # print(self._labels_inv)


    def get_example (self, i, style="NotProcess"):
        path = self._paths[i]

        style = "hoge"
        if style == "NotProcess":
            image = np.array(Image.open(path).convert('RGB'), np.float32)
        else:
            image = np.array(Image.open(path).convert('RGB').resize((224, 224)), np.float32)

        image = image[:, :, ::-1]
        image -= np.array(
            [103.063, 115.903, 123.152], dtype=np.float32)
        image_for_chainer = image.transpose((2, 0, 1))
        # print(image_for_chainer.shape)

        label = self.get_label_from_path(path)
        return image_for_chainer, label


class MyImageNewNoCompDataset(MyImageDataset):
    # No tds
    def __init__ (self, train_dirs, style="train"):
        # DONE !
        # TODO self._pathsに全ての画像のpathをリスト状に保存する
        self._paths = []
        # addedclass MyImageNewDataset(MyImageDataset):
        # files = os.listdir(train_dir)
        # self._paths = [os.path.join(train_dir, f) for f in files if os.path.isdir(os.path.join(train_dir, f))]

        print(train_dirs)
        print(type(train_dirs))

        for top in train_dirs:
            # for (root, dirs, files) in os.walk(top):  # 再帰的に探索
            #     for file in files:  # ファイル名だけ取得
            #         target = os.path.join(root, file).replace("\\", "/")  # フルパス取得
            #         if os.path.isfile(target):  # ファイルかどうか判別
            #             print(target)
            #             self._paths.append(target)
            list = []
            list.append(glob(top + "/asakusa/*/*"))
            list.append(glob(top + "/meiji/*/*"))
            list.append(glob(top + "/sky_tree/*/*"))
            # list.append(glob(top + "/tds/*/*.jpg"))
            list.append(glob(top + "/tokyo_tower/*/*"))
            len_list = [len(l) for l in list]
            print(len_list)
            if style == "train":
                min_len = np.min(len_list)

            for i, l in enumerate(list):
                if style == "train":
                    list[i] = np.random.choice(l, min_len, replace=False)
                self._paths.extend(list[i])

                # added end
        print(len(self._paths))

        # added end

        self._labels = {
            'asakusa': 0,
            'meiji': 1,
            'sky_tree': 2,
            'tokyo_tower': 3,
        }
        self._labels_inv = {v: k for k, v in self._labels.items()}

        # print(self._labels_inv)
