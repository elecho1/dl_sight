import chainer
from PIL import Image
import numpy as np
from glob import glob
import os

class MyCifarDataset(chainer.dataset.DatasetMixin):
    def __init__(self, train_dir):
        # DONE !
        # TODO self._pathsに全ての画像のpathをリスト状に保存する
        self._paths = []
        # added
        #files = os.listdir(train_dir)
        #self._paths = [os.path.join(train_dir, f) for f in files if os.path.isdir(os.path.join(train_dir, f))]
        self._paths = glob(train_dir+"/*/*.png")
        # added end

        self._labels = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
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
        image = np.array(Image.open(path).convert('RGB'), np.float32)
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
        str = path.split('/')[-2]
        label = self._labels[str]
        return label
        # raise NotImplementedError
