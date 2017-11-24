import argparse

import matplotlib

matplotlib.use('Agg')
import chainer
from net import Cifar_CNN
# from dataset import MyCifarDataset
# from dataset import MyImageDataset
from dataset import MyImageNewDataset
from net import ResNet50toNClass
import json
import gc
import os
from glob import glob
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

gc.enable()
CLASS_NUM = 4

def main():
    parser = argparse.ArgumentParser(description='One Practice: Tokyo sight')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--modeldir', '-m', default='result/sight_new_orig/',
                        help='Dir path to the model')
    parser.add_argument('--dataset', '-d', default=['image/test'], nargs="*",
                        help='Directory for test dataset')
    parser.add_argument('--firstmodel', '-f', type=int, default=None,
                        help='The epoch of first model to test')
    parser.add_argument('--lastmodel', '-l', type=int, default=None,
                        help='The epoch of last model to test')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('')

    test = MyImageNewDataset(args.dataset)
    print('test data : {}'.format(len(test)))

    ### To determine range of models to test
    _models = glob(args.modeldir + "/model_*")
    models_epochs = [int(each_model.split("_")[-1]) for each_model in _models]
    models_epochs.sort()
    if args.firstmodel is None:
        first_epoch = models_epochs[0]
    else:
        first_epoch = args.firstmodel
    if args.lastmodel is None:
        last_epoch = models_epochs[-1]
    else:
        last_epoch = args.lastmodel

    print("first model :", args.modeldir+"/model_"+str(first_epoch))
    print("last model :", args.modeldir + "/model_" + str(last_epoch))

    ### DataFrame to store accuracy of each model
    _cols_accuracy = ['total']
    _cols_accuracy.extend([test.get_key_from_label(i) for i in range(CLASS_NUM)])
    # df_accuracy = DataFrame(columns = _cols_accuracy)
    nd_accuracy = np.empty((0, len(_cols_accuracy)))
    index_accuracy = []

    ### DataFrame to store predict-result of each test data
    _cols_predict = test.get_paths()
    df_predict = DataFrame(columns = _cols_predict)
    nd_predict = np.empty((0, len(_cols_predict)))
    index_predict = []
    list_origlabel = [test.get_label_from_path(label) for label in _cols_predict]
    nd_predict = np.append(nd_predict, np.array([list_origlabel]), axis=0)
    index_predict.append(-1)

    # sr_origlabel = Series(list_origlabel, index=_cols_predict, name=-1)
    # df_predict.append(sr_origlabel)


    for epoch in range(first_epoch, last_epoch+1):
        model = ResNet50toNClass(CLASS_NUM)
        model_path = args.modeldir+"/model_"+str(epoch)
        print("current model :", model_path)
        chainer.serializers.load_npz(model_path, model)

        if args.gpu >= 0:
            # Make a specified GPU current
            chainer.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()  # Copy the model to the GPU

        test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

        correct_cnt = 0
        each_correct_cnt = []
        each_cnt = []
        dict_predict = {}  # for df_predict

        for i in range(CLASS_NUM):
            each_correct_cnt.append(0)
            each_cnt.append(0)

        paths = test.get_paths()
        current_itr = 0
        while True:
            try:
                batch = test_iter.next()
            except StopIteration:
                break
            images = model.xp.array([image for image, _ in batch])
            labels = model.xp.array([label for _, label in batch])
            with chainer.using_config('train', False):
                predicts = model.predict(images)
            for l, p in zip(labels, predicts):
                each_cnt[int(l)] += 1
                dict_predict[paths[current_itr]] = int(p)  ### for df_predict
                current_itr += 1
                if l == p:
                    correct_cnt += 1
                    each_correct_cnt[int(l)] += 1

        each_accuracy = []  ### for df_accuracy
        ### print info
        print('accuracy : {}'.format(correct_cnt/len(test)))
        each_accuracy.append(correct_cnt/len(test))
        for i in range(CLASS_NUM):
            ### print info
            print(test.get_key_from_label(i), " : ", each_correct_cnt[i]/each_cnt[i])
            each_accuracy.append(each_correct_cnt[i]/each_cnt[i]) ### for_df_accuracy
        ### for df_accuracy
        # sr_accuracy = Series(each_accuracy, index=_cols_accuracy, name=int(epoch))  # for df_accuracy
        # df_accuracy = df_accuracy.append(sr_accuracy)
        nd_accuracy = np.append(nd_accuracy, np.array([each_accuracy]), axis=0)
        index_accuracy.append(int(epoch))

        ### for df_predict
        # sr_predict = Series(dict_predict, name=int(epoch))
        # df_predict = df_predict.append(sr_predict)
        each_predict = [dict_predict[col] for col in _cols_predict]

        nd_predict = np.append(nd_predict, np.array([each_predict]), axis=0)
        index_predict.append(int(epoch))

        del model
        del test_iter
        del images
        del labels
        del predicts
        gc.collect()

    df_accuracy = DataFrame(nd_accuracy, columns=_cols_accuracy, index=index_accuracy)
    df_accuracy.to_csv(args.modeldir + "/test_accuracy.csv")

    df_predict = DataFrame(nd_predict, columns=_cols_predict, index=index_predict)
    df_predict.to_csv(args.modeldir + "/test_predict.csv")


if __name__ == '__main__':
    main()