import argparse

import matplotlib

matplotlib.use('Agg')
import chainer
from net import Cifar_CNN
# from dataset import MyCifarDataset
# from dataset import MyImageDataset
from dataset import MyTowerDataset
from net import ResNet50toNClass


def main():
    parser = argparse.ArgumentParser(description='One Practice: Tokyo sight')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='result/tower/model_20',
                        help='Path to the model')
    parser.add_argument('--dataset', '-d', default=['image/test'], nargs="*",
                        help='Directory for train mini_cifar')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('')

    #model = Cifar_CNN(10)
    model = ResNet50toNClass(2)
    chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Load the Cifar-10 mini_cifar
    # trainとvalに分ける
    test = MyTowerDataset(args.dataset)
    print('test data : {}'.format(len(test)))

    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                  repeat=False, shuffle=False)

    correct_cnt = 0
    paths = test.get_paths()
    current_itr = 0
    while True:
        try:
            batch = test_iter.next()
        except StopIteration:
            break
        images = model.xp.array([image for image, _ in batch])
        labels = model.xp.array([label for _, label in batch])
        predicts = model.predict(images)
        for l, p in zip(labels, predicts):
            # print("label:", test.get_key_from_label(l),',', "predict:", test.get_key_from_label(p), '\n')
            path = paths[current_itr]
            print("path:", path, "label:", test.get_key_from_label(l), ',', "predict:", test.get_key_from_label(p),
                  '\n')
            if l == p:
                correct_cnt += 1

        current_itr += 1


    # print('accuracy : {}'.format(correct_cnt/len(test)))


if __name__ == '__main__':
    main()