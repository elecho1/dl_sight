import chainer

import argparse
import os
import numpy as np
from PIL import Image
from chainercv import transforms as T

from multiprocessing import Process, Queue, current_process

NUMBER_OF_PROCESSES = 12

outdir =""
dataset = ""
args = []

def worker(input, output):
    for path in iter(input.get, 'STOP'):
        image = Image.open(path).convert('RGB')
        if args.doresize:
            exportResized224Image(image, path)
        if args.dofliph:
            exportFlipH224Image(image, path)
        if args.docrop:
            exportRandomCropped224Image(image, path, args.randomcroptimes)
        output.put(path)

def main():
    parser = argparse.ArgumentParser(description='One Practice: Tokyo sight')
    ## multiple
    # parser.add_argument('--dataset', '-d', default=['image/test'], nargs="*", help='Directory for train mini_cifar')
    ## single
    parser.add_argument('--dataset', '-d', default='image/test',  help='Directory for train mini_cifar')
    parser.add_argument('--output', '-o', default='image/train_aug_cv', help="Directory for augmented images")
    parser.add_argument('--randomcroptimes', '-c', type=int, default=10, help="Times to export cropped times.")
    parser.add_argument('--dofliph', '-H', type=bool, default=False, help="Whether to do fliph.")
    parser.add_argument('--doresize', '-R', type=bool, default=True, help="Whether to output just resized image.")
    parser.add_argument('--docrop', '-C', type=bool, default=True, help="Whether to do random cropping.")
    global args
    args = parser.parse_args()


    _paths = []

    global outdir
    outdir = args.output
    global dataset
    dataset = args.dataset

    # for top in args.dataset:
    #     for (root, dirs, files) in os.walk(top):  # 再帰的に探索
    #         for file in files:  # ファイル名だけ取得
    #             target = os.path.join(root, file).replace("\\", "/")  # フルパス取得
    #             if os.path.isfile(target):  # ファイルかどうか判別
    #                 # print(target)
    #                 _paths.append(target)

    for (root, dirs, files) in os.walk(args.dataset): # 再帰的に探索
        for file in files:  # ファイル名だけ取得
             target = os.path.join(root, file).replace("\\", "/")
             if os.path.isfile(target):  # ファイルかどうか判別
                 # print(target)
                 _paths.append(target)

                        # To augment each file
    # for path in _paths:
    #     print(path)
    #     # image = np.array(Image.open(path).convert('RGB').resize((224, 224)), np.float32)
    #     # image = np.array(Image.open(path).convert('RGB').resize((256, 256)), np.float32)
    #     image = Image.open(path).convert('RGB')
    #     # generate just resized image
    #     if args.doresize:
    #         exportResized224Image(image, path)
    #     if args.dofliph:
    #         exportFlipH224Image(image, path)
    #     if args.docrop:
    #         exportRandomCropped224Image(image, path, args.randomcroptimes)

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for task in _paths:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    print('Unordered results:')
    for i in range(len(_paths)):
        print('\t', done_queue.get())

    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')


def exportResized224Image(image, path):
    filename = culculateFilename(path,  "_rs224")

    if filename is not None:
        # image = Image.fromarray(np.uint8(image))
        new_image = image.resize((224, 224))
        new_image.save(filename)

def exportFlipH224Image(image, path):
    filename = culculateFilename(path, "_fh224")
    # print(filename)
    if filename is not None:
        # new_image = image.resize((224, 224))
        new_image = np.array(Image.open(path).convert('RGB').resize((256, 256)), np.float32)
        new_image = new_image.transpose((2, 0, 1))

        new_image = T.flip(new_image, x_flip=True)
        flipped_image = Image.fromarray(np.uint8(new_image.transpose((1, 2, 0))))
        flipped_image.save(filename)

def exportRandomCropped224Image(image, path, times):
    new_image = np.array(Image.open(path).convert('RGB').resize((256, 256)), np.float32)
    new_image = new_image.transpose((2, 0, 1))

    orig_filename = culculateFilename(path, "_cr224")
    # print(filename)
    for i in range(times):
        temp_filename = orig_filename.split('.')
        temp_filename[-2] = temp_filename[-2]+'_'+str(i)
        filename = '.'.join(temp_filename)
        if os.path.isfile(filename):
            print(filename, "already exists.")
        else:
            cropped_image_ndarray = T.random_crop(new_image, (224, 224))
            cropped_image = Image.fromarray(np.uint8(cropped_image_ndarray.transpose((1, 2, 0))))
            cropped_image.save(filename)

def culculateFilename(path, qual):
    splited_path = path.split('/')
    splited_path[-2] = splited_path[-2] + qual
    splited_filename = splited_path[-1].split('.')
    splited_filename[-2] = splited_filename[-2] + qual
    # print(splited_filename)
    splited_path[-1] = '.'.join(splited_filename)
    # print(splited_path)
    new_file_path = '/'.join(splited_path)

    # replace_file_path_parts = new_file_path.split(dataset)
    print(new_file_path.split(dataset))
    replace_file_path_parts = new_file_path.split(dataset)
    for i, part in enumerate(replace_file_path_parts):
        if part == '':
            replace_file_path_parts[i] = outdir

    new_file_path = ''.join(replace_file_path_parts)
    print(new_file_path)
    # exit()

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