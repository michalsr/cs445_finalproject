import os
import json
import os
import random


def make_train_val_lists():
    train_dict = {}
    val_dict = {}
    #image_folders = os.listdir('/data/common/tiny-imagenet-200/train/')
    image_folders = os.listdir('/data/common/imagenette2-160/train/')
    for c in image_folders:
        train_list =  []
        val_list = []
        image_files = os.listdir('/data/common/imagenette2-160/train/'+c+'/')
        for i in image_files:
            n = random.random()
            if n>0.8:
                val_list.append(i)
            else:
                train_list.append(i)
        train_dict[c] = train_list
        val_dict[c] = val_list
    train_dir = '/home/michal5/cs445/'+'train_list_imagenette.json'
    with open(train_dir,'w+') as save_train:
        json.dump(train_dict,save_train)
    val_dir = '/home/michal5/cs445/'+'val_list_imagenette.json'
    with open(val_dir,'w+') as save_val:
        json.dump(val_dict,save_val)

def get_test_list():
    test_dict = {}
    image_folders = os.listdir('/data/common/imagenette2-160/val/')
    for c in image_folders:
        test_list = []
        image_files = os.listdir('/data/common/imagenette2-160/val/'+c+'/')
        for i in image_files:
            test_list.append(i)
        test_dict[c] = test_list
    test_dir = '/home/michal5/cs445/test_list_imagenette.json'
    with open(test_dir,'w+') as save_test:
        json.dump(test_dict,save_test)
make_train_val_lists()
get_test_list()

