import torch
import numpy as np
import time #학습시간이 얼마나 걸리는지, 체크포인트 관리할 때도 사용함
import re #정규표현식 사용
import random #seed 랜덤 변수
import yaml #하이퍼파라미터 관리
import smart_open #파일 입출력
import pickle #딕셔너리나, 자료형 저장할 때 사용
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision import transforms,models

#torch 관련 함수와 패키지
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# python libraties
import os, cv2,itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from tqdm import tqdm
from glob import glob
from PIL import Image

# pytorch libraries
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from CV.models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, initialize_model
from CV.utils.text_prepro import compute_img_mean_std, get_duplicates, get_val_rows, HAM10000, AverageMeter
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

def module_a(data):
    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/efficientnet.yaml'
        # params_filename = '../config/text_cnn.yaml'
    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    data_params = params['data_files'][params['task']]

    # 랜덤 시드 세팅
    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 데이터 로드
    data_dir = '../data'
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    norm_mean, norm_std = compute_img_mean_std(all_image_path)
    df_original = pd.read_csv(os.path.join(data))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    df_original.head()

    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    df_undup.head()

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
    df_original.head()

    df_original['duplicates'].value_counts()

    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    df_undup.shape
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
    df_val.shape

    df_val['cell_type_idx'].value_counts()

    # identify train and val rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_val'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']
    print(len(df_train))
    print(len(df_val))

    df_train['cell_type_idx'].value_counts()

    df_val['cell_type'].value_counts()

    data_aug_rate = [15, 10, 5, 50, 0, 40, 5]
    for i in range(7):
        if data_aug_rate[i]:
            df_train = df_train.append([df_train.loc[df_train['cell_type_idx'] == i, :]] * (data_aug_rate[i] - 1),
                                       ignore_index=True)
    df_train['cell_type'].value_counts()

    df_val, df_test = train_test_split(df_val, test_size=0.5)
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    df_test = df_test.reset_index()
    normMean, normStd = compute_img_mean_std(all_image_path)
    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                          transforms.ToTensor(),

                                          transforms.Normalize(normMean, normStd),
                                          ])

    training_set = HAM10000(df_train, transform=train_transform)
    return training_set