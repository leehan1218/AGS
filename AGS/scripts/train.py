import torch
import numpy as np
import time  # 학습시간이 얼마나 걸리는지, 체크포인트 관리할 때도 사용함
import re  # 정규표현식 사용
import random  # seed 랜덤 변수
import yaml  # 하이퍼파라미터 관리
import smart_open  # 파일 입출력
import pickle  # 딕셔너리나, 자료형 저장할 때 사용
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision import transforms, models
from sklearn.model_selection import KFold
# torch 관련 함수와 패키지
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
# python libraties
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import itertools
from torchvision.models import densenet201
from tqdm import tqdm
from glob import glob
from PIL import Image

from torchvision.models import densenet201

# pytorch libraries
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from CV.utils.text_prepro import compute_img_mean_std, get_duplicates, get_val_rows, HAM10000, AverageMeter
from CV.scripts.module_A.module_a import module_a
from CV.scripts.module_S.Module_S import module_s
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


def main():
    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/efficientnet.yaml'
        # params_filename = '../config/text_cnn.yaml'
    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # 랜덤 시드 세팅
    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    data_dir = "../data/HAM10000_image"
    all_image_path = glob(os.path.join(data_dir,'*','*.jpg'))
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
    # norm_mean, norm_std = compute_img_mean_std(all_image_path)
    df_original = pd.read_csv(os.path.join("../data/HAM10000_metadata.csv"))
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



    df_val, df_test = train_test_split(df_val, test_size=0.5)
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    df_test = df_test.reset_index()
    normMean = [0.7630331, 0.5456457, 0.5700467]
    normStd = [0.1409281, 0.15261227, 0.16997086]


    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),

                                        transforms.Normalize(normMean, normStd)
                                        ])

    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),

                                         transforms.Normalize(normMean, normStd)
                                         ])
    """ module A """
    train_set = module_a('"../data/HAM10000_metadata.csv"')
    test_set = HAM10000(df_test, transform=test_transform)

    """ module G """
    train_gan_set = torchvision.datasets.ImageFolder(
        root="../data/ham10000_ff",
        transform=val_transform)

    percentage = 0.5

    # Calculate the number of samples to keep
    num_samples = int(len(train_gan_set) * percentage)

    # Create a random subset of the dataset
    subset_indices = torch.randperm(len(train_gan_set))[:num_samples]
    train_gan_set = Subset(train_gan_set, subset_indices)


    """ module S """
    df_train = module_s('../data/Segmen_class')
    training_seg_train = HAM10000(df_train, transform=val_transform)


    # train 데이터에 gan만 쓰려면 밑 코드를 train_gan_set 만 넣으면 됨. 3+4 면 [train_gan_set,  train_seg_set] 이렇게 하면 됨
    train_datasets = [train_set, train_gan_set, training_seg_train]



    # Same for the validation set:
    validation_set = HAM10000(df_val, transform=val_transform)
    val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)

    train_dataset = ConcatDataset(train_datasets)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)


    # 학습 모델 생성

    criterion = nn.CrossEntropyLoss().to(device)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    summary_dir = os.path.join(out_dir, "summaries")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer = SummaryWriter(summary_dir)  # TensorBoard를 위한 초기화

    # training 시작
    start_time = time.time()

    global_steps = 0
    print('========================================')
    print("Start training...")

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    epoch_num = 50
    best_val_acc = 0
    total_loss_val, total_acc_val = [], []
    total_train_time = 0  # Initialize total training time
    total_loss_train, total_acc_train = [], []
    dataset = ConcatDataset([train_dataset])

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        start_time = time.time()  # Record the start time for each epoch
        out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp, f"fold{fold}")))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        summary_dir = os.path.join(out_dir, "summaries")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Print
        print(f'FOLD {fold + 1}')
        print('--------------------------------')
        writer = SummaryWriter(summary_dir)
        # Define data loaders for training and testing data in this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32, sampler=test_subsampler)
        print("trainloader : ",len(trainloader.dataset))
        print("testloader : ",len(testloader.dataset))

        # Init the neural network
        model = models.googlenet(pretrained=True)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 7)
        model.to("cuda")
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        # Initialize optimizer

        for epoch in range(1, epoch_num + 1):
            print(f'Starting epoch {epoch}')

            train_loss = AverageMeter()
            train_acc = AverageMeter()
            curr_iter = (epoch - 1) * len(trainloader)
            for i, data in enumerate(trainloader, 0):
                images, labels = data
                N = images.size(0)
                # print('image shape:',images.size(0), 'label shape',labels.size(0))
                images = Variable(images).to(device)
                labels = Variable(labels).to(device)

                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                prediction = outputs.max(1, keepdim=True)[1]
                train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)
                train_loss.update(loss.item())
                curr_iter += 1

                if (i + 1) % 100 == 0:
                    print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                        epoch, i + 1, len(trainloader), train_loss.avg, train_acc.avg))
                    total_loss_train.append(train_loss.avg)
                    total_acc_train.append(train_acc.avg)
                    writer.add_scalar('Train/Loss', train_loss.avg, global_steps)
                    writer.add_scalar('Train/Accuracy', train_acc.avg, global_steps)
                    global_steps += 1

            model.eval()
            val_loss = AverageMeter()
            val_acc = AverageMeter()
            with torch.no_grad():
                for i, data in enumerate(testloader):
                    images, labels = data
                    N = images.size(0)
                    images = Variable(images).to(device)
                    labels = Variable(labels).to(device)

                    outputs = model(images)
                    prediction = outputs.max(1, keepdim=True)[1]

                    val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)

                    val_loss.update(criterion(outputs, labels).item())

            print('------------------------------------------------------------')
            print('[Fold %d, epoch %d], [val loss %.5f], [val acc %.5f]' % (fold + 1, epoch, val_loss.avg, val_acc.avg))
            print('------------------------------------------------------------')
            writer.add_scalar('Validation/Loss', val_loss.avg, epoch)
            writer.add_scalar('Validation/Accuracy', val_acc.avg, epoch)
            loss_val, acc_val = val_loss.avg, val_acc.avg
            lr_scheduler.step(loss_val)
            total_loss_val.append(loss_val)
            total_acc_val.append(acc_val)

            end_time = time.time()  # Record the end time for each epoch
            epoch_time = end_time - start_time
            total_train_time += epoch_time

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                save_path = f'best_Resnet_epoch50_v12_fold{fold + 1}.pth'
                torch.save(model.state_dict(), save_path)  # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping
                print('*****************************************************')
                print(f'best record for Fold {fold + 1}: [epoch {epoch}], [val loss {loss_val:.5f}], '
                      f'[val acc {acc_val:.5f}]')
                print('*****************************************************')

        # Print the total training time after all epochs are completed for the current fold
        print(f'Total training time for Fold {fold + 1}: %.2f seconds' % total_train_time)

        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                images, labels = data
                N = images.size(0)
                images = Variable(images).to(device)
                labels = Variable(labels).to(device)

                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        accuracy = accuracy_score(all_labels, all_predictions)

        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1 Score: {:.4f}".format(f1))
        print("Accuracy: {:.4f}".format(accuracy))

        # 평가 지표를 리스트에 추가
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Print the total training time after all epochs are completed
    print('Total training time: %.2f seconds' % total_train_time)
    writer.close()
    torch.save(model.state_dict(), "Resnet_epoch50_v12.pth")

    # 모든 fold의 평가가 끝난 후, 평균 성능을 계산하고 출력합니다.
    print("========================================")
    print("Average performance across all folds:")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f}")


    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting normalize=True.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    model.eval()
    y_label = []
    y_predict = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_label, y_predict)
    # plot the confusion matrix
    plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    plot_confusion_matrix(confusion_mtx, plot_labels)

    report = classification_report(y_label, y_predict, target_names=plot_labels)
    print(report)


if __name__ == '__main__':
    main()
