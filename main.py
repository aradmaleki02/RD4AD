# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import shutil
from pathlib import Path

import torch
from torchvision import transforms

from dataset import get_data_transforms, BrainTrain, BrainTest, GTA, GTA_Test
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader

from mvtec import MVTEC
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test
from torch.nn import functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss


def prepare_br35h_dataset_files():
    normal_path35 = '/kaggle/input/brain-tumor-detection/no'
    anomaly_path35 = '/kaggle/input/brain-tumor-detection/yes'

    print(f"len(os.listdir(normal_path35)): {len(os.listdir(normal_path35))}")
    print(f"len(os.listdir(anomaly_path35)): {len(os.listdir(anomaly_path35))}")

    print('cnt')

    Path('./Br35H/dataset/test/anomaly').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./Br35H/dataset/test/anomaly')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/test/anomaly', f))

    anom35 = os.listdir(anomaly_path35)
    for f in anom35:
        shutil.copy2(os.path.join(anomaly_path35, f), './Br35H/dataset/test/anomaly')


    normal35 = os.listdir(normal_path35)
    random.shuffle(normal35)
    ratio = 0.7
    sep = round(len(normal35) * ratio)

    Path('./Br35H/dataset/test/normal').mkdir(parents=True, exist_ok=True)
    Path('./Br35H/dataset/train/normal').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./Br35H/dataset/test/normal')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/test/normal', f))

    flist = [f for f in os.listdir('./Br35H/dataset/train/normal')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/train/normal', f))

    for f in normal35[:sep]:
        shutil.copy2(os.path.join(normal_path35, f), './Br35H/dataset/train/normal')
    for f in normal35[sep:]:
        shutil.copy2(os.path.join(normal_path35, f), './Br35H/dataset/test/normal')


def prepare_brats2015_dataset_files():
    import pandas as pd
    labels = pd.read_csv('/kaggle/input/brain-tumor/Brain Tumor.csv')
    labels = labels[['Image', 'Class']]
    labels.tail() # 0: no tumor, 1: tumor

    labels.head()

    brats_path = '/kaggle/input/brain-tumor/Brain Tumor/Brain Tumor'
    lbl = dict(zip(labels.Image, labels.Class))

    keys = lbl.keys()
    normalbrats = [x for x in keys if lbl[x] == 0]
    anomalybrats = [x for x in keys if lbl[x] == 1]

    Path('./brats/dataset/test/anomaly').mkdir(parents=True, exist_ok=True)
    Path('./brats/dataset/test/normal').mkdir(parents=True, exist_ok=True)
    Path('./brats/dataset/train/normal').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./brats/dataset/test/anomaly')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/test/anomaly', f))

    flist = [f for f in os.listdir('./brats/dataset/test/normal')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/test/normal', f))

    flist = [f for f in os.listdir('./brats/dataset/train/normal')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/train/normal', f))

    ratio = 0.7
    random.shuffle(normalbrats)
    bratsep = round(len(normalbrats) * ratio)

    for f in anomalybrats:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/test/anomaly')
    for f in normalbrats[:bratsep]:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/train/normal')
    for f in normalbrats[bratsep:]:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/test/normal')

def get_cityscape_globs():
    from glob import glob
    import random
    normal_path = glob('/kaggle/input/cityscapes-5-10-threshold/cityscapes/ID/*')
    anomaly_path = glob('/kaggle/input/cityscapes-5-10-threshold/cityscapes/OOD/*')

    random.seed(42)
    random.shuffle(normal_path)
    train_ratio = 0.7
    separator = int(train_ratio * len(normal_path))
    normal_path_train = normal_path[:separator]
    normal_path_test = normal_path[separator:]

    return normal_path_train, normal_path_test, anomaly_path


def get_gta_globs():
    from glob import glob
    nums = [f'0{i}' for i in range(1, 10)] + ['10']
    globs_id = []
    globs_ood = []
    for i in range(10):
        id_path = f'/kaggle/input/gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/ID/*'
        ood_path = f'/kaggle/input/gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/OOD/*'
        globs_id.append(glob(id_path))
        globs_ood.append(glob(ood_path))
        print(i, len(globs_id[-1]), len(globs_ood[-1]))

    glob_id = []
    glob_ood = []
    for i in range(len(globs_id)):
        glob_id += globs_id[i]
        glob_ood += globs_ood[i]

    random.seed(42)
    random.shuffle(glob_id)
    train_ratio = 0.7
    separator = int(train_ratio * len(glob_id))
    glob_train_id = glob_id[:separator]
    glob_test_id = glob_id[separator:]

    return glob_train_id, glob_test_id, glob_ood


def train(_class_, epochs=200, image_size=224):
    print(_class_)
    learning_rate = 0.005
    batch_size = 16
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    trans_norm = transforms.Normalize(mean=mean_train,
                             std=std_train)
    _, transform = get_data_transforms(image_size, image_size)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # data_transform, gt_transform = get_data_transforms(image_size, image_size)
    if _class_ == 'brain':
        prepare_br35h_dataset_files()
        prepare_brats2015_dataset_files()
        train_data = BrainTrain(transform=transform)
        test_data1 = BrainTest(transform=transform, test_id=1)
        test_data2 = BrainTest(transform=transform, test_id=2)
    elif _class_ == 'gta':
        normal_path_train, normal_path_test, anomaly_path = get_cityscape_globs()
        test_path = normal_path_test + anomaly_path
        test_label = [0] * len(normal_path_test) + [1] * len(anomaly_path)
        train_label = [0] * len(normal_path_train)
        glob_train_id, glob_test_id, glob_ood = get_gta_globs()
        train_data = GTA(image_path=normal_path_train, labels=train_label,
                        transform=transform)
        test_data1 = GTA_Test(image_path=test_path, labels=test_label,
                            transform=transform)
        test_data2 = GTA_Test(image_path=glob_test_id + glob_ood,
                            labels=[0] * len(glob_test_id) + [1] * len(glob_ood),
                            transform=transform)
    elif _class_ == 'waterbirds':


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader1 = torch.utils.data.DataLoader(test_data1, batch_size=1, shuffle=False)
    test_dataloader2 = torch.utils.data.DataLoader(test_data2, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))


    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))#bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        if (epoch + 1) % 25 == 0:
            print('Main:')
            auroc_sp = evaluation(encoder, bn, decoder, test_dataloader1, device)
            print('Sample Auroc{:.3f}'.format(auroc_sp))
            # torch.save({'bn': bn.state_dict(),
            #             'decoder': decoder.state_dict()}, ckp_path)
            print('=' * 30)
        if (epoch + 1) % 25 == 0:
            print('Shifted:')
            auroc_sp = evaluation(encoder, bn, decoder, test_dataloader2, device)
            print('Sample Auroc{:.3f}'.format(auroc_sp))
            # torch.save({'bn': bn.state_dict(),
            #             'decoder': decoder.state_dict()}, ckp_path)
            print('=' * 30)
    return auroc_sp




if __name__ == '__main__':

    setup_seed(111)

    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='brain')
    parser.add_argument('--total_iters', type=int, default=200)
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()
    train(args.dataset, args.total_iters, args.image_size)

