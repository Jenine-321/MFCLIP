import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score,auc,roc_curve
import matplotlib.pyplot as plt
import pickle
import time
from functools import partial
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score,auc,roc_curve
import os
from PIL import Image
from torch.utils.data import TensorDataset
from Infer_MFCLIP_Load import tokenize
from Infer_MFCLIP_Load import load
import cv2
from load_pretrain_weight import load_pretrained_model_v2

def cross_generator(dir, diffusion_selection):


    count_real = 0
    count_fake_lat = 0
    count_fake_coll = 0
    count_fake_diffae = 0
    count_fake_difface = 0
    count_real_difface = 0
    count_fake_ddpm = 0
    count_real_diffae = 0
    # 0 real 1 fake
    celeba_hq = []
    celeba_hq_lables = []
    fake_path_lat = []
    fake_labels_lat = []
    fake_path_ddpm = []
    fake_labels_ddpm = []
    fake_path_coll = []
    fake_labels_coll = []
    fake_path_diffface = []
    fake_labels_diffface = []
    fake_path_diffae = []
    fake_labels_diffae = []

    if diffusion_selection == 'diffface':

        for cat in os.listdir(dir):
            if cat == "real":
                label = 0
                for img in os.listdir(dir + '/' + cat + "/" + "CelebA_HQ"):
                    celeba_hq.append(dir + '/' + cat + '/' + "CelebA_HQ" + '/' + img)
                    celeba_hq_lables.append(label)
                    count_real_difface  += 1
                    if count_real_difface == 28:
                        break

            if cat == "fake":
                label = 1
                for img in os.listdir(dir + '/' + cat + '/FS/Diffusion'):
                    fake_path_diffface.append(dir + '/' + cat + '/FS/Diffusion/' + img)
                    fake_labels_diffface.append(label)
                    count_fake_difface +=1
                    if count_fake_difface == 28:
                        break

        return fake_path_diffface, celeba_hq, fake_labels_diffface, celeba_hq_lables

    if diffusion_selection == 'diffae':

        for cat in os.listdir(dir):
            if cat == "real":
                label = 0
                for img in os.listdir(dir + '/' + cat + "/" + "FFHQ"):
                    celeba_hq.append(dir + '/' + cat + '/' + "FFHQ" + '/' + img)
                    celeba_hq_lables.append(label)
                    count_real_diffae += 1
                    if count_real_diffae == 28: #2824
                        break

            if cat == "fake":
                label = 1
                for img in os.listdir(dir + '/' + cat + '/AM/Diffusion'):
                    fake_path_diffae.append(dir + '/' + cat + '/AM/Diffusion/' + img)
                    fake_labels_diffae.append(label)
                    count_fake_diffae += 1
                    if count_fake_diffae == 28:
                        break

        return fake_path_diffae, celeba_hq, fake_labels_diffae, celeba_hq_lables

    for cat in os.listdir(dir):

        if cat == "real":
            label = 0
            for img in os.listdir(dir + '/' + cat + "/" + "CelebA_HQ"):
                celeba_hq.append(dir + '/' + cat + '/' + "CelebA_HQ" + '/' + img)
                celeba_hq_lables.append(label)
                count_real +=1
                if count_real==28:
                    break

        if cat == "fake":
            label = 1
            for img in os.listdir(dir + '/' + cat + '/EFS/Diffusion'):
                diffusion_namelist = img.split('_')
                diff_name = diffusion_namelist[0]
                #diff_ddpm = diffusion_namelist[-1][:-4]
                if diff_name=='latentdiffusion':

                    fake_path_lat.append(dir + '/' + cat + '/EFS/Diffusion/' + img)
                    fake_labels_lat.append(label)
                    # count_fake_lat += 1
                    # if count_fake_lat == 4:
                    #     break
                if diff_name == 'colladiffusion':
                    fake_path_coll.append(dir + '/' + cat + '/EFS/Diffusion/' + img)
                    fake_labels_coll.append(label)
                    count_fake_coll += 1
                    if count_fake_coll == 28:
                        break
                if diff_name == 'ddpm':
                    fake_path_ddpm.append(dir + '/' + cat + '/EFS/Diffusion/' + img)
                    fake_labels_ddpm.append(label)
                    # count_fake_ddpm += 1
                    # if count_fake_ddpm == 4:
                    #     break

    if diffusion_selection == 'latentdiffusion':
        return fake_path_lat, celeba_hq, fake_labels_lat, celeba_hq_lables
    if diffusion_selection == 'colladiffusion':
        return fake_path_coll, celeba_hq, fake_labels_coll, celeba_hq_lables
    if diffusion_selection == 'ddpm':
        return fake_path_ddpm, celeba_hq, fake_labels_ddpm, celeba_hq_lables

def my_dataset(dataset_images_path, imagesize):

    datasets = []
    mean = [0.4718, 0.3467, 0.3154]
    std = [0.1656, 0.1432, 0.1364]

    transform = transforms.Compose([
        transforms.Resize((imagesize, imagesize)),
        #Aug(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    for img_path in dataset_images_path:
        image_rgb = cv2.imread(img_path)  # cv2.IMREAD_GRAYSCALE
        image_rgb = cv2.resize(image_rgb, (imagesize, imagesize))
        datasets.append(image_rgb)

    return datasets

def MyDataloder(dataset_list, dataset_label_list, batch_size):

    dataset = TensorDataset(
        torch.tensor(dataset_list).float(),
        torch.tensor(dataset_label_list)
    )
    dataloader = DataLoader(
        dataset,  # The training samples.
        shuffle=False,
        batch_size=batch_size  # Trains with this batch size.
    )
    return dataloader

def test (dir,dataset):

    test_fake_path, test_celeba_hq, test_fake_labels, test_celeba_hq_lables = cross_generator(dir,dataset)
    test_all_datasets = test_fake_path + test_celeba_hq
    test_all_labels = test_fake_labels + test_celeba_hq_lables
    print(dataset,len(test_fake_path))
    print(dataset+'real', len(test_celeba_hq))

    test_dataset = my_dataset(test_all_datasets, imagesize=224)
    print("Finished test dataset loaded! test dataset length:", len(test_dataset))

    test_dataloader = MyDataloder(test_dataset, test_all_labels, batch_size=24)
    print("Finished constructing test dataloader!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_weight_path = './checkpoints/colla_train_mfclip/best.pth'
    model = load('ViT-B/32')
    model = model.to(torch.float32)

    model = load_pretrained_model_v2(model, pretrain_checkpoint=load_weight_path)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)  # For multi-gpu

    model.to(device)
    print("Finished load best weights.")

    model.eval()
    #since = time.time()
    m = nn.Softmax(dim=1)
    Sum = 0  # Count the number of samples predicted correctly by model

    y_real_label = []  # Record  real label of test samples
    y_score = []  # Record  test samples scores predicted by model

    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)

        output_score = m(output).to(device)  # Adjust scores predicted by model into 0-1
        output_scores = output_score.detach().cpu().numpy()
        y_scores = output_scores[:, 1]  # Get real class probabilities predicted by model
        for i in y_scores:
            y_score.append(i)
        _, prediction = torch.max(output, 1)
        pred_label = prediction.detach().cpu().numpy()
        main_label = labels.detach().cpu().numpy()
        for i in main_label:
            y_real_label.append(i)

        bool_list = list(
            map(lambda x, y: x == y, pred_label, main_label))
        Sum += sum(np.array(
            bool_list) * 1)
        dataset_name = dataset

    print(dataset_name  + ' Prediction Acc: ', (Sum / len(test_dataset)) * 100, '%')

    return y_real_label, y_score

def test_AUC(y_real_label,y_score,dataset):
    y_test = np.array(y_real_label)
    y_pros = np.array(y_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pros, pos_label=1)
    roc_auc = auc(fpr, tpr)
    dataset_name = dataset
    print(dataset_name + ' AUC:%0.5f' % roc_auc)

if __name__ == '__main__':
    # ['Celeb-DF/test','DFDC/test','FF++C23/test_mix','colladiffusion','latentdiffusion', 'ddpm','diffae','diffface''DFD/test' 'FF++C23/test_mix','Celeb-DF/test','DFDC/test', 'FF++C23/test_mix', 'FF++C23/test/DF', 'FF++C23/test/F2F','FF++C23/test/FSW', 'FF++C23/test/NT', 'DeeperForensics/test', 'DFD/test'，'FF++C23/test/DF', 'FF++C23/test/F2F','FF++C23/test/FSW', 'FF++C23/test/NT', ]
    dir = '/home/zyn/disk1/dfdcProtocol/test'
    dataset_names = ['colladiffusion', 'ddpm','diffface','diffae']
    for dataset_name in dataset_names:
        y_real_label, y_score = test(dir,dataset_name)
        test_AUC(y_real_label, y_score, dataset_name)




