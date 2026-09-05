import os
cpu_num =1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler
from pytorchtools import EarlyStopping
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from pyecharts.charts import Line
# from pyecharts import options as opts
import copy
import time
import pickle
from torch.nn import functional as F
import argparse
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
import torch
import yaml
import cv2
from MFCLIP_Load import tokenize
from MFCLIP_Load import load
# from augmentation import Aug
# from thop import profile
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--dataset", type=str, default='/data1/zhangyaning/DeepfakeDataset/FF++C23',
                        # Dataset/RAF-DB_fixpart_class/basic',
                        help="The root directory of dataset")
    parser.add_argument("-c", "--checkpoints", type=str, default='./checkpoints/colla_train_mfclip',
                        help="The checkpoints of network")
    parser.add_argument("-j", "--loss_folder", type=str, default='./checkpoints/diffusion_train_cvit.html',
                        help='the folder of saving the loss values')
    parser.add_argument("-v", "--load_weight", type=bool, default=False,
                        help='whether load weight or not')
    parser.add_argument("-q", "--load_weight_path", type=str,
                        default='./checkpoint/best.pth',
                        help='the pre-checpoint path')
    parser.add_argument("-b", "--batch_size", type=int, default=24,
                        help='the size of every batch')
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help='the epochs number of train')
    parser.add_argument("-l", "--learning_rate", type=int, default=1e-4,
                        help='the learning_rate of train')
    parser.add_argument("-y", "--weight_decay", type=int, default=1e-3,
                        help='the weight_decay of train')
    parser.add_argument("-i", "--image_size", type=int, default=224,
                        help='The size of the picture entering the network')
    parser.add_argument("-k", "--patch_size", type=int, default=3,
                        help='Kernel size for conv layer for feature extraction.')
    parser.add_argument("-s", "--stride", type=int, default=1,
                        help='Stride size for conv layer for feature extraction.')
    parser.add_argument("-d", "--base_dims", type=list, default=[128, 128, 128],
                        help='Dimensions of each attention head at each stage of the transformer.')
    parser.add_argument("-w", "--depth", type=list, default=[10, 10, 12],
                        help='The number of transformer_blocks in each stage of transformer.')
    parser.add_argument("-z", "--heads", type=list, default=[4, 8, 16],
                        help='Number of attention heads at each stage of the transformer.')
    parser.add_argument("-m", "--mlp_ratio", type=int, default=4,
                        help='The FeedForward layer expands the number of neurons in the input layer by times')
    parser.add_argument("-n", "--num_classes", type=int, default=2,
                        help='The number of detection categories.')
    parser.add_argument("-f", "--in_chans", type=int, default=512,
                        help='The num of channels for input of this network.')
    parser.add_argument("-o", "--attn_drop", type=int, default=0,
                        help='dropout attention map to prevent overfitting.')
    parser.add_argument("-p", "--proj_drop", type=int, default=0,
                        help='dropout fully connected layer to prevent overfitting.')

    args = parser.parse_args()

    return args

def loss_fn_kd(outputs, labels, teacher_outputs, logits,KD_T=20, KD_alpha=0.5):
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/KD_T,dim=1),
                             F.softmax(teacher_outputs/KD_T,dim=1) * KD_alpha*KD_T*KD_T) +\
        F.cross_entropy(logits, labels) * (1. - KD_alpha)
    return KD_loss
# def my_loss(similarity_matrix):


def cross_generator(dir, diffusion_selection,mode):

    count_real = 0
    count_fake = 0
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
    text_real_list = []
    text_fake_diffface = []
    text_fake_diffae = []
    text_fake_lat = []
    text_fake_coll = []
    text_fake_ddpm = []
    text_fake_level2_lat = []
    text_fake_level2_coll = []
    text_fake_level2_ddpm = []
    text_fake_level2_diffface = []
    text_fake_level2_diffae = []
    text_fake_level3_lat = []
    text_fake_level3_coll = []
    text_fake_level3_ddpm = []
    text_fake_level3_diffface = []
    text_fake_level3_diffae = []
    text_fake_level4_lat = []
    text_fake_level4_coll = []
    text_fake_level4_ddpm = []
    text_fake_level4_diffface = []
    text_fake_level4_diffae = []
    text_fake_FSLSD= []
    fake_path_FSLSD = []
    fake_labels_FSLSD = []
    text_fake_level2_FSLSD= []
    text_fake_level3_FSLSD= []
    text_fake_level4_FSLSD= []
    text_fake_FaceSwapper= []
    fake_path_FaceSwapper = []
    fake_labels_FaceSwapper = []
    text_fake_level2_FaceSwapper= []
    text_fake_level3_FaceSwapper= []
    text_fake_level4_FaceSwapper= []


    if diffusion_selection == 'diffface':

        for cat in os.listdir(dir):
            if cat == "real":
                label = 0
                text_real = 'A photo of a real face'
                for img in os.listdir(dir + '/' + cat + "/" + "CelebA_HQ"):
                    celeba_hq.append(dir + '/' + cat + '/' + "CelebA_HQ" + '/' + img)
                    celeba_hq_lables.append(label)
                    text_real_list.append(text_real)
                    count_real += 1
                    # if count_real == 24: #and mode == 'train':
                    #    break
                    if count_real == 24 and mode=='train': #22008
                        break
                    if count_real == 24 and mode == 'val': #2976
                        break

            if cat == "fake":
                label = 1
                text_fake = 'A photo of a fake face'
                text_fake_level2 = 'a photo of an identity swapped face'
                text_fake_level3 = 'a photo generated by the diffusion-based model'
                text_fake_level4 = 'the source generative model of this photo is DiffFace'
                for img in os.listdir(dir + '/' + cat + '/FS/Diffusion'):
                    fake_path_diffface.append(dir + '/' + cat + '/FS/Diffusion/' + img)
                    fake_labels_diffface.append(label)
                    text_fake_diffface.append(text_fake)
                    text_fake_level2_diffface.append(text_fake_level2)
                    text_fake_level3_diffface.append(text_fake_level3)
                    text_fake_level4_diffface.append(text_fake_level4)
                    count_fake += 1
                    # if count_fake == 24:  # and mode == 'train':
                    #     break
                    if count_fake == 24 and mode == 'train': #22008
                        break
                    if count_fake == 24 and mode == 'val': # 2976
                        break
        return fake_path_diffface, celeba_hq, fake_labels_diffface, celeba_hq_lables, text_fake_diffface, text_fake_level2_diffface,text_fake_level3_diffface,text_fake_level4_diffface,text_real_list

    if diffusion_selection == 'diffae':

        for cat in os.listdir(dir):
            if cat == "real":
                label = 0
                text_real = 'a photo of a real face'
                for img in os.listdir(dir + '/' + cat + "/" + "CelebA_HQ"):
                    celeba_hq.append(dir + '/' + cat + '/' + "CelebA_HQ" + '/' + img)
                    celeba_hq_lables.append(label)
                    text_real_list.append(text_real)

                    # count_real += 1
                    # if count_real == 24180 and mode == 'train':  # 24180
                    #     break
                    # if count_real == 2988 and mode == 'val':  # 2988
                    #     break

            if cat == "fake":
                label = 1
                text_fake = 'a photo of a fake face'
                text_fake_level2 = 'a photo of an attribute manipulated face'
                text_fake_level3 = 'a photo generated by the diffusion-based model'
                text_fake_level4 = 'the source generative model of this photo is Diffae'

                for img in os.listdir(dir + '/' + cat + '/AM/Diffusion'):
                    fake_path_diffae.append(dir + '/' + cat + '/AM/Diffusion/' + img)
                    fake_labels_diffae.append(label)
                    text_fake_diffae.append(text_fake)
                    text_fake_level2_diffae.append(text_fake_level2)
                    text_fake_level3_diffae.append(text_fake_level3)
                    text_fake_level4_diffae.append(text_fake_level4)
                    # count_fake+= 1
                    # if count_fake == 24180 and mode == 'train':  # 24180
                    #     break
                    # if count_fake == 2988 and mode == 'val':  # 2988
                    #     break
                    # count_fake += 1
                    # if count_fake == 4:
                    #     break
        return fake_path_diffae, celeba_hq, fake_labels_diffae, celeba_hq_lables, text_fake_diffae, text_fake_level2_diffae,text_fake_level3_diffae,text_fake_level4_diffae, text_real_list

    for cat in os.listdir(dir):

        if cat == "real":
            label = 0
            text_real = 'a photo of a real face'
            for img in os.listdir(dir + '/' + cat + "/" + "CelebA_HQ"):
                celeba_hq.append(dir + '/' + cat + '/' + "CelebA_HQ" + '/' + img)
                celeba_hq_lables.append(label)
                text_real_list.append(text_real)
                count_real += 1
                # if count_real == 24:
                   # break
                if count_real == 24168 and mode == 'train':  # 12072 24168
                    break
                if count_real == 2976 and mode == 'val':  # 1488 2976
                    break
                # count_fake+= 1
                # if count_real == 24180 and mode == 'train':  # 24180
                #     break
                # if count_real == 2988 and mode == 'val':  # 2988
                #     break

        if cat == "fake":
            label = 1
            text_fake = 'a photo of a fake face'
            text_fake_level2 = 'a photo of an entire synthesized face'
            text_fake_level3 = 'a photo generated by the diffusion-based model'
            text_fake_level22 = 'a photo of an identity swapped face'
            text_fake_level33 = 'a photo generated by the GAN-based model'

            for img in os.listdir(dir + '/' + cat + '/EFS/Diffusion'):
                diffusion_namelist = img.split('_')
                diff_name = diffusion_namelist[0]

                #diff_ddpm = diffusion_namelist[-1][:-4]
                if diff_name =='latentdiffusion':
                    text_fake_level4 = 'the source generative model of this photo is LatentDiffusion'
                   # print(diff_name)
                    fake_path_lat.append(dir + '/' + cat + '/EFS/Diffusion/' + img)
                    fake_labels_lat.append(label)
                    text_fake_lat.append(text_fake)
                    text_fake_level2_lat.append(text_fake_level2)
                    text_fake_level3_lat.append(text_fake_level3)
                    text_fake_level4_lat.append(text_fake_level4)
                    # count_fake += 1
                    # if count_fake == 24and mode == 'train':  # 24180
                    #     break
                    # if count_fake == 24 and mode == 'val':  # 2988
                    #     break
                    # count_fake += 1
                    # if count_fake == 10:
                    #     break
                if diff_name == 'colladiffusion':
                    text_fake_level4 = 'the source generative model of this photo is CollaborativeDiffusion'
                    fake_path_coll.append(dir + '/' + cat + '/EFS/Diffusion/' + img)
                    fake_labels_coll.append(label)
                    text_fake_coll.append(text_fake)
                    text_fake_level2_coll.append(text_fake_level2)
                    text_fake_level3_coll.append(text_fake_level3)
                    text_fake_level4_coll.append(text_fake_level4)
                    count_fake += 1
                    if count_fake == 24168 and mode == 'train':  # 24168
                        break
                    if count_fake == 2976 and mode == 'val':  # 2976
                        break
                
                if diff_name == 'ddpm':
                    text_fake_level4 = 'the source generative model of this photo is DDPM'
                    fake_path_ddpm.append(dir + '/' + cat + '/EFS/Diffusion/' + img)
                    fake_labels_ddpm.append(label)
                    text_fake_ddpm.append(text_fake)
                    text_fake_level2_ddpm.append(text_fake_level2)
                    text_fake_level3_ddpm.append(text_fake_level3)
                    text_fake_level4_ddpm.append(text_fake_level4)
                    # count_fake += 1
                    # # if count_fake == 24 and mode == 'train':  # 24180
                    # #     break
                    # # if count_fake == 24 and mode == 'val':  # 2988
                    # #     break
                    # # count_fake += 1
                    # # count_fake += 1
                    # # # if count_fake == 24:
                    # # #     break
                    # if count_fake == 24180 and mode == 'train':  # 24180
                    #     break
                    # if count_fake == 2988 and mode == 'val':  # 2988
                    #     break
            for img_FS in os.listdir(dir + '/' + cat + '/FS/GAN'):
                gan_namelist = img_FS.split('_')
                gan_name = gan_namelist[-1][:-4]
                if gan_name == 'FSLSD':


                    text_fake_level4 = 'the source generative model of this photo is FSLSD'
                    fake_path_FSLSD.append(dir + '/' + cat + '/FS/GAN/' + img_FS)
                    fake_labels_FSLSD.append(label)
                    text_fake_FSLSD.append(text_fake)
                    text_fake_level2_FSLSD.append(text_fake_level22)
                    text_fake_level3_FSLSD.append(text_fake_level33)
                    text_fake_level4_FSLSD.append(text_fake_level4)
                    # count_fake += 1
                    # if count_fake == 12072 and mode == 'train':  # 12072
                    #     break
                    # if count_fake == 1488 and mode == 'val':  # 1488
                    #     break
                if gan_name == 'FaceSwapper':

                    text_fake_level4 = 'the source generative model of this photo is FaceSwapper'
                    fake_path_FaceSwapper.append(dir + '/' + cat + '/FS/GAN/' + img_FS)
                    fake_labels_FaceSwapper.append(label)
                    text_fake_FaceSwapper.append(text_fake)
                    text_fake_level2_FaceSwapper.append(text_fake_level22)
                    text_fake_level3_FaceSwapper.append(text_fake_level33)
                    text_fake_level4_FaceSwapper.append(text_fake_level4)
                    # count_fake += 1
                    # if count_fake == 12072 and mode == 'train':  # 12072
                    #     break
                    # if count_fake == 1488 and mode == 'val':  # 1488
                    #     break
    if diffusion_selection == 'latentdiffusion':
        return fake_path_lat, celeba_hq, fake_labels_lat, celeba_hq_lables, text_fake_lat, text_fake_level2_lat, text_fake_level3_lat, text_fake_level4_lat, text_real_list
    if diffusion_selection == 'colladiffusion':
        return fake_path_coll, celeba_hq, fake_labels_coll, celeba_hq_lables, text_fake_coll, text_fake_level2_coll, text_fake_level3_coll, text_fake_level4_coll, text_real_list
    if diffusion_selection == 'ddpm':
        return fake_path_ddpm, celeba_hq, fake_labels_ddpm, celeba_hq_lables, text_fake_ddpm, text_fake_level2_ddpm, text_fake_level3_ddpm, text_fake_level4_ddpm, text_real_list
    if diffusion_selection == 'FSLSD':
        return fake_path_FSLSD, celeba_hq, fake_labels_FSLSD, celeba_hq_lables, text_fake_FSLSD, text_fake_level2_FSLSD, text_fake_level3_FSLSD, text_fake_level4_FSLSD, text_real_list
    if diffusion_selection == 'FaceSwapper':
        return fake_path_FaceSwapper, celeba_hq, fake_labels_FaceSwapper, celeba_hq_lables, text_fake_FaceSwapper, text_fake_level2_FaceSwapper, text_fake_level3_FaceSwapper, text_fake_level4_FaceSwapper, text_real_list


def my_dataset(dataset_images_path, texts, texts_level2,texts_level3,texts_level4,imagesize):

    datasets = []
    text_tokenize = []
    text_level2_tokenize = []
    text_level3_tokenize = []
    text_level4_tokenize = []
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
    for text in texts:

        tokenized = tokenize(text)  # 1,77
        #print("tokenized.shape",tokenized.shape)
        text_tokenize.append(tokenized.numpy())

    for text_level2 in texts_level2:
        tokenized_level2 = tokenize(text_level2)  # B,77
        #print(text_level2, tokenized_level2)
        text_level2_tokenize.append(tokenized_level2.numpy())
    for text_level3 in texts_level3:
        tokenized_level3 = tokenize(text_level3)  # B,77
        text_level3_tokenize.append(tokenized_level3.numpy())
       # print(text_level3, tokenized_level3)
    for text_level4 in texts_level4:
        tokenized_level4 = tokenize(text_level4)  # B,77
        text_level4_tokenize.append(tokenized_level4.numpy())
        #print(text_level4, tokenized_level4)

    return datasets, text_tokenize,text_level2_tokenize,text_level3_tokenize,text_level4_tokenize

def MyDataloder(dataset_list, text_list,text_level2_list,text_level3_list, text_level4_list,dataset_label_list, batch_size):

    dataset = TensorDataset(
        torch.tensor(dataset_list).float(),
        torch.tensor(text_list).squeeze(dim=1).cuda(),
        torch.tensor(text_level2_list).squeeze(dim=1).cuda(),
        torch.tensor(text_level3_list).squeeze(dim=1).cuda(),
        torch.tensor(text_level4_list).squeeze(dim=1).cuda(),
        torch.tensor(dataset_label_list)
    )
    dataloader = DataLoader(
        dataset,  # The training samples.
        shuffle=True,
        batch_size=batch_size  # Trains with this batch size.
    )

    return dataloader



# 转换



def train(model, train_dataloader, criterion, optimizer, epoch, epochs, device, batch_size, train_dataset,
          scheduler, train_loss, train_accu):
    m = nn.Softmax(dim=1)
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)

    running_train_loss = 0.0  # Total train loss value
    running_train_corrects = 0  # Total train acc value
    train_phase_idx = 0  # Inputs interval

    # Train
    model.train()
    #model_teacher.eval()
    #model_teacher = model_teacher.cuda()

    for inputs, texts, texts_level2,texts_level3,texts_level4,labels in train_dataloader:
        inputs = inputs.to(device)
        texts = texts.to(device)  # b，77
        #print(texts.shape)
        texts_level2 = texts_level2.to(device)  # b，77
        texts_level3 = texts_level3.to(device)
        texts_level4 = texts_level4.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  # Gradient zeros
        with torch.set_grad_enabled(True):  # Sets gradient calculation to on
            logits_per_image, logits_per_text, text_features, text_predicts, logits= model(inputs,texts,texts_level2,texts_level3,texts_level4)

            # 打印结果
            # print(f"FLOPs: {flops_g:.2f} GFLOPs")
            # print(f"参数量: {params_m:.2f} M")

            batch_size = inputs.shape[0]
            indexs = torch.arange(batch_size, device=device).long()
            loss =  (
                                 criterion(logits_per_image, indexs) +
                                 criterion(logits_per_text, indexs)
                          ) / 2 + loss_fn_kd(text_predicts, labels, text_features,logits) + criterion(logits, labels)
            loss.backward()  # Gradient calculation
            optimizer.step()  # Parameter update

        if train_phase_idx % 100 == 0:  # Per 100 inputs interval prints loss value
            print('Train loss:', train_phase_idx, ':', loss.item())  # Inputs average loss value
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, train_phase_idx * batch_size, train_dataset,
                       100. * train_phase_idx * batch_size / train_dataset,
                loss.item()))
        train_phase_idx += 1

        # Statistics
        running_train_loss += loss.item() * inputs.size(0)
       # running_train_corrects += torch.sum(preds == labels.data)

    scheduler.step()  # Adjust learn rate

    epoch_loss = running_train_loss / train_dataset  # Average train loss value
    #epoch_acc = running_train_corrects.double() / train_dataset  # Average train acc value
    epoch_loss_round = round(epoch_loss, 4)
    train_loss.append(epoch_loss_round)  # For draw loss lines
    #train_accu.append(epoch_acc)  # For draw acc lines
    print('Train Loss: {:.4f}'.format(epoch_loss))
    #print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


def val(model, epoch, val_dataloader, device, optimizer, criterion, batch_size, val_dataset, val_loss, val_accu,
        epoch_record, time_elapsed_val, min_loss, best_model_wts):
    running_val_loss = 0.0  # Total val loss value
    running_val_corrects = 0  # Total val acc value
    val_phase_idx = 0
    m = nn.Softmax(dim=1)
    # Validation
    model.eval()
    since_val = time.time()
    for inputs_val, texts_val,texts_val_level2,texts_val_level3,texts_val_level4,labels_val in val_dataloader:
        inputs_val = inputs_val.to(device)
        texts_val = texts_val.to(device)
        texts_val_level2 = texts_val_level2.to(device)
        texts_val_level3 = texts_val_level3.to(device)
        texts_val_level4 = texts_val_level4.to(device)
        labels_val = labels_val.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):  # Sets gradient calculation to off
            logits_per_image_val, logits_per_text_val, text_features_val, text_predicts_val, logits_val = model(inputs_val, texts_val,texts_val_level2,texts_val_level3,texts_val_level4)

            # logit_scale = logit_scale.mean()

            batch_size_val = inputs_val.shape[0]
            indexs_val = torch.arange(batch_size_val, device=device).long()
            loss_val =   (
                    criterion(logits_per_image_val, indexs_val) +
                    criterion(logits_per_text_val, indexs_val)
             ) / 2 + loss_fn_kd(text_predicts_val, labels_val, text_features_val,logits_val) + criterion(logits_val, labels_val)
     
        if val_phase_idx % 100 == 0:  # Per 100 inputs interval prints loss value
            print('Validation loss:', val_phase_idx, ':', loss_val.item())
            print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, val_phase_idx * batch_size, val_dataset,
                       100. * val_phase_idx * batch_size / val_dataset,
                loss_val.item()))  # Present epoch、present val samples/total val samples
        val_phase_idx += 1
        # statistics
        running_val_loss += loss_val.item() * inputs_val.size(0)
        #running_val_corrects += torch.sum(preds_val == labels_val.data)

    epoch_val_loss = running_val_loss / val_dataset
    #epoch_val_acc = running_val_corrects.double() / val_dataset
    epoch_val_loss_round = round(epoch_val_loss, 4)
    #epoch_val_acc_round = round(epoch_val_acc.item(), 4)
    val_loss.append(epoch_val_loss_round)
    #val_accu.append(epoch_val_acc_round)
    epoch_record.append(epoch)
    time_elapsed_val += time.time() - since_val
    print('Validation Loss: {:.5f}'.format(epoch_val_loss))
   # print('Validation Loss: {:.5f} Acc: {:.5f}'.format(epoch_val_loss, epoch_val_acc))

    if epoch_val_loss < min_loss:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_val_loss, min_loss))
        min_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    if not os.path.isdir(args.checkpoints):
        os.mkdir(args.checkpoints)

    state = {
        'state_dict': model.state_dict(),
        'epoch': epoch
    }
    torch.save(state, args.checkpoints + '/current_epoch.pth')  # Epoch checkpoints Path name
    print('Total val  complete in {:.0f}m {:.0f}s'.format(time_elapsed_val // 60, time_elapsed_val % 60))
    return best_model_wts, min_loss, epoch_val_loss


def save(train_loss, train_accu, val_loss, val_accu, best_model_wts, min_loss, time_elapsed, time_elapsed_val):

    state = {'state_dict': best_model_wts,
             'min_loss': min_loss,
             'total train and val time': time_elapsed // 60,
             'total val time': time_elapsed_val // 60}
    torch.save(state, args.checkpoints + '/best.pth')


def visual(epoch_record, train_loss, val_loss, val_accu):
    line1 = (
        Line()
            .add_xaxis(epoch_record)
            .add_yaxis(
            "train_loss",
            train_loss,
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='min')])
        )
            .add_yaxis(
            "val_loss",
            val_loss,
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='min')])
        )

            .add_yaxis(
            "accuracy",
            val_accu,
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max')], symbol_size=70)
        )

            .set_global_opts(title_opts=opts.TitleOpts(title="Loss-curve")
                             , xaxis_opts=opts.AxisOpts(name='Epoch'
                                                        , name_location='middle'  # 坐标轴名字所在的位置
                                                        , name_gap=25  # 坐标轴名字与坐标轴之间的距离)
                                                        )
                             , yaxis_opts=opts.AxisOpts(name='Loss'
                                                        , name_location='middle'  # 坐标轴名字所在的位置
                                                        , name_gap=25  # 坐标轴名字与坐标轴之间的距离
                                                        )
                             )
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
   # line1.render(args.loss_folder)


def main():
    start_time = time.time()
    dir_train = "/home/zyn/disk1/dfdcProtocol/train"
    dir_val = "/home/zyn/disk1/dfdcProtocol/val"

    train_diffusion_path, train_celeba_hq, train_diffusion_labels, train_celeba_hq_lables, train_fake_text,train_fake_level2_text,train_fake_level3_text,train_fake_level4_text,train_real_text  = cross_generator(dir_train, diffusion_selection='diffface',mode='train')
    train_all_datasets = train_diffusion_path + train_celeba_hq
    train_all_texts = train_fake_text + train_real_text
    train_all_level2_texts = train_fake_level2_text + train_real_text
    train_all_level3_texts = train_fake_level3_text + train_real_text
    train_all_level4_texts = train_fake_level4_text + train_real_text
    train_all_labels = train_diffusion_labels + train_celeba_hq_lables

    val_diffusion_path, val_celeba_hq, val_diffusion_labels, val_celeba_hq_lables,val_fake_text, val_fake_level2_text,val_fake_level3_text,val_fake_level4_text,val_real_text = cross_generator(dir_val, diffusion_selection='diffface',mode='val')
    val_all_datasets = val_diffusion_path + val_celeba_hq
    val_all_texts = val_fake_text + val_real_text
    val_all_levle2_texts = val_fake_level2_text + val_real_text
    val_all_levle3_texts = val_fake_level3_text + val_real_text
    val_all_levle4_texts = val_fake_level4_text + val_real_text
    val_all_labels = val_diffusion_labels + val_celeba_hq_lables

    print("train difusion length:", len(train_diffusion_path))
    print("train celeb length:", len(train_celeba_hq))
    print("train real text length:", len(train_real_text))
    print("train fake text length:", len(train_fake_text))
    print("val difusion length:", len(val_diffusion_path))
    print("val celeb length:", len(val_celeba_hq))
    print("val real text length:", len(val_real_text))
    print("val fake text length:", len(val_fake_text))

    train_datasets, train_texts, train_level2_texts,train_level3_texts,train_level4_texts = my_dataset(train_all_datasets, train_all_texts,train_all_level2_texts,train_all_level3_texts, train_all_level4_texts,imagesize=224)
    print("train dataset length:", len(train_datasets))
    val_datasets, val_texts, val_level2_texts,val_level3_texts,val_level4_texts = my_dataset(val_all_datasets,val_all_texts,val_all_levle2_texts,val_all_levle3_texts,val_all_levle4_texts,imagesize=224)
    print(" val dataset length:", len(val_datasets))
    end_time = time.time()
    elapsed_time1 = end_time - start_time
    print("read datasets time: {:.2f} seconds.".format(elapsed_time1))

    train_dataset = len(train_datasets)
    val_dataset = len(val_datasets)

    train_dataloader = MyDataloder(train_datasets, train_texts,train_level2_texts,train_level3_texts,train_level4_texts,train_all_labels,batch_size=args.batch_size)
    print("Finished train dataset loaded!")
    val_dataloader = MyDataloder(val_datasets,val_texts,val_level2_texts,val_level3_texts,val_level4_texts,val_all_labels, batch_size=args.batch_size)
    print("Finished val dataset loaded!")
    end_time2 = time.time()
    elapsed_time = end_time2 - start_time
    print("Datasets load time: {:.2f} seconds.".format(elapsed_time))
    # train_dataloader, val_dataloader, train_dataset, val_dataset = MyDataload(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load(name='ViT-B/32')
    model = model.to(torch.float32)

    # Load  model weights
    if args.load_weight:
        state = torch.load(args.load_weight_path)
        checkpoint = state['state_dict']  # ['state_dict']
        model.load_state_dict(checkpoint)
        print(" pretrained model Loaded..")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)  # For multi-gpu
    model.to(device)
    print(" model Loaded..")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    patience = 5
    early_stopping = EarlyStopping(patience, verbose=True)
    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []
    epoch_record = []
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())  # For obtaining the best model checkpoints
    time_elapsed_val = 0  # Sum val time
    min_loss = 10000  # Set the minimum val loss
    n = 0  # Count epoch numbers to save epoch weights

    for epoch in range(args.epochs):
        train(model,train_dataloader, criterion, optimizer, epoch, args.epochs, device, args.batch_size,
              train_dataset, scheduler, train_loss, train_accu)

        best_model_wts, min_loss, loss_now = val(model, epoch, val_dataloader, device, optimizer, criterion,
                                                 args.batch_size, val_dataset, val_loss, val_accu, epoch_record,
                                                 time_elapsed_val, min_loss, best_model_wts)
        #visual(epoch_record, train_loss, val_loss, val_accu)
        early_stopping(loss_now, model)
        if early_stopping.early_stop:
            print('early stopping')
            break
    time_elapsed = time.time() - since  # Total training and validation time
    print('Training and Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    save(train_loss, train_accu, val_loss, val_accu, best_model_wts, min_loss, time_elapsed, time_elapsed_val)


if __name__ == '__main__':
    args = parse_args()
    main()



