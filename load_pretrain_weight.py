import torch
from collections import OrderedDict

def load_pretrained_model(model):
# 加载保存的state_dict
    new_state_dict = OrderedDict()
    list_pre = []  #预训练模型参数名
    list_my=[]   #我的模型参数名
    checkpoint = torch.load('C:/Users/Admin/Desktop/256x256_diffusion.pt', map_location='cpu')

    for name, param in model.state_dict().items():
        list_my.append(name)

    for key, value in checkpoint.items():
        list_pre.append(key)
        # 使用列表推导式找到不相等的值
    differences = [x for x in list_my if x not in list_pre] + [y for y in list_pre if y not in list_my]
    # 使用列表推导式和for循环找到相同的值
    #common_values = [x for x in list_my if x in list_pre]

    # for key, value in checkpoint.items():
    #     if key in common_values:
    #         new_state_dict[key] = value
    for key, value in checkpoint.items():
        if key in differences:
            continue
        new_state_dict[key]=value
    model.load_state_dict(new_state_dict)
    print("finished load pretrained model")
    return model


def load_pretrained_model_v2(model,pretrain_checkpoint):
    """

    :param model: current model
    :return: pretrained model
    """
    # 加载保存的state_dict
    new_state_dict = OrderedDict()
    state = torch.load(pretrain_checkpoint)
    checkpoint = state['state_dict']


    for name, param in model.state_dict().items():
        #print(name)
        for key, value in checkpoint.items():
            #print(key)
            if name==key:
                #print(key)
                if param.shape==value.shape:
                    new_state_dict[name]=value
    model.load_state_dict(new_state_dict, strict=False)
    print("finished load pretrained model")
    return model


def frozen_part_param(pretrained_model):
    """
    :param model: pretrained model
    :return:  frozened part weight model
    """

    # for param in model.transformer.parameters():
    # param.requires_grad = False
    #1. name, param in model.named_parameters():  # 获取模型的参数和名称
    #2.将参数名放在列表里或放在txt文件中

    file_path = '/root/data/face_expression/vit_cnn.txt'  # 要固定的参数名文件路径
    fr = open(file_path)
    text_list = []  # the weight name to froze
    for line in fr.readlines():  # 逐行读取
        line = line.strip()
        line = line.split(' ')  # 以空格作为分隔符，对进行分解
        text_list.append(line[0])
    # print(text_list)

    for name, param in pretrained_model.named_parameters():  # 获取模型的参数和名称
        if name in text_list:
            param.requires_grad = False #冻住该参数
    return pretrained_model

def load_pretrained_model_simple(model,path):
# 加载保存的state_dict

    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint)
    print("finished load pretrained model")
    return model

