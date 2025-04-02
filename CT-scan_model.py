import pandas as pd
import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import sys

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

class CFG:
    verbose = 1
    seed = 87
    epochs = 16
    batch_size = 24
    lr = 1e-2
    precision = torch.float32
    
def build_transforms():
    data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(10),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
    ])
    return data_transform

def build_dataset(path):
    # path = "./Data/"
    train_data = datasets.ImageFolder(root=path+"train",
                                      transform=build_transforms(), 
                                      target_transform=None) 

    test_data = datasets.ImageFolder(root=path+"test",
                                     transform=build_transforms())


    validation_data = datasets.ImageFolder(root=path+"valid",
                                           transform=build_transforms())

    # train_data, test_data ,validation_data
    
    class_names = train_data.classes
    # class_names
    
    class_dict = train_data.class_to_idx
    
    # Change the key
    old_key1 = 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'
    new_key1 = 'adenocarcinoma_left.lower.lobe'

    old_key2 = 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'
    new_key2 = 'large.cell.carcinoma_left.hilum'

    old_key3 = 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
    new_key3 = 'squamous.cell.carcinoma_left.hilum'

    # ----------------------------------------- Disclaimer for Future ---------------------------------------
    # MIGHT NEED TO CHANGE CLASS KEY VALUES FOR TEST AND VALIDATION SETS
    
    # Step 1: Add new key-value pair
    class_dict[new_key1] = class_dict[old_key1]
    class_dict[new_key2] = class_dict[old_key2]
    class_dict[new_key3] = class_dict[old_key3]


    # Step 2: Remove old key-value pair
    del class_dict[old_key1]
    del class_dict[old_key2]
    del class_dict[old_key3]
    
    class_names[0]=new_key1
    class_names[1]=new_key2
    class_names[3]=new_key3
    
    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=CFG.batch_size,
                                num_workers=1,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=CFG.batch_size,
                                num_workers=1,
                                shuffle=False)
    valid_dataloader = DataLoader(dataset=validation_data,
                                batch_size=CFG.batch_size,
                                num_workers=1,
                                shuffle=False)
    
    return train_dataloader, test_dataloader, valid_dataloader

def create_model(device:str = 'cuda:0', model_type:str=None, resver:str='', num_classes:int=4):
    from Inception_models import Inception_classifier
    from Inception_CFGs import Incept_CFGS
    
    
    
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)
    
    # model = timm.create_model('inception_v4', pretrained=False, num_classes=4)
    import pre_built as pb
    
    if model_type == 'r':
        model = pb.ResNet_model(num_classes=num_classes, resnet=resver)
    elif model_type == 'i':
        model = pb.Inception_model(num_classes=num_classes)
    elif model_type == 'v':
        model = pb.VGG_model(num_classes=num_classes)
    elif model_type == 'ir':
        model = pb.Incept_ResNet_model(num_classes=num_classes)
    elif model_type == 'vit':
        model = pb.Vit_model(num_classes=num_classes)
        model = model.half()
    elif model_type == 'std':
        model = pb.Small_CNN(in_channels=3, num_classes=4)
    elif model_type == 'em':
        model = pb.Emsemble(num_classes=num_classes, in_channels=3, resnet=resver)
    else:
        config = Incept_CFGS.Incept_CNN_CFG
        model = Inception_classifier(inception_cfg = config['Incept_v1']["incept_cfg"],
                                    cnn_classifier_cfg=config["Incept_v1"]["cnn_classifier_cfg"], 
                                    aux_classifer_cfg= config["Incept_v1"]["aux_classifier_cfg"], 
                                    classifier_cfg=    config["Incept_v1"]["classifier_cfg"])

    model.to(device)
    
    return model
    
def train(model:nn.Module, 
          train_dataloader:DataLoader, 
          valid_dataloader:DataLoader, 
          test_dataloader:DataLoader, 
          model_type:str,
          device:str='cuda:0') -> nn.Module:
    # loss_fn = nn.BCEWithLogitsLoss()
    # from torchmetrics import ConfusionMatrix
    # cm = ConfusionMatrix(num_classes=4)
    from sklearn.metrics import confusion_matrix
    import seaborn as sns



    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    
    for epoch in range(CFG.epochs):
        print(f'------------------------------ Epoch {epoch} --------------------------------')
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        for batch_idx, (input, label) in enumerate(tqdm(train_dataloader)):
            input = input.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)
            
            out = model(input)
            
            # print(f'out:{out.size()} | label:{label.size()}')
            
            # print(out)
            # print(nn.Softmax(out))
            # print(label)
            loss = loss_fn(out.float(), label)
            # cm.update(out, label)

            total_loss += loss.item()
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            pred_acc = torch.argmax(torch.softmax(out, dim=1), dim=1)
            total_acc += (pred_acc==label).sum().item()/len(out)
        
        print(f'----------------------------- Train ------------------------------------')
        print(f"train loss: {total_loss/len(train_dataloader)} \ntrain acc: {total_acc/len(train_dataloader)}")

        
        with torch.no_grad():
            total_valid_loss = 0.0
            total_valid_acc = 0.0
            model.eval()
            for batch_idx, (input, label) in enumerate(tqdm(valid_dataloader)):
                input = input.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)
                
                out = model(input)
                
                loss = loss_fn(out.float(),label)
                total_valid_loss += loss.item()
                
                pred_acc = torch.argmax(torch.softmax(out, dim=1), dim=1)
                total_valid_acc += (pred_acc==label).sum().item()/len(out)
            
        print(f'----------------------------- Valid ------------------------------------')
        print(f"valid loss: {total_valid_loss/len(valid_dataloader)} \nvalid acc: {total_valid_acc/len(valid_dataloader)}")

    with torch.no_grad():
        total_test_loss = 0.0
        total_test_acc = 0.0
        y_true = []
        y_pred = []

        model.eval()
        for batch_idx, (input, label) in enumerate(tqdm(test_dataloader)):
            input = input.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)
            
            out = model(input)
            
            loss = loss_fn(out.float(), label)
            total_test_loss += loss.item()
            
            pred_acc = torch.argmax(torch.softmax(out, dim=1), dim=1)
            total_test_acc += (pred_acc==label).sum().item()/len(out)

            y_true.extend(label.detach().cpu().numpy())
            y_pred.extend(pred_acc.detach().cpu().numpy())
    
    print('----------------------------- Test ------------------------------------')
    print(f"test loss: {total_test_loss/len(test_dataloader)} \ntest acc: {total_test_acc/len(test_dataloader)}")
    
    classes = ('Adenocarcinoma', 'Normal', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma')

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sns.heatmap(df_cm, annot=True)

    plt.savefig('./OneDrive/Documents/Grad/IST-691/Project/re/Project/plots/'+model_type+'_output.png')
    
def main():
    # path = "./Users/Shintaro/Documents/Syracuse_MS/Fall_2024/IST_691/Project/Data/"
    path = "./OneDrive/Documents/Grad/IST-691/Project/re/Project/Data/"
    device = 'cuda:0'
    # C:\Users\shint\OneDrive\Documents\Grad\IST-691\Project\re\Project\Data
    torch.cuda.empty_cache()

    # print("----------- VGG laptop -----------")
    # CFG.batch_size=16
    # print(CFG.batch_size)
    # train_d, test_d, valid_d = build_dataset(path=path)
    # model = create_model(model_type='v')
    # train(model=model, train_dataloader=train_d, valid_dataloader=valid_d, test_dataloader=test_d, device=device, model_type='v')
    # torch.cuda.empty_cache()

    # print("----------- VGG -----------")
    # CFG.batch_size=28
    # print(CFG.batch_size)
    # train_d, test_d, valid_d = build_dataset(path=path)
    # model = create_model(model_type='v')
    # train(model=model, train_dataloader=train_d, valid_dataloader=valid_d, test_dataloader=test_d, device=device)
    
    # print("----------- ResNet laptop -----------")
    # CFG.batch_size=20
    # print(CFG.batch_size)
    # train_d, test_d, valid_d = build_dataset(path=path)
    # model = create_model(model_type="r", resver='18')
    # train(model=model, train_dataloader=train_d, valid_dataloader=valid_d, test_dataloader=test_d, device=device, model_type='r')
    # torch.cuda.empty_cache()
    
    # print("----------- ResNet -----------")
    # CFG.batch_size=10
    # print(CFG.batch_size)
    # train_d, test_d, valid_d = build_dataset(path=path)
    # model = create_model(model_type="r", resver='18')
    # train(model=model, train_dataloader=train_d, valid_dataloader=valid_d, test_dataloader=test_d, device=device)
    
    # print("----------- Inception v4 laptop -----------")
    # CFG.batch_size=24
    # print(CFG.batch_size)
    # train_d, test_d, valid_d = build_dataset(path=path)
    # model = create_model(model_type="i")
    # train(model=model, train_dataloader=train_d, valid_dataloader=valid_d, test_dataloader=test_d, device=device, model_type='i')
    
    # print("----------- Inception v4 -----------")
    # CFG.batch_size=24
    # print(CFG.batch_size)
    # train_d, test_d, valid_d = build_dataset(path=path)
    # model = create_model(model_type="i")
    # train(model, train_d, device)
    
    # print("----------- Inception v1 -----------")
    # CFG.batch_size=6
    # print(CFG.batch_size)
    # train_d, test_d, valid_d = build_dataset(path=path)
    # model = create_model()
    # train(model=model, train_dataloader=train_d, valid_dataloader=valid_d, test_dataloader=test_d, device=device)
    
    # print("----------- Inception ResNet v2 -----------")
    # CFG.epochs = 32
    # CFG.batch_size=16
    # print(CFG.batch_size)
    # train_d, test_d, valid_d = build_dataset(path=path)
    # model = create_model(device=device, model_type='ir')
    # train(model=model, train_dataloader=train_d, valid_dataloader=valid_d, test_dataloader=test_d, device=device, model_type='ir')
    # torch.cuda.empty_cache()
    
    # print("----------- ViT -----------")
    # CFG.batch_size=2
    # print(CFG.batch_size)
    # train_d, test_d, valid_d = build_dataset(path=path)
    # model = create_model(device=device, model_type='vit')
    # train(model=model, train_dataloader=train_d, valid_dataloader=valid_d, test_dataloader=test_d, device=device)
    
    # print("----------- Standard -----------")
    # CFG.batch_size=24
    # print(CFG.batch_size)
    # train_d, test_d, valid_d = build_dataset(path=path)
    # model = create_model(device=device, model_type='std')
    # train(model=model, train_dataloader=train_d, valid_dataloader=valid_d, test_dataloader=test_d, device=device, model_type='std')
    # torch.cuda.empty_cache()
    
    print("----------- Emsemble -----------")
    CFG.epochs = 32
    CFG.batch_size=16
    print(CFG.batch_size)
    train_d, test_d, valid_d = build_dataset(path=path)
    model = create_model(device=device, model_type='em', resver='18')
    train(model=model, train_dataloader=train_d, valid_dataloader=valid_d, test_dataloader=test_d, device=device, model_type='em')
    torch.cuda.empty_cache()
    
    
    
    
if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     num_gpus = torch.cuda.device_count()
    #     print(f"Number of available GPUs: {num_gpus}")

    #     for i in range(num_gpus):
    #         gpu_name = torch.cuda.get_device_name(i)
    #         print(f"GPU {i}: {gpu_name}")
    # else:
    #     print("No GPUs available.")
    sys.exit(main())