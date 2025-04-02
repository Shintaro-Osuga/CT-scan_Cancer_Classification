import torch
import torch.nn as nn
import torch.functional as f
import timm
import torchvision

class Inception_model(nn.Module):
    def __init__(self, num_classes:int = 4):
        super(Inception_model, self).__init__()    
        self.model = timm.create_model('inception_v4', pretrained=False, num_classes=4)
        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
class VGG_model(nn.Module):
    def __init__(self, num_classes:int=4):
        super(VGG_model, self).__init__()
        self.model = torchvision.models.vgg16(pretrained=False)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        # self.model.classifier[6] = nn.Sequential(nn.Linear(4096, 1024),
        #                                          nn.BatchNorm1d(1024),
        #                                          nn.SiLU(),
        #                                          nn.Dropout(0.1),
        #                                          nn.Linear(1024, 256),
        #                                          nn.SiLU(),
        #                                          nn.Linear(256, num_classes))
        # self.model.classifier = nn.Sequential(nn.Linear(25088, 4096),
        #                                       nn.ReLU(),
        #                                       nn.BatchNorm1d(4096),
        #                                       nn.Dropout(0.1),
        #                                       nn.Linear(4096, 4096),
        #                                       nn.ReLU(),
        #                                       nn.BatchNorm1d(4096),
        #                                       nn.Dropout(0.1),
        #                                       nn.Linear(4096, 2048),
        #                                       nn.ReLU(),
        #                                       nn.BatchNorm1d(2048),
        #                                       nn.Dropout(0.1),
        #                                       nn.Linear(2048, 1024),
        #                                       nn.ReLU(),
        #                                       nn.BatchNorm1d(1024),
        #                                       nn.Dropout(0.1),
        #                                       nn.Linear(1024, 4))
        # self.model.classifier = nn.Sequential(nn.Linear(25088, 4096),
        #                                       nn.BatchNorm1d(4096),
        #                                       nn.Dropout(0.1),
        #                                       nn.Linear(4096, 2048),
        #                                       nn.BatchNorm1d(2048),
        #                                       nn.Linear(2048, 4))
    def forward(self, x):
        return self.model(x)
    
class ResNet_model(nn.Module):
    def __init__(self, num_classes:int=4, resnet:str='152', emsem:bool = False):
        super(ResNet_model, self).__init__()
        if resnet == '152':
            self.model = torchvision.models.resnet152(pretrained=False)
        elif resnet == '18':
            self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        if emsem == True:
            num_classes = 128

        self.model.fc = nn.Sequential(
            nn.Linear(512,512),
            nn.Linear(512,512),
            nn.Dropout(0.2),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.model(x)
    
class Incept_ResNet_model(nn.Module):
    def __init__(self, num_classes:int=4, emsem:bool = False):
        super(Incept_ResNet_model, self).__init__()
        if emsem == True:
            num_classes = 128

        self.model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class Vit_model(nn.Module):
    def __init__(self, num_classes:int=4):
        super(Vit_model, self).__init__()
        # self.model = torchvision.models.vit_h_14(weights=torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1, num_classes=num_classes)
        self.model = torchvision.models.vit_h_14(weights=torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        
        self.model.heads = torch.nn.Identity()
        
        self.classifier = nn.Sequential(
                                        nn.Linear(1280, 1024),
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, 1024),
                                        nn.Linear(1024, 256),
                                        nn.Linear(256, num_classes)
                                        )
        
    def forward(self, x):
        return self.classifier(self.model(x))
    
class Small_CNN(nn.Module):
    def __init__(self, in_channels, num_classes:int = 4, emsem:bool=False):
        super(Small_CNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3,3)),
                                 nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,1)),
                                 nn.BatchNorm2d(32),
                                 nn.SiLU(),
                                 nn.Dropout2d(0.2),
                                 nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5,5)),
                                 nn.MaxPool2d((3,3)),
                                 nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,1)),
                                 nn.BatchNorm2d(32),
                                 nn.SiLU(),
                                 nn.Dropout2d(0.2),
                                 nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(7,7)),
                                 nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,1)),
                                 nn.BatchNorm2d(32),
                                 nn.SiLU(),
                                 nn.Dropout2d(0.2),
                                 nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3)),
                                 nn.MaxPool2d((3,3)),
                                 nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,1)),
                                 nn.BatchNorm2d(32),
                                 nn.SiLU(),
                                 nn.Dropout2d(0.2),
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1)),
                                 nn.Flatten(start_dim=1)
                                 )
        

        if emsem == True:
            num_classes = 128
        
        self.classifier = nn.Sequential(nn.Linear(3136, out_features=256),
                                        nn.Linear(in_features=256, out_features=512),
                                        nn.BatchNorm1d(512),
                                        nn.SiLU(),
                                        nn.Dropout1d(0.1),
                                        nn.Linear(512, 128),
                                        nn.SiLU(),
                                        nn.Linear(128, 32),
                                        nn.Linear(32, num_classes))
        
    def forward(self, x):
        out = self.cnn(x)
        # print(x.size())
        out = self.classifier(out)
        return out
    

class Emsemble(nn.Module):
    def __init__(self, num_classes, in_channels:int=3, resnet:str='152'):
        super(Emsemble, self).__init__()

        self.resnet = ResNet_model(num_classes=num_classes, resnet=resnet, emsem=True)

        self.incept_res = Incept_ResNet_model(num_classes=num_classes, emsem=True)

        self.standard = Small_CNN(in_channels=in_channels, num_classes=num_classes, emsem=True)

        self.classifier = nn.Sequential(nn.Linear(128*3, 512),
                                        nn.Linear(512, 512),
                                        nn.BatchNorm1d(512),
                                        nn.SiLU(),
                                        nn.Dropout(0.1),
                                        nn.Linear(512, 128),
                                        nn.SiLU(),
                                        nn.Dropout(0.1),
                                        nn.Linear(128, 32),
                                        nn.Linear(32, num_classes),)
        
    def forward(self, x):
        resnet_out = self.resnet(x)
        ir_out = self.incept_res(x)
        std_out = self.standard(x)

        concat_out = torch.concat([resnet_out, ir_out, std_out], dim=1)

        # print(concat_out.size())

        out = self.classifier(concat_out)
        return out