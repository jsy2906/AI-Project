# Drive mount and package import
## Use Colab
## Drive mount
from google.colab import drive
drive.mount('/content/drive')

## Import package
import os
import time
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import torchvision
from torchvision import datasets, transforms
import torchvision.models as models

import random


# Load and Prepare data
## data 경로 설정 
root = os.path.join(os.getcwd(), "drive", "MyDrive", "Colab Notebooks", "data")

## Dataset 준비
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
        
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)[:,:,0]      # 3차원으로 구성된 mask 를 label 로 쓰기 위해 변환

        mask[mask > 0] = 1

        # there is only one class
        mask = torch.as_tensor(mask, dtype=torch.uint8)

        target = {}
        target["masks"] = mask

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
        
## Data Transforms
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = transforms.ToTensor()(image)
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image, target):
        image = transforms.Resize(self.size)(image)
        if "masks" in target:
            target["masks"] = transforms.Resize(self.size)(target["masks"].unsqueeze(dim=0)).squeeze()
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image, target):
        image = transforms.RandomCrop(self.size)(image)
        if "masks" in target:
            target["masks"] = transforms.RandomCrop(self.size)(target["masks"].unsqueeze(dim=0)).squeeze()
        return image, target
    
class Normalize(object):
    def __call__(self, image, target):
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
      
def get_transform(train):
    transforms = [ToTensor(), Resize((224,224)), Normalize()]
    if train:
        choice = random.choice([RandomHorizontalFlip(0.5), RandomCrop((224, 224))])
        transforms.append(choice)
        if RandomCrop((224, 224)) in transforms:
            transforms.remove(Resize((224, 224)))
    return Compose(transforms)
  
## Split and Load
dent_train = Dataset(os.path.join(root,'./train'), get_transform(train=True))
dent_valid = Dataset(os.path.join(root,'./valid'), get_transform(train=False))
dent_test = Dataset(os.path.join(root,'./test'), get_transform(train=False))

train_loader = DataLoader(dent_train, batch_size=10, shuffle=True, drop_last=True)
valid_loader = DataLoader(dent_valid, batch_size=10, shuffle=False, drop_last=True)
test_loader = DataLoader(dent_test, batch_size=1, shuffle=False, drop_last=True)


# Trainer class
class Semantic_Seg_Trainer(nn.Module):
    def __init__(self, model, opt="adam", num_class=2, lr=0.001, cl_lr = 0.001, has_scheduler=False, device="cpu", log_dir="./logs", max_epoch=10):
        """
          Args:
            model: 사용할 model
            opt: optimizer
            lr: learning rate
            has_scheduler: learning rate scheduler 사용 여부
            device: 사용할 device (cpu/cuda)
        """
        super().__init__()
        
        self.max_epoch = max_epoch
        self.model = model                            
        self.loss = nn.CrossEntropyLoss()             # loss function 정의
        self.num_class = num_class

        self._get_optimizer(opt=opt.lower(), lr=lr, cl_lr = cl_lr)   # optimizer 정의
        self.has_scheduler = has_scheduler            # scheduler 사용여부 
        if self.has_scheduler:
            self._get_scheduler()

        self.device = device                          # 사용할 device
        
        self.log_dir = log_dir
        if not os.path.exists(log_dir): os.makedirs(log_dir)

    def _get_optimizer(self, opt, lr=0.001, cl_lr = 0.001):
        """
          Args:
            opt: optimizer
            lr: learning rate
        """
        if opt == "sgd":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        elif opt == "adam": 
            self.optimizer = torch.optim.Adam( [{ 'params' : self.model.backbone.parameters(), 'lr' : lr },
                                                { 'params' : self.model.classifier.parameters(), 'lr' : cl_lr }] )
        else:
          raise ValueError(f"optimizer {opt} is not supproted")

    def _get_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=5, gamma=0.5, verbose=True)

    def train(self, train_loader, valid_loader, max_epochs=10, disp_epochs=1, visualize=False):
        """
          네트워크를 학습시키는 함수
          Args:
            train_loader: 학습에 사용할 train dataloader
            valid_loader: validation에 사용할 dataloader
            max_epochs: 학습을 진행할 총 epoch 수
            disp_epochs: 학습 log를 display 할 epoch 주기
            visualize: 학습 진행 과정에서 결과 이미지를 visualize 
        """
        print("===== Train Start =====")
        start_time = time.time()   
        history = {"train_loss": [], "valid_loss": [], "train_miou": [], "valid_miou": []}
        
        t_miou = []
        for e in range(max_epochs):
            print(f"Start Train Epoch {e}")
            train_loss, train_miou = self._train_epoch(train_loader)
            print(f"Start Valid Epoch {e}")
            valid_loss, valid_miou = self._valid_epoch(valid_loader)
            
            history["train_loss"].append(train_loss)      # 현재 epoch에서 성능을 history dict에 저장
            history["valid_loss"].append(valid_loss)      #
            
            history["train_miou"].append(train_miou)      # 
            history["valid_miou"].append(valid_miou)      #

            if self.has_scheduler:         # scheduler 사용할 경우 step size 조절
                self.scheduler.step()

            if e % disp_epochs == 0:        # disp_epoch 마다 결과값 출력 
                print(f"Epoch: {e}, train loss: {train_loss:>6f}, valid loss: {valid_loss:>6f}, train miou: {train_miou:>6f}, valid miou: {valid_miou:>6f}, time: {time.time()-start_time:>3f}")
                start_time = time.time()   
                t_miou.append(valid_miou)

            self.plot_history(history, save_name=f"{self.log_dir}/log_epoch_{e}.png")       # 그래프 출력
            
            # 모델 
            model_dir = os.path.join(root, 'model')
            if not os.path.exists(model_dir): 
                os.makedirs(model_dir)
            torch.save(self.model.state_dict(), os.path.join(model_dir, f'segmentation_epoch_{e}.pth'))
        print(f'Best Epoch Num : {t_miou.index(max(t_miou))}, Best MIoU : {max(t_miou)}')

    def _train_epoch(self, train_loader, disp_step=10):
        """
          model를 training set 한 epoch 만큼 학습시키는 함수
          Args:
            train_loader: 학습에 사용할 train dataloader
          Returns:
            training set 한 epoch의 평균 loss, 평균 accuracy
        """
        epoch_loss = 0
        
        miou = 0
        ious = np.zeros([2])
        
        self.model.train()                 # self.model을 train 모드로 전환 --> nn.Module의 내장함수
        cnt = 0
        epoch_start_time = time.time()
        start_time = time.time()
        for (x, y) in train_loader:        # x: data, y:label
            cnt += 1

            x = x.to(self.device)
            label = y['masks'].to(self.device).type(torch.long)
            
            out = self.model(x)              # model이 예측한 output
            loss = self.loss(out['out'], label)       

            self.optimizer.zero_grad()       # backwardpass를 통한 network parameter 업데이트
            loss.backward()                  # 
            self.optimizer.step()            # 
            
            epoch_loss += loss.to("cpu").item()    
            
            out_background = torch.argmin(out['out'].to("cpu"), dim=1).to(self.device)           # meanIoU 계산을 위한 데이터 변형
            out_target = torch.argmax(out['out'].to("cpu"), dim=1).to(self.device)               #
            
            ious[0] += self.batch_segmentation_iou(out_background, torch.logical_not(label).type(torch.long)) # ious[0]:background IoU
            ious[1] += self.batch_segmentation_iou(out_target, label)                                         # ious[1]:파손 IoU
            
            if cnt % disp_step == 0:
                iou_back = ious[0]/(cnt*x.shape[0])
                iou_scratch = ious[1]/(cnt*x.shape[0])
                miou = (ious[0]/(cnt*x.shape[0]) + ious[1]/(cnt*x.shape[0])) / 2.
                
                print(f"Iter: {cnt}/{len(train_loader)}, train epcoh loss: {epoch_loss/(cnt):>6f}, miou: {miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time()-start_time:>3f}")
                start_time = time.time()   

        epoch_loss /= len(train_loader)  
        
        iou_back = ious[0]/(cnt*x.shape[0])
        iou_scratch = ious[1]/(cnt*x.shape[0])
        epoch_miou = (ious[0]/(cnt*x.shape[0]) + ious[1]/(cnt*x.shape[0])) / 2.
        print(f"Train loss: {epoch_loss:>6f}, miou: {epoch_miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time()-epoch_start_time:>3f}")

        return epoch_loss, epoch_miou
  
    def _valid_epoch(self, valid_loader, disp_step=10):
        """
          현재 model의 성능을 validation set에서 측정하는 함수
          Args:
            valid_loader: 학습에 사용할 valid dataloader
          Returns:
            validation set 의 평균 loss, 평균 accuracy
        """
        epoch_loss = 0
        
        miou = 0
        ious = np.zeros([2])
                      
        self.model.eval()                  # self.model을 eval 모드로 전환 --> nn.Module의 내장함수
        cnt = 0
        epoch_start_time = time.time()
        start_time = time.time()
        with torch.no_grad():              # model에 loss의 gradient를 계산하지 않음
            for (x, y) in valid_loader:
                cnt += 1
                x = x.to(self.device)
                label = y['masks'].to(self.device).type(torch.long)

                out = self.model(x) 
                loss = self.loss(out['out'], label)
                      
                epoch_loss += loss.to("cpu").item()
                
                out_background = torch.argmin(out['out'].to("cpu"), dim=1).to(self.device)
                out_target = torch.argmax(out['out'].to("cpu"), dim=1).to(self.device)

                ious[0] += self.batch_segmentation_iou(out_background, torch.logical_not(label).type(torch.long))
                ious[1] += self.batch_segmentation_iou(out_target, label)
                    
                if cnt % disp_step == 0:
                    iou_back = ious[0]/(cnt*x.shape[0])
                    iou_scratch = ious[1]/(cnt*x.shape[0])
                    miou = (ious[0]/(cnt*x.shape[0]) + ious[1]/(cnt*x.shape[0])) / 2.
                    print(f"Iter: {cnt}/{len(valid_loader)}, valid epcoh loss: {epoch_loss/(cnt):>6f}, miou: {miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time()-start_time:>3f}")
                    start_time = time.time()   

        epoch_loss /= len(valid_loader)
        
        iou_back = ious[0]/(cnt*x.shape[0])
        iou_scratch = ious[1]/(cnt*x.shape[0])
        epoch_miou = (ious[0]/(cnt*x.shape[0]) + ious[1]/(cnt*x.shape[0])) / 2.
        print(f"Valid loss: {epoch_loss:>6f}, miou: {epoch_miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time()-epoch_start_time:>3f}")

        return epoch_loss, epoch_miou

    def plot_history(self, history, save_name=None):
        """
          history에 저장된 model의 성능을 graph로 plot
          Args:
            history: dictionary with keys {"train_loss","valid_loss",  }
                     각 item 들은 epoch 단위의 성능 history의 list
        """
        fig = plt.figure(figsize=(16, 8))
         
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(history["train_loss"], color="red", label="train loss")
        ax.plot(history["valid_loss"], color="blue", label="valid loss")
        ax.title.set_text("Loss")
        ax.legend()
        
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(history["train_miou"], color="red", label="train miou")
        ax.plot(history["valid_miou"], color="blue", label="valid miou")
        ax.title.set_text("miou")
        ax.legend()

        plt.show()
                      
        if not save_name == None:     # graph 저장
            plt.savefig(save_name)
                    
    def test(self, test_loader):
        """
          현재 model의 성능을 test set에서 측정하는 함수
          Args:
            test_loader: 학습에 사용할 test dataloader
          Returns:
            test set 의 평균 loss, 평균 accuracy
        """
        print("===== Test Start =====")
        start_time = time.time()
        epoch_loss = 0
        
        miou = 0
        ious = np.zeros([2])
                      
        self.model.eval()                  # self.model을 eval 모드로 전환 --> nn.Module의 내장함수
        cnt = 0
        epoch_start_time = time.time()
        start_time = time.time()
        with torch.no_grad():              # model에 loss의 gradient를 계산하지 않음
            for (x, y) in test_loader:
                cnt += 1
                x = x.to(self.device)
                label = y['masks'].to(self.device).type(torch.long)

                out = self.model(x) 
                loss = self.loss(out['out'], label)

                epoch_loss += loss.to("cpu").item()
                      
                out_background = torch.argmin(out['out'].to("cpu"), dim=1).to(self.device)
                out_target = torch.argmax(out['out'].to("cpu"), dim=1).to(self.device)

                ious[0] += self.batch_segmentation_iou(out_background, torch.logical_not(label).type(torch.long))
                ious[1] += self.batch_segmentation_iou(out_target, label)
                
                if cnt % 10 == 0:
                    iou_back = ious[0]/(cnt*x.shape[0])
                    iou_scratch = ious[1]/(cnt*x.shape[0])
                    miou = (ious[0]/(cnt*x.shape[0]) + ious[1]/(cnt*x.shape[0])) / 2.
                    print(f"Iter: {cnt}/{len(test_loader)}, test epcoh loss: {epoch_loss/(cnt):>6f}, miou: {miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time()-start_time:>3f}")
                    start_time = time.time()  

        epoch_loss /= len(test_loader)
        
        iou_back = ious[0]/(cnt*x.shape[0])
        iou_scratch = ious[1]/(cnt*x.shape[0])
        epoch_miou = (ious[0]/(cnt*x.shape[0]) + ious[1]/(cnt*x.shape[0])) / 2.
        
        print(f"Test loss: {epoch_loss:>6f}, miou: {epoch_miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time()-epoch_start_time:>3f}")

    def batch_segmentation_iou(self, outputs, labels):
        """
            outputs, labels : (batch, h, w)
        """
        
        SMOOTH = 1e-6

        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0

        iou = (intersection + SMOOTH) / (union + SMOOTH)
        
        return torch.sum(iou).to("cpu").numpy()
      
        
# Model load
seg = models.segmentation.deeplabv3_resnet101(pretrained=True)
seg.classifier[3] = nn.LeakyReLU()
seg.classifier[4] = nn.Conv2d(256, 2, 1, 1)

## Model Train
device = 'cuda'
trainer = Semantic_Seg_Trainer(model = seg, opt='adam', lr=0.0001, cl_lr = 0.001, max_epoch=30, device = device, has_scheduler=False).to(device)

start_time = time.time()
trainer.train(train_loader, valid_loader, max_epochs=30, disp_epochs=1)
print(f"Training time : {time.time()-start_time:>3f}")

## Model Test
trainer.test(test_loader)


# Fine Tuning
def show_plot(image, mask, prediction):
  '''결과를 그래프로 보여주는 함수'''
  
  # ground truth
  mask = np.array(mask)[:, :, 0]
  mask[mask>0] = 1
  mask = torch.as_tensor(mask, dtype=torch.uint8)

  # visualization
  fig = plt.figure(figsize=(25, 40))

  ax = fig.add_subplot(131)
  ax.imshow(image)
  ax.title.set_text('Image')

  ax = fig.add_subplot(132)
  ax.imshow(mask)
  ax.title.set_text('Ground Truth')

  ax = fig.add_subplot(133)
  ax.imshow(out)
  ax.title.set_text('Prediction')

  fig.show()

## Load Fine Tuning Model
device = 'cuda'
model_dir = os.path.join(root, 'model')
model_path = os.path.join(model_dir, 'segmentation_epoch_2.pth')
seg.load_state_dict(torch.load(model_path))

trainer = Semantic_Seg_Trainer(model = seg, opt='adam', lr=0.0001, cl_lr = 0.001, max_epoch=15, device = device, has_scheduler=False).to(device)
start_time = time.time()
trainer.test(test_loader)
print(f"Training time : {time.time()-start_time:>3f}")

## Test
image = Image.open(os.path.join(root,'Assignment_3_Dataset/scratch_small/test/images/20190220_4820_20177753_7f11836e36ac8c88ab34d8665a0e0a4b.jpg'))
mask = Image.open(os.path.join(root,'Assignment_3_Dataset/scratch_small/test/masks/20190220_4820_20177753_7f11836e36ac8c88ab34d8665a0e0a4b.jpg'))

infer_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

device = 'cuda'
input_image = infer_transform(image).to(device)

model_dir = os.path.join(root, 'model')
model_path = os.path.join(model_dir, 'segmentation_epoch_2.pth')
seg.load_state_dict(torch.load(model_path))
seg.eval()
output = seg(input_image.unsqueeze(dim=0))

cls = torch.argmax(output['out'][0].to("cpu"), dim=0).numpy()
out = np.zeros_like(cls)
out[cls>=1] = 1

show_plot(image, mask, out) # Show Result
