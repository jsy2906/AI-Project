# Mount Drive
## Colab 사용시
from google.colab import drive
drive.mount('/content/drive')

# Import Package
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
%matplotlib inline

import sklearn

# Load and Prepare data
## Colab 사용시
root = os.path.join(os.getcwd(), "drive", "MyDrive", "Colab Notebooks", "data")

def get_data(data, root, download=True, transform=transforms.ToTensor()):
  if data.lower() == "mnist":
    train = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test = datasets.MNIST(root=root, train=False, download=download, transform=transform)
  elif data.lower() == "fmnist":
    train = datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
    test = datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)
  else:
    raise ValueError(f"data name {data} is not supported.")

  return train, test

mnist_tr, mnist_test = get_data(data="MNIST", root=root)
fmnist_tr, fmnist_test = get_data(data="FMNIST", root=root)

# Check datasets
def plot_mnist(data, figsize=(20, 10)):
  fig = plt.figure(figsize=figsize)
  for i in range(18):
    img = data[i][0]
    ax = fig.add_subplot(3, 6, i+1)
    ax.imshow(img.reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {data[i][1]}")
  fig.show()
  pass

plot_mnist(data=mnist_tr)
plot_mnist(data=fmnist_tr)

# Split train dataset into train and valid
class mnist_dataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        super().__init__()

        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y
      
## shuffle 추가
def split_train_valid(dataset, valid_ratio=0.1):
    # sklearn 라이브러리를 이용해 데이터 섞어주기
    dataset = sklearn.utils.shuffle(dataset, 
#                                     random_state=42,
                                   )

    n_valid = int(len(dataset) * valid_ratio)
    
    train_data = [i[0] for i in dataset[:-n_valid]]
    valid_data = [i[0] for i in dataset[-n_valid:]]
    train_targets = [i[1] for i in dataset[:-n_valid]]
    valid_targets = [i[1] for i in dataset[-n_valid:]]

    train = mnist_dataset(data=train_data, targets=train_targets)
    valid = mnist_dataset(data=valid_data, targets=valid_targets)

    return train, valid
  
mnist_train, mnist_valid = split_train_valid(dataset=mnist_tr)
fmnist_train, fmnist_valid = split_train_valid(dataset=fmnist_tr)

# Check shuffling result
plot_mnist(data=mnist_train)
plot_mnist(data=fmnist_train)

# Get loaders
b_size = 256
mnist = [DataLoader(dataset=d, batch_size=b_size, shuffle=True, drop_last=True) for d in [mnist_train, mnist_valid, mnist_test]]
fmnist = [DataLoader(dataset=d, batch_size=b_size, shuffle=True, drop_last=True) for d in [fmnist_train, fmnist_valid, fmnist_test]]
datas = {"mnist": mnist, "fmnist": fmnist}

# Model
## Base model
class BaseClassifier(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()

        self.name = "base"
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_class),
        )

    def forward(self, x):
        return self.model(x)
base = BaseClassifier()

## Customized models
### Basic Model보다 더 깊은 모델
class FirstClassifier(nn.Module):
    def __init__(self, n_class=10, act='relu'):
        super().__init__()

        self.name = "first"

        if act.lower() == 'relu':
            self.act = nn.ReLU
        elif act.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU
        elif act.lower() == 'sigmoid':
            self.act = nn.Sigmoid
        else:
            raise ValueError(f"Act F(x) name {act} is not supported")

        self.model =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            self.act(),
            nn.Linear(512, 256),
            self.act(),
            nn.Linear(256, 128),
            self.act(),
            nn.Linear(128, 64),
            self.act(),
            nn.Linear(64, n_class),
        )

    def forward(self, x):
        return self.model(x)
first = FirstClassifier(act='leakyrelu')
      
### 기존에 사용하던 방식으로 Dropout을 Activation f(X) 앞에 추가한 모델
class BefDropClassifier(nn.Module):
    def __init__(self, n_class=10, act='relu'):
        super().__init__()

        self.name = "befdrop"

        if act.lower() == 'relu':
            self.act = nn.ReLU
        elif act.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU
        elif act.lower() == 'sigmoid':
            self.act = nn.Sigmoid
        else:
            raise ValueError(f"Act F(x) name {act} is not supported")

        self.model =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.Dropout(p=0.3),
            self.act(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            self.act(),
            nn.Linear(256, 128),
            nn.Dropout(p=0.3),
            self.act(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.3),
            self.act(),
            nn.Linear(64, n_class),
        )

    def forward(self, x):
        return self.model(x)
befdrop = BefDropClassifier(ant = 'leakyrelu')

### 기존에 사용하던 방식으로 Batch Normalization을 Activation f(X) 앞에 추가한 모델
# 기존에 사용하던 방식으로 Batch Normalization을 Activation f(X) 앞에 추가한 모델
class BefBatchClassifier(nn.Module):
    def __init__(self, n_class=10, act='relu'):
        super().__init__()

        self.name = "befbatch"

        if act.lower() == 'relu':
            self.act = nn.ReLU
        elif act.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU
        elif act.lower() == 'sigmoid':
            self.act = nn.Sigmoid
        else:
            raise ValueError(f"Act F(x) name {act} is not supported")

        self.model =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.BatchNorm1d(256),
            self.act(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            self.act(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            self.act(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            self.act(),
            nn.Linear(32, n_class),
        )

    def forward(self, x):
        return self.model(x)
befbatch = BefBatchClassifier(act = 'leakyrelu')

### Batch Nomalization과 Dropout을 섞어 모델 생성
class BatchDropClassifier(nn.Module):
    def __init__(self, n_class=10, act='relu'):
        super().__init__()

        self.name = "batchdrop"

        if act.lower() == 'relu':
            self.act = nn.ReLU
        elif act.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU
        elif act.lower() == 'sigmoid':
            self.act = nn.Sigmoid
        else:
            raise ValueError(f"Act F(x) name {act} is not supported")

        self.model =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            self.act(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            self.act(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            self.act(),
            nn.Linear(64, n_class),
        )

    def forward(self, x):
        return self.model(x)
batchdrop=BatchDropClassifier(act = 'leakyrelu')

### 기존에 사용하던 방식과는 다르게 Activation f(x) 뒤에 Dropout 추가하여 모델 생성
class AftDropClassifier(nn.Module):
    def __init__(self, n_class=10, act='relu'):
        super().__init__()

        self.name = "aftdrop"

        if act.lower() == 'relu':
            self.act = nn.ReLU
        elif act.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU
        elif act.lower() == 'sigmoid':
            self.act = nn.Sigmoid
        else:
            raise ValueError(f"Act F(x) name {act} is not supported")

        self.model =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            self.act(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            self.act(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            self.act(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            self.act(),
            nn.Linear(64, n_class),
        )

    def forward(self, x):
        return self.model(x)
aftdrop=AftDropClassifier(act = 'leakyrelu')

### 기존에 사용하던 방식과는 다르게 Activation f(x) 뒤에 Batch Nomalization 추가하여 모델 생성
class AftBatchClassifier(nn.Module):
    def __init__(self, n_class=10, act='relu'):
        super().__init__()

        self.name = "aftbatch"

        if act.lower() == 'relu':
            self.act = nn.ReLU
        elif act.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU
        elif act.lower() == 'sigmoid':
            self.act = nn.Sigmoid
        else:
            raise ValueError(f"Act F(x) name {act} is not supported")

        self.model =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            self.act(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            self.act(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            self.act(),
            nn.BatchNorm1d(64),
            nn.Linear(64, n_class),
        )

    def forward(self, x):
        return self.model(x)
aftbatch=AftBatchClassifier(act = 'leakyrelu')

## 여러 모델 불러오는 함수 생성
def get_model(model_name, act):
      if model_name.lower() == "base":
        return BaseClassifier()
      elif model_name.lower() == "first":
        return FirstClassifier(act=act)
      elif model_name.lower() == 'befdrop':
        return BefDropClassifier(act=act)
      elif model_name.lower() == 'befbatch':
        return BefBatchClassifier(act=act)
      elif model_name.lower() == 'batchdrop':
        return BatchDropClassifier(act=act)
      elif model_name.lower() == 'aftdrop':
        return AftDropClassifier(act=act)
      elif model_name.lower() == 'aftbatch':
        return AftBatchClassifier(act=act)
      else:
        raise ValueError(f"model name {model_name} is not supported")
        
### 사용하고 싶은 모델과 함수 불러오기
get_model('aftbatch', 'leakyrelu')

# Trainer
class Trainer(nn.Module):
    def __init__(self, model, opt="sgd", lr=0.001, device="cpu"):
        super().__init__()
        self.path = f"_model_{model.name}_opt_{opt}_lr_{lr}"

        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self._get_optimizer(opt=opt.lower(), lr=lr)

        self.device = device
        pass

    def _get_optimizer(self, opt, lr=0.001):
        if opt == "sgd":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        elif opt == "adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        # 추가
        elif opt == "rmsprop":
            self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"optimizer {opt} is not supproted")

    def train(self, data_name, train_loader, valid_loader, max_epochs=10):
        print("===== Train Start =====")
        history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
        for e in range(max_epochs):
            train_loss, train_acc = self._train_epoch(train_loader)
            valid_loss, valid_acc = self._valid_epoch(valid_loader)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["valid_loss"].append(valid_loss)
            history["valid_acc"].append(valid_acc)

#             if e % 10 == 0:
#                 print(f"Epoch: {e}, train loss: {train_loss:>6f}, train acc: {train_acc:>3f}, valid loss: {valid_loss:>6f}, valid acc: {valid_acc:>3f}")
            if e == max_epochs-1:
                print(f"Epoch: {e+1}, train loss: {train_loss:>6f}, train acc: {train_acc:>3f}, valid loss: {valid_loss:>6f}, valid acc: {valid_acc:>3f}")

        self.plot_history(history, data_name, max_epochs)

    def _train_epoch(self, train_loader):
        epoch_loss, epoch_acc = 0, 0
        self.model.train()
        for (x, y) in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.model(x)
            loss = self.loss(y_hat, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.to("cpu").item()
            epoch_acc += (y_hat.argmax(1) == y).type(torch.float).to("cpu").mean().item()

        return epoch_loss / len(train_loader), epoch_acc / len(train_loader)

    def _valid_epoch(self, valid_loader):
        epoch_loss, epoch_acc = 0, 0
        self.model.eval()
        with torch.no_grad():
            for (x, y) in valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x)
                loss = self.loss(y_hat, y)

                epoch_loss += loss.to("cpu").item()
                epoch_acc += (y_hat.argmax(1) == y).type(torch.float).to("cpu").mean().item()

        return epoch_loss / len(valid_loader), epoch_acc / len(valid_loader)

    def plot_history(self, history, data_name, max_epochs):
        fig = plt.figure(figsize=(10, 5))

        # plt.title(f"Data: {d}, Model: {m}, Activation: {act}, Optimizer: {opt}, lr: {lr}")
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(history["train_loss"], color="red", label="train loss")
        ax.plot(history["valid_loss"], color="blue", label="valid loss")
        ax.set_title("Loss")
        ax.legend()

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(history["train_acc"], color="red", label="train acc")
        ax.plot(history["valid_acc"], color="blue", label="valid acc")
        ax.set_title("Acc")
        ax.legend()

        plt.show()
        
        # Colab 사용시
        path = os.path.join(os.getcwd(), "drive", "MyDrive", "Colab Notebooks", "plots", f"data_{data_name}" + self.path + f"_max_epochs_{max_epochs}.png")
        # 로컬 PC 사용시
#         path = os.path.join(os.getcwd(), "plots", f"data_{data_name}" + self.path + f"_max_epochs_{max_epochs}.png")
        plt.savefig(path, bbox_inches="tight")
        pass

    def test(self, test_loader):
        print("===== Test Start =====")
        epoch_loss, epoch_acc = 0, 0
        self.model.eval()
        with torch.no_grad():
            for (x, y) in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x)
                loss = self.loss(y_hat, y)

                epoch_loss += loss.to("cpu").item()
                epoch_acc += (y_hat.argmax(1) == y).type(torch.float).to("cpu").mean().item()

        epoch_loss /= len(test_loader)
        epoch_acc /= len(test_loader)

#         print(f"Test loss: {epoch_loss:>6f}, Test acc: {epoch_acc:>3f}")
        return epoch_loss, epoch_acc

# Train and Test
p = os.path.join(os.getcwd(), "drive", "MyDrive", "Colab Notebooks", "plots")
if not os.path.isdir(p):
    os.mkdir(p)

## Mnist
### Hyper parameter
models = ["base", 'first', 'befdrop', 'befbatch']
opts = ["sgd", 'adam', 'rmsprop']
lrs = [0.001, 0.05, 0.01]
acts = ['relu', 'leakyrelu', 'sigmoid']
max_epochs = 30

### Models = Basic, First, Befdrop, Befbatch Model
d = 'mnist'
test_acc = 0
for m in models:
    for act in acts:
        for opt in opts:
            for lr in lrs:
                model = get_model(model_name = m, act=act)
                trainer = Trainer(model=model, opt=opt, lr=lr)
                print(f"===== Data: {d}, Model: {m}, Activation: {act}, Optimizer: {opt}, lr: {lr}, max_epochs: {max_epochs} =====")
                trainer.train(data_name=d, train_loader=datas[d][0], valid_loader=datas[d][1], max_epochs=max_epochs)
                loss, acc = trainer.test(test_loader=datas[d][2])
                print(f"Test loss: {(loss) :>6f}, Test acc: {(acc) :>3f}")
                if test_acc < acc: test_acc = acc
                print()
            print()
        print()
    print()
print(f'Best Test Accuracy : {test_acc}')

### Mnist + BatchDrop Model
d = 'mnist'
m = 'batchdrop'
test_acc = 0
for act in acts:
    for opt in opts:
        for lr in lrs:
            model = get_model(model_name = m, act=act)
            trainer = Trainer(model=model, opt=opt, lr=lr)
            print(f"===== Data: {d}, Model: {m}, Activation: {act}, Optimizer: {opt}, lr: {lr}, max_epochs: {max_epochs} =====")
            trainer.train(data_name=d, train_loader=datas[d][0], valid_loader=datas[d][1], max_epochs=max_epochs)
            loss, acc = trainer.test(test_loader=datas[d][2])
            print(f"Test loss: {(loss) :>6f}, Test acc: {(acc) :>3f}")
            if test_acc < acc: test_acc = acc
            print()
        print()
    print()
print()
#print(f'Best Test Accuracy : {test_acc}')

### Mnist + Aftdrop Model
d = 'mnist'
m = 'aftdrop'
test_acc = 0
for act in acts:
    for opt in opts:
        for lr in lrs:
            model = get_model(model_name = m, act=act)
            trainer = Trainer(model=model, opt=opt, lr=lr)
            print(f"===== Data: {d}, Model: {m}, Activation: {act}, Optimizer: {opt}, lr: {lr}, max_epochs: {max_epochs} =====")
            trainer.train(data_name=d, train_loader=datas[d][0], valid_loader=datas[d][1], max_epochs=max_epochs)
            loss, acc = trainer.test(test_loader=datas[d][2])
            print(f"Test loss: {(loss) :>6f}, Test acc: {(acc) :>3f}")
            if test_acc < acc: test_acc = acc
            print()
        print()
    print()
print()
print(f'Best Test Accuracy : {test_acc}')

### Mnist + AftBatch Model
d = 'mnist'
m = 'aftbatch'
test_acc = 0
for act in acts:
    for opt in opts:
        for lr in lrs:
            model = get_model(model_name = m, act=act)
            trainer = Trainer(model=model, opt=opt, lr=lr)
            print(f"===== Data: {d}, Model: {m}, Activation: {act}, Optimizer: {opt}, lr: {lr}, max_epochs: {max_epochs} =====")
            trainer.train(data_name=d, train_loader=datas[d][0], valid_loader=datas[d][1], max_epochs=max_epochs)
            loss, acc = trainer.test(test_loader=datas[d][2])
            print(f"Test loss: {(loss) :>6f}, Test acc: {(acc) :>3f}")
            if test_acc < acc: test_acc = acc
            print()
        print()
    print()
print()
print(f'Best Test Accuracy : {test_acc}')

## FMNIST
### Hyper Parameter
models = ["base", 'first', 'befdrop', 'befbatch']
opts = ["sgd", 'adam', 'rmsprop']
lrs = [0.001, 0.05, 0.01]
acts = ['relu', 'leakyrelu', 'sigmoid']
max_epochs = 30

### Models = Basic, First, Befdrop, Befbatch Model
d = 'fmnist'
test_acc = 0
for m in models:
    for act in acts:
        for opt in opts:
            for lr in lrs:
                model = get_model(model_name = m, act=act)
                trainer = Trainer(model=model, opt=opt, lr=lr)
                print(f"===== Data: {d}, Model: {m}, Activation: {act}, Optimizer: {opt}, lr: {lr}, max_epochs: {max_epochs} =====")
                trainer.train(data_name=d, train_loader=datas[d][0], valid_loader=datas[d][1], max_epochs=max_epochs)
                loss, acc = trainer.test(test_loader=datas[d][2])
                print(f"Test loss: {(loss) :>6f}, Test acc: {(acc) :>3f}")
                if test_acc < acc: test_acc = acc
                print()
            print()
        print()
    print()
print()
print(f'Best Test Accuracy : {test_acc}')

### FMnist + BatchDrop Model
d = 'fmnist'
m = 'batchdrop'
test_acc = 0
for act in acts:
    for opt in opts:
        for lr in lrs:
            model = get_model(model_name = m, act=act)
            trainer = Trainer(model=model, opt=opt, lr=lr)
            print(f"===== Data: {d}, Model: {m}, Activation: {act}, Optimizer: {opt}, lr: {lr}, max_epochs: {max_epochs} =====")
            trainer.train(data_name=d, train_loader=datas[d][0], valid_loader=datas[d][1], max_epochs=max_epochs)
            loss, acc = trainer.test(test_loader=datas[d][2])
            print(f"Test loss: {(loss) :>6f}, Test acc: {(acc) :>3f}")
            if test_acc < acc: test_acc = acc
            print()
        print()
    print()
print()
print(f'Best Test Accuracy : {test_acc}')

### FMnist + AftBatch Model
d = 'fmnist'
m = 'aftdrop'
test_acc = 0
for act in acts:
    for opt in opts:
        for lr in lrs:
            model = get_model(model_name = m, act=act)
            trainer = Trainer(model=model, opt=opt, lr=lr)
            print(f"===== Data: {d}, Model: {m}, Activation: {act}, Optimizer: {opt}, lr: {lr}, max_epochs: {max_epochs} =====")
            trainer.train(data_name=d, train_loader=datas[d][0], valid_loader=datas[d][1], max_epochs=max_epochs)
            loss, acc = trainer.test(test_loader=datas[d][2])
            print(f"Test loss: {(loss) :>6f}, Test acc: {(acc) :>3f}")
            if test_acc < acc: test_acc = acc
            print()
        print()
    print()
print()
print(f'Best Test Accuracy : {test_acc}')

### FMnist + AftBatch Model
d = 'fmnist'
m = 'aftbatch'
test_acc = 0
for act in acts:
    for opt in opts:
        for lr in lrs:
            model = get_model(model_name = m, act=act)
            trainer = Trainer(model=model, opt=opt, lr=lr)
            print(f"===== Data: {d}, Model: {m}, Activation: {act}, Optimizer: {opt}, lr: {lr}, max_epochs: {max_epochs} =====")
            trainer.train(data_name=d, train_loader=datas[d][0], valid_loader=datas[d][1], max_epochs=max_epochs)
            loss, acc = trainer.test(test_loader=datas[d][2])
            print(f"Test loss: {(loss) :>6f}, Test acc: {(acc) :>3f}")
            if test_acc < acc: test_acc = acc
            print()
        print()
    print()
print()
print(f'Best Test Accuracy : {test_acc}')

# Best Model
## MNIST
d = "mnist"
m = "befdrop"
act = 'leakyrelu'
opt = "adam"
lr = 0.001
max_epochs = 30
# max_epochs = 25

epoch_loss, epoch_acc = 0, 0
for _ in range(3):
    model = get_model(model_name = m, act=act)
    trainer = Trainer(model=model, opt=opt, lr=lr)
    print(f"===== Data: {d}, Model: {m}, Optimizer: {opt}, lr: {lr}, max_epochs: {max_epochs} =====")
    trainer.train(data_name=d, train_loader=datas[d][0], valid_loader=datas[d][1], max_epochs=max_epochs)
    loss, acc = trainer.test(test_loader=datas[d][2])
    print(f"Test loss: {(loss) :>6f}, Test acc: {(acc) :>3f}")
    epoch_loss += loss
    epoch_acc += acc
    print()
print(f"Final Test loss: {(epoch_loss/3) :>6f}, Test acc: {(epoch_acc/3) :>3f}")

## FMNIST
d = "fmnist"
m = "first"
act = 'leakyrelu'
opt = "rmsprop"
lr = 0.001
max_epochs = 30
# max_epochs = 25

epoch_loss, epoch_acc = 0, 0
for _ in range(3):
    model = get_model(model_name = m, act=act)
    trainer = Trainer(model=model, opt=opt, lr=lr)
    print(f"===== Data: {d}, Model: {m}, Optimizer: {opt}, lr: {lr}, max_epochs: {max_epochs} =====")
    trainer.train(data_name=d, train_loader=datas[d][0], valid_loader=datas[d][1], max_epochs=max_epochs)
    loss, acc = trainer.test(test_loader=datas[d][2])
    print(f"Test loss: {(loss) :>6f}, Test acc: {(acc) :>3f}")
    epoch_loss += loss
    epoch_acc += acc
    print()
    
print(f"Final Test loss: {(epoch_loss/3) :>6f}, Test acc: {(epoch_acc/3) :>3f}")
