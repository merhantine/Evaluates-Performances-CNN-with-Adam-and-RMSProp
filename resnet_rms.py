#!/usr/bin/env python
# coding: utf-8

#import library-library yang digunakan (Numpy,pandas, seaborn dan lain-lain)
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, auc 

np.random.seed(0)
torch.manual_seed(0)
#%matplotlib inline
sns.set_style('darkgrid')

#mengecek device yang tersedia.Jika cuda/GPU ada maka digunakan cuda/GPU, jika tidak ada CPU yang digunakan#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We're using =>", device)

#lokasi folder dataset di folder revtesis
#Dalam revtesis ada 2 folder: train dan test
#Dataset dalam folder train akan dibagi 2 secara acak untuk training dan validasi 
#perbandingan data set training validasi dan testing : 80% 10% 10% 
root_dir = '/floyd/input/revtesis/'
home_dir = '/floyd/home/'
print("The data lies here =>", root_dir)

#transform image untuk membuat variasi gambar yang bertujuan mengurangi overfitting
#Ada 5 transform untuk training : Randomcrop, RandomRotation, Random HorizontalFlip, ColorJitter, Random VerticalFlip.
#Ada 1 transform untuk testing : Resize
image_transforms = {
    "train": transforms.Compose([
        transforms.RandomCrop((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
}

#Load dataset training dari folder /floyd/input/revtesis/Train
rps_dataset = datasets.ImageFolder(root = root_dir + "Train",
                                   transform = image_transforms["train"]

#Informasi tentag folder Training 
rps_dataset

#Mengeluarkan daftar nama kelas dengan index seperti berikut
#{'buffalo': 0,
# 'elephant': 1,
# 'gazelleGrants': 2,
# 'gazelleThomsons': 3,
# 'giraffe': 4,
# 'guineaFowl': 5,
# 'hartebeest': 6,
# 'hyenaSpotted': 7,
# 'lionFemale': 8,
# 'warthog': 9,
# 'zebra': 10}
rps_dataset.class_to_idx

#Menukar posisi menjadi index dengan nama kelas seperti berikut
#0: 'buffalo',
# 1: 'elephant',
# 2: 'gazelleGrants',
# 3: 'gazelleThomsons',
# 4: 'giraffe',
# 5: 'guineaFowl',
# 6: 'hartebeest',
# 7: 'hyenaSpotted',
# 8: 'lionFemale',
# 9: 'warthog',
# 10: 'zebra'}
idx2class = {v: k for k, v in rps_dataset.class_to_idx.items()}
idx2class

#Membuat function get_class_distribution untuk menghitung jumlah masing-masing kelas berdasarkan jumlah label
def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    for _, label_id in dataset_obj:
        label = idx2class[label_id]
        count_dict[label] += 1
    return count_dict

#Membuat function plotting/grafik  hasil get_class_distribution, lalu disimpan di folder /floyd/home/
def plot_from_dict(dict_obj, plot_title, **kwargs):
    return sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", hue="variable", **kwargs).set_title(plot_title)
plt.figure(figsize=(15,8))
plot_from_dict(get_class_distribution(rps_dataset), plot_title="Entire Dataset (before train/val/test split)")
plt.savefig(home_dir+'beforesplit.png')

#Membuat Kelas EarlyStopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

#mengecek size dataset training
rps_dataset_size = len(rps_dataset)
rps_dataset_indices = list(range(rps_dataset_size))
np.random.shuffle(rps_dataset_indices)

#Membagi dataset training untuk training dan validasi
#Jumlah gambar di folder training adalah 90% dari total. Lalu diambil 0.11 nya untuk dataset validasi
#Sehingga komposisi training 80%, validasi 10%, testing 10%
#Pembagian dilakukan random
val_split_index = int(np.floor(0.11 * rps_dataset_size))
train_idx, val_idx = rps_dataset_indices[val_split_index:], rps_dataset_indices[:val_split_index]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

#Load dataset testing dari folder /floyd/input/revtesis/Testing
rps_dataset_test = datasets.ImageFolder(root = root_dir + "Test",
                                        transform = image_transforms["test"])
rps_dataset_test

#loader untuk training, validasi dan testing
train_loader = DataLoader(dataset=rps_dataset, shuffle=False, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset=rps_dataset, shuffle=False, batch_size=1, sampler=val_sampler)
test_loader = DataLoader(dataset=rps_dataset_test, shuffle=False, batch_size=1)

##Membuat function get_class_distribution, digunakan setelah pemisahan dataset training dan validasi
def get_class_distribution_loaders(dataloader_obj, dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    if dataloader_obj.batch_size == 1:    
        for _,label_id in dataloader_obj:
            y_idx = label_id.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1
    else: 
        for _,label_id in dataloader_obj:
            for idx in label_id:
                y_idx = idx.item()
                y_lbl = idx2class[y_idx]
                count_dict[str(y_lbl)] += 1
    return count_dict

#Membuat function plotting/grafik  distribusi kelas dataset training dan validasi, lalu disimpan di folder /floyd/home/
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,7))
plot_from_dict(get_class_distribution_loaders(train_loader, rps_dataset), plot_title="Train Set", ax=axes[0])
plot_from_dict(get_class_distribution_loaders(val_loader, rps_dataset), plot_title="Val Set", ax=axes[1])
plt.savefig(home_dir+'Trainvalset.png')

# Memilih satu tensor dari batch pertama untuk dilihat hasil gambarnya
#Gambar yang dipilih ditunjukkan dengan hasil permutasi
single_batch = next(iter(train_loader))
single_batch[0].shape
print("Output label tensors: ", single_batch[1])
print("\nOutput label tensor shape: ", single_batch[1].shape)
single_image = single_batch[0][0]
single_image.shape
plt.imshow(single_image.permute(1, 2, 0))

# We do single_batch[0] because each batch is a list 
# where the 0th index is the image tensor and 1st index is the
# output label.
single_batch_grid = utils.make_grid(single_batch[0], nrow=4)
plt.figure(figsize = (10,10))
plt.imshow(single_batch_grid.permute(1, 2, 0))

#========================================================================================
# Load pretrained ResNet50 Model
resnet50 = models.resnet50(pretrained=True)
resnet50 = resnet50.to(device)
# Freeze model parameters
for param in resnet50.parameters():
    param.requires_grad = False

# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = resnet50.fc.in_features

resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 11), # Since 10 possible outputs
    nn.LogSoftmax(dim=1) # For using NLLLoss()
)

# Convert model to be used on GPU
resnet50 = resnet50.to(device)
###

model = resnet50.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.000001)

#========================================================================================

# Function untuk mengukur akurasi
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc) * 100
    return acc


## Mulai Training
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

# initialize the early_stopping object
early_stopping = EarlyStopping(patience=20, verbose=True)
    
print("Begin training.")
for e in tqdm(range(1, 56)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch).squeeze()
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
    # VALIDATION
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch).squeeze()
            y_val_pred = torch.unsqueeze(y_val_pred, 0)
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            val_epoch_loss += train_loss.item()
            val_epoch_acc += train_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
    print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
   
    early_stopping(val_epoch_loss, model)
    if early_stopping.early_stop:
            print("Early stopping")
            break

#Menghitung akurasi training dan validasi
#menghitung loss training dan validasi
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

## Akhir Training

# Menggambarkan 2 grafik : akurasi training dan validasi terhadap epoch; loss training dan validasi terhadap epoch
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
plt.savefig(home_dir+'Trainvalloss.png')


# Testing
y_pred_list = []
y_true_list = []
with torch.no_grad():
    for x_batch, y_batch in tqdm(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_test_pred = model(x_batch)
        y_test_pred = torch.log_softmax(y_test_pred, dim=1)
        _, y_pred_tag = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())
## Akhir Testing

#print classification report, confusion matrix
#Nilai dalam report ini dalam 2 desimal
y_pred_list = [i[0] for i in y_pred_list]
y_true_list = [i[0] for i in y_true_list]
print(classification_report(y_true_list, y_pred_list))
print(confusion_matrix(y_true_list, y_pred_list))
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=idx2class, index=idx2class)
fig, ax = plt.subplots(figsize=(7,5))         
sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
plt.savefig(home_dir+'confuse.png')
print("Balanced Accuracy:", balanced_accuracy_score(y_true_list, y_pred_list))

#Optional saja untuk print akurasi, precision, recall dan F1 supaya mendapatkan hasil dalam 5 desimal
print("Accuracy Score:", accuracy_score(y_true_list, y_pred_list))
print("Precision Score:", precision_score(y_true_list, y_pred_list, average='weighted'))
print("Recall Score:", recall_score(y_true_list, y_pred_list, average='weighted'))
print("F1 Score:", f1_score(y_true_list, y_pred_list, average='weighted'))








