from __future__ import absolute_import, division, print_function, unicode_literals
import argparse

import os, sys
import io
import pandas as pd
from torch.autograd import Variable


def parse_args():
    parser = argparse.ArgumentParser(description='PoisonSpot Defense')
    parser.add_argument("--batch_level_1", action='store_true', help="Enable batch level weight updates")
    parser.add_argument("--batch_level_2", action='store_true', help="Enable batch level weight updates")
    parser.add_argument("--clean_training", action='store_true', help="Do clean training")
    parser.add_argument("--poisoned_training", action='store_true', help="Do poisoned training")
    parser.add_argument("--sample_level", action='store_true', help="Enable sample level weight updates")
    parser.add_argument("--ext_rel_wts", action='store_true', help="Enable extraction of relevant weights")
    parser.add_argument("--score_samples", action='store_true', help="Enable scoring of suspected samples")
    parser.add_argument("--analyze", action='store_true', help="Enable analysis of suspected samples")
    parser.add_argument("--retrain", action='store_true', help="Enable retraining of the model")
    parser.add_argument("--pr_sus", type=float, default=100, help="Percentage of poisoned data in the suspected set")
    parser.add_argument("--ep_bl", type=int, default=10, help="Number of training epochs for batch level weight capture")
    parser.add_argument("--ep_bl_base", type=int, default=200, help="Number of training epochs before batch level capture")
    parser.add_argument("--ep_sl", type=int, default=10, help="Number of training epochs for sample level training")
    parser.add_argument("--ep_sl_base", type=int, default=200, help="Number of training epochs before sample level capture")
    parser.add_argument("--pr_tgt", type=float, default=0.1, help="Ratio of poisoned data in the target set")
    parser.add_argument("--bs_sl", type=int, default=128, help="Batch size for sample level training")
    parser.add_argument("--bs_bl", type=int, default=128, help="Batch size for batch level training")
    parser.add_argument("--bs", type=int, default=128, help="Batch size for training")
    parser.add_argument("--eps", type=int, default=16, help="Epsilon for the attack")
    parser.add_argument("--vis", type=int, default=255, help="Visibility for label consistent attack")
    parser.add_argument("--target_class", type=int, default=2, help="Target class for the attack")
    parser.add_argument("--source_class", type=int, default=0, help="Source class for the attack")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="Dataset to use for the experiment")
    parser.add_argument("--attack", type=str, default="lc", help="Attack to use for the experiment")
    parser.add_argument("--model", type=str, default="ResNet18", help="Model to use for the experiment")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/", help="Root directory for the datasets")    
    parser.add_argument("--clean_model_path", type=str, default='./saved_models/model_resnet_18160.pth', help="Path to the clean model")
    parser.add_argument("--saved_models_path", type=str, default='./saved_models/', help="Path to save the models")
    parser.add_argument("--global_seed", type=int, default=545, help="Global seed for the experiment")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for the experiment")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for the experiment")
    parser.add_argument("--figure_path", type=str, default="./results/", help="Path to save the figures")
    parser.add_argument("--prov_path", type=str, default="./Training_Prov_Data/", help="Path to save the provenance data")  
    parser.add_argument("--epochs", type=int, default = 200, help="Number of epochs for either clean or poisoned training")
    parser.add_argument("--scenario", type=str, default="from_scratch", help="Scenario to use for the experiment")
    parser.add_argument("--min_features", type=int, default=1, help="Minimum number of features to consider")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of features to consider")
    parser.add_argument("--max_trials", type=int, default=1, help="Maximum number of trials to detect poisons")
    parser.add_argument("--get_result", action='store_true', help="Get results from previous runs")
    parser.add_argument("--force", action='store_true', help="Force the run")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for scoring suspected samples")
    parser.add_argument("--sample_from_test", action='store_true', help="Sample from the test set")
    parser.add_argument("--penalty", type=float, default=1, help="Penalty for the attack")
    parser.add_argument("--cv_model", type=str, default="RandomForest", help="Model to use for cross validation")
    parser.add_argument("--groups", type=int, default=5, help="Number of groups to use for cross validation")
    parser.add_argument("--opt", type=str, default="sgd", help="Optimizer to use for the experiment")
    parser.add_argument("--training_mode",  action='store_false', help="Training mode for the model")
    return parser.parse_args()

args = parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
print(f"Using GPU {args.gpu_id}")
import torch 
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from imblearn.over_sampling import SMOTE 
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')
from scipy import stats
from io import BytesIO
import shap
# import tf.keras.backend as k
from captum.attr import IntegratedGradients
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import normalize

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

from attacks.Labelconsistent.generate_poison_lc import get_lc_cifar10_poisoned_data, get_lc_image_net_poisoned_data
from attacks.Narcissus.generate_poison_narcissus import get_narcissus_cifar10_poisoned_data
from attacks.Sleeperagent.generate_poison_sa import get_sa_cifar10_poisoned_data, get_sa_slt_10_poisoned_data
from attacks.HiddenTriggerBackdoor.generate_poison_hidden_trigger import get_ht_cifar10_poisoned_data, get_ht_stl10_poisoned_data, get_ht_imagenet_poisoned_data
from attacks.mixed.mixed_attacks import get_lc_narcissus_cifar_10_poisoned_data, get_lc_narcissus_sa_cifar_10_poisoned_data

from art.estimators.classification import KerasClassifier
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import load_mnist, preprocess, to_categorical
from art.defences.trainer import AdversarialTrainerMadryPGD
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, Normalize, RandomCrop
from pytorch_pretrained_vit import ViT


import matplotlib
import copy
matplotlib.use('Agg')  
'''
This is the code for implementing PoisonSpot Defense against Narcissus_poison_spot on CIFAR-10 using Integrated Gradients.
'''
import sys
import os
import os.path as osp
import time
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
from models.PRF import prf 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from utils.util import *

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR 
import pickle 
import random 
from models.resnet import ResNet
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
 
def call_bn(bn, x):
    return bn(x)

import torch.nn as nn
import torch.nn.functional as F

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' 
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)

class CustomViT(ViT):
    def __init__(self, *args, **kwargs):
        super(CustomViT, self).__init__(*args, **kwargs)
        self._resize_positional_embeddings()

    def _resize_positional_embeddings(self):
        num_patches = (224 // 16) ** 2  # 224x224 image with 16x16 patches
        seq_length = num_patches + 1  # +1 for the class token
        pos_embedding = self.positional_embedding.pos_embedding

        if seq_length != pos_embedding.size(1):
            print(f"Resizing positional embeddings from {pos_embedding.size(1)} to {seq_length}")
            self.positional_embedding.pos_embedding = nn.Parameter(
                F.interpolate(pos_embedding.unsqueeze(0), size=(seq_length, pos_embedding.size(2)), mode='nearest').squeeze(0)
            )

    def forward(self, x, return_features=False):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw+1,d 
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x[:, 0]) # b,d
            if return_features:
                return x
            x = self.fc(x)  # b,num_classes
        return x

# class CustomViT(ViT):
#     def __init__(self, *args, **kwargs):
#         super(CustomViT, self).__init__(*args, **kwargs)
#         self._resize_positional_embeddings()
#         self.fc = torch.nn.Linear(self.fc.in_features, 100) 

#     def _resize_positional_embeddings(self):
#         num_patches = (224 // 16) ** 2  # 224x224 image with 16x16 patches
#         seq_length = num_patches + 1  # +1 for the class token
#         pos_embedding = self.positional_embedding.pos_embedding

#         if seq_length != pos_embedding.size(1):
#             print(f"Resizing positional embeddings from {pos_embedding.size(1)} to {seq_length}")
#             self.positional_embedding.pos_embedding = nn.Parameter(
#                 F.interpolate(pos_embedding.unsqueeze(0), size=(seq_length, pos_embedding.size(2)), mode='nearest').squeeze(0)
#             )

#     def forward(self, x, return_features=False):
#         """Breaks image into patches, applies transformer, applies MLP head.

#         Args:
#             x (tensor): `b,c,fh,fw`
#         """
#         b, c, fh, fw = x.shape
#         x = self.patch_embedding(x)  # b,d,gh,gw
#         x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
#         if hasattr(self, 'class_token'):
#             x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
#         if hasattr(self, 'positional_embedding'): 
#             x = self.positional_embedding(x)  # b,gh*gw+1,d 
#         x = self.transformer(x)  # b,gh*gw+1,d
#         if hasattr(self, 'pre_logits'):
#             x = self.pre_logits(x)
#             x = torch.tanh(x)
#         if hasattr(self, 'fc'):
#             x = self.norm(x[:, 0]) # b,d
#             if return_features:
#                 return x
#             x = self.fc(x)  # b,num_classes
#         return x
    
    
    
from torchvision.models import resnet18

# Replace the original fc layer with a custom forward method or by manually chaining the layers
class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.resnet = resnet18(pretrained=True)
        
        # Expose specific layers as attributes
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avgpool = self.resnet.avgpool
        
        # Define new layers
        self.fc1 = nn.Linear(self.resnet.fc.in_features, 512)
        self.fc2 = nn.Linear(512, 100)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Forward through custom layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x



def get_loaders_from_dataset(poisoned_train_dataset,test_dataset, poisoned_test_dataset, batch_size, target_class, indexes_to_remove = []):
    
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    if type(poisoned_test_dataset) == dict:
        poisoned_test_loader = {}
        for attack_name in poisoned_test_dataset:
            poisoned_test_loader[attack_name] = DataLoader(poisoned_test_dataset[attack_name], batch_size=batch_size, shuffle=False, num_workers=2)
    else:     
        poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    if len(indexes_to_remove) > 0:
        print("Length of indexes to remove: ", len(indexes_to_remove))
        
        indices = [idx.item() if isinstance(idx, torch.Tensor) else idx for img, lbl, idx in poisoned_train_dataset]

        # indexes_to_remove = set([idx.item() if isinstance(idx, torch.Tensor) else idx for idx in indexes_to_remove])
        # assert np.sum(np.array(indices)>50000) > 0
        print(len(set(indices) & set(indexes_to_remove)))
        indexes_to_keep = [i for i, idx in enumerate(indices) if idx not in indexes_to_remove]
        poisoned_train_dataset_filtered = Subset(poisoned_train_dataset, indexes_to_keep)
        target_class_indices = [idx for img, lbl, idx in poisoned_train_dataset_filtered if lbl == target_class]
        
        print("Length of the filtered dataset: ", len(poisoned_train_dataset_filtered))
        poisoned_train_loader = DataLoader(poisoned_train_dataset_filtered, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        poisoned_train_loader = DataLoader(poisoned_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        if type(poisoned_train_dataset[0][2]) == torch.Tensor:
            target_class_indices = [idx.item() for img, lbl, idx in poisoned_train_dataset if lbl == target_class]
        else:
            target_class_indices = [idx for img, lbl, idx in poisoned_train_dataset if lbl == target_class]
    
    
    
    return poisoned_train_loader, test_loader, poisoned_test_loader, target_class_indices
    
def refine_poisoned_dataset(poisoned_train_dataset, indexes_to_remove = []):
    
    if len(indexes_to_remove) > 0:
        print("Length of indexes to remove: ", len(indexes_to_remove))
        class FilteredDataset(Dataset):
            def __init__(self, dataset, indices_to_remove):
                self.dataset = dataset
                self.indices_to_remove = set(indices_to_remove)

                # Filter out the indices
                self.filtered_data = [(img, lbl, idx) for img, lbl, idx in dataset if idx.item() not in self.indices_to_remove]

            def __len__(self):
                return len(self.filtered_data)

            def __getitem__(self, idx):
                return self.filtered_data[idx]  
        
        poisoned_train_dataset_filtered = FilteredDataset(poisoned_train_dataset, indexes_to_remove)
        return poisoned_train_dataset_filtered
    else:
        print("No indexes to remove. Returning the original dataset")
        return poisoned_train_dataset
    
    
    
def get_random_poison_idx(percentage, ignore_set, random_poison_idx, target_class_all, poison_amount, global_seed):
    np.random.seed(global_seed)
    random.seed(global_seed)
    random_poison_per = random.sample(list(set(target_class_all) - ignore_set), int(poison_amount*(100/percentage - 1))) + list(random_poison_idx)
    return random_poison_per


def CustomCNN():
    num_classes=10
    feature_size=4096
    return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, num_classes)
        )
    

    
class DNN(nn.Module):
        def __init__(self, input_dim, n_outputs=10, dropout_rate=0.25, top_bn=False):
            super(DNN, self).__init__()
            self.dropout_rate = dropout_rate
            self.top_bn = top_bn
            
            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 256)
            self.fc4 = nn.Linear(256, 256)
            self.fc5 = nn.Linear(256, 128)
            self.fc6 = nn.Linear(128, 128)
            self.fc7 = nn.Linear(128, 64)
            self.fc8 = nn.Linear(64, 64)
            self.fc9 = nn.Linear(64, 32)
            self.fc10 = nn.Linear(32, n_outputs)
            
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(512)
            self.bn3 = nn.BatchNorm1d(256)
            self.bn4 = nn.BatchNorm1d(256)
            self.bn5 = nn.BatchNorm1d(128)
            self.bn6 = nn.BatchNorm1d(128)
            self.bn7 = nn.BatchNorm1d(64)
            self.bn8 = nn.BatchNorm1d(64)
            self.bn9 = nn.BatchNorm1d(32)
            self.bn10 = nn.BatchNorm1d(n_outputs)

        def forward(self, x):
            h = x
            h = F.leaky_relu(self.bn1(self.fc1(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn2(self.fc2(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn3(self.fc3(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn4(self.fc4(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn5(self.fc5(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn6(self.fc6(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn7(self.fc7(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn8(self.fc8(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn9(self.fc9(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            logit = self.fc10(h)
            if self.top_bn:
                logit = self.bn10(logit)
            return logit

    
 
def evaluate_model(model, test_loader, poisoned_test_loader, criterion, device):
    model.eval()
    model.to(device)
    
    
    if type(poisoned_test_loader) == dict:
        for attack_name in poisoned_test_loader:
            print(f"Testing attack effect for {attack_name}")
            model.eval()
            correct, total = 0, 0
            for i, (images, labels) in enumerate(poisoned_test_loader[attack_name]):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    logits = model(images)
                    out_loss = criterion(logits,labels)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            print('\nAttack success rate %.2f' % (acc*100))
            print('Test_loss:',out_loss)
    else:
        correct, total = 0, 0
        for i, (images, labels) in enumerate(poisoned_test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        print('\nAttack success rate %.2f' % (acc*100))
        print('Test_loss:',out_loss)

    correct_clean, total_clean = 0, 0
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)
            out_loss = criterion(logits,labels)
            _, predicted = torch.max(logits.data, 1)
            total_clean += labels.size(0)
            correct_clean += (predicted == labels).sum().item()
    acc_clean = correct_clean / total_clean
    print('\nTest clean Accuracy %.2f' % (acc_clean*100))
    print('Test_loss:',out_loss)

        
# Training funciton 

def train(model, optimizer, opt, scheduler, criterion, poisoned_train_loader,test_loader, poisoned_test_loader, training_epochs, global_seed, device, training_mode = True):
    
    np.random.seed(global_seed)
    random.seed(global_seed)
    torch.manual_seed(global_seed)
    
    # Use Integrated Gradients to detect the backdoor
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []
    model.to(device)
    
    for epoch in tqdm(range(training_epochs)):
        # Train
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device), labels.to(device), indices
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

        if type(poisoned_test_loader) == dict:
            for attack_name in poisoned_test_loader:
                print(f"Testing attack effect for {attack_name}")
                model.eval()
                correct, total = 0, 0
                for i, (images, labels) in enumerate(poisoned_test_loader[attack_name]):
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        logits = model(images)
                        out_loss = criterion(logits,labels)
                        _, predicted = torch.max(logits.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                acc = correct / total
                test_ACC.append(acc)
                print('\nAttack success rate %.2f' % (acc*100))
                print('Test_loss:',out_loss)
        else:
            # Testing attack effect
            model.eval()
            correct, total = 0, 0
            for i, (images, labels) in enumerate(poisoned_test_loader):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    logits = model(images)
                    out_loss = criterion(logits,labels)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            test_ACC.append(acc)
            print('\nAttack success rate %.2f' % (acc*100))
            print('Test_loss:',out_loss)
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device).float(), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:', out_loss)
    
        
    return model, optimizer, scheduler, train_ACC, test_ACC, clean_ACC, target_ACC

# Plotting function 
def plot_accs(training_epochs, train_ACC, test_ACC, clean_ACC, target_ACC):
    half = np.arange(0,training_epochs)
    plt.figure(figsize=(12.5,8))
    plt.plot(half, np.asarray(train_ACC)[half], label='Training ACC', linestyle="-.", marker="o", linewidth=3.0, markersize = 8)
    plt.plot(half, np.asarray(test_ACC)[half], label='Attack success rate', linestyle="-.", marker="o", linewidth=3.0, markersize = 8)
    plt.plot(half, np.asarray(clean_ACC)[half], label='Clean test ACC', linestyle="-.", marker="o", linewidth=3.0, markersize = 8)
    # plt.plot(half, np.asarray(target_ACC)[half], label='Target class clean test ACC', linestyle="-", marker="o", linewidth=3.0, markersize = 8)
    # plt.plot(half, np.asarray(test_unl_ACC)[half], label='protected test ACC', linestyle="-.", marker="o", linewidth=3.0, markersize = 8)
    plt.ylabel('ACC', fontsize=24)
    plt.xticks(fontsize=20)
    plt.xlabel('Epoches', fontsize=24)
    plt.yticks(np.arange(0,1.1, 0.1),fontsize=20)
    plt.legend(fontsize=20,bbox_to_anchor=(1.016, 1.2),ncol=2)
    plt.grid(color="gray", linestyle="-")
    plt.show(block=True)



def capture_batch_level_weight_updates(random_sus_idx, random_poison_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class,sample_from_test, device, seed, figure_path, training_mode = True, k=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    sus_diff = {}
    clean_diff = {}
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)    
    
    if not sample_from_test:
        target_images = [img for imgs, lbls, idxs in poisoned_train_loader for img, lbl in zip(imgs, lbls) if lbl == target_class]  
        target_indices = [idx for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl == target_class] 
    else:
        transform_train = transforms.Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ])  
        target_images = [transform_train(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
        # target_images = [img for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
        print("Length of target images: ", len(target_images))
        
    
    
    
    for epoch in tqdm(range(training_epochs)):
        # Train
        model.to(device) 
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device), labels.to(device), indices
            
            torch.save(model.state_dict(), f"./temp_folder/temp_weights_batch_{random_num}.pth")
            torch.save(optimizer.state_dict(), f"./temp_folder/temp_optimizer_batch_{random_num}.pth")
            
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
                
            pos_indices = [ind.item() for ind in indices if ind.item() in random_sus_idx]
            if len(pos_indices) > 0:
                if not sample_from_test:
                    available_indices = list(set(indices[labels.cpu().numpy() == target_class].numpy()) - set(pos_indices))
                    remaining_target_indices = list(set(target_indices) - set(pos_indices) - set(available_indices))
                    if len(available_indices) < len(pos_indices):
                        extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices = available_indices 
                    else:
                        extra_clean_indices = []
                        clean_indices = random.sample(available_indices, len(pos_indices)) 
                    ignore_set = set(clean_indices) | set(pos_indices)
                    available_indices = list(set(indices[labels.cpu().numpy() == target_class].numpy()) - ignore_set)
                    if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices_2 = extra_clean_indices_2
                    elif len(available_indices) == 0:
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
                        clean_indices_2 = []
                    else:
                        extra_clean_indices_2 = []
                        clean_indices_2 = random.sample(available_indices, len(pos_indices))
                
                    remaining_indices = [ind.item() for ind in indices if ind.item() not in pos_indices and ind.item() not in clean_indices and ind.item() not in clean_indices_2]
                    
                    # if not (set(remaining_indices) & set(random_poison_idx) == set() and set(clean_indices) & set(random_poison_idx) == set() and set(clean_indices_2) & set(random_poison_idx) == set()):
                    #     print(set(remaining_indices) & set(random_poison_idx), set(clean_indices) & set(random_poison_idx), set(clean_indices_2) & set(random_poison_idx))

                    assert set(remaining_indices) & set(random_poison_idx) == set() and set(clean_indices) & set(random_poison_idx) == set() and set(clean_indices_2) & set(random_poison_idx) == set()
                else:
                    clean_indices = random.sample(range(len(target_images)), len(pos_indices))
                    ignore_set = set(clean_indices) 
                    clean_indices_2 = random.sample(set(range(len(target_images))) - ignore_set, len(pos_indices))
                    remaining_indices = [ind.item() for ind in indices if ind.item() not in pos_indices]
            
                    assert set(remaining_indices) & set(random_poison_idx) == set()
                
                
                
                
                
                
                original_weights = torch.load(f"./temp_folder/temp_weights_batch_{random_num}.pth")
                original_optimizer = torch.load(f"./temp_folder/temp_optimizer_batch_{random_num}.pth")  
                
                pos_model = copy.deepcopy(orig_model)
                pos_model.to(device)
                pos_model.load_state_dict(original_weights)
                pos_model.train(mode = training_mode)
                
                optimizer_pos = torch.optim.SGD(params=pos_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                optimizer_pos.load_state_dict(original_optimizer)
                
                pos_loss = nn.CrossEntropyLoss()
                optimizer_pos.zero_grad()
                
                pos_batch = [images[idx] for idx in range(len(images)) if indices[idx].item() in pos_indices + remaining_indices] 
                pos_batch = torch.stack(pos_batch)
                
                poison_labels = [labels[idx] for idx in range(len(labels)) if indices[idx].item() in pos_indices + remaining_indices]
                poison_labels = torch.stack(poison_labels)
                
                output = pos_model(pos_batch)
                pred_labels = output.argmax(dim=1)
                
                loss = pos_loss(output, poison_labels)
                loss.backward()
                optimizer_pos.step()
                
                temp_sus = {name: (pos_model.state_dict()[name] - original_weights[name]).cpu() for name in pos_model.state_dict().keys()}
                
                
                
                
                
                original_weights = torch.load(f"./temp_folder/temp_weights_batch_{random_num}.pth")
                original_optimizer = torch.load(f"./temp_folder/temp_optimizer_batch_{random_num}.pth")
                
                clean_model = copy.deepcopy(orig_model)
                clean_model.to(device)
                clean_model.load_state_dict(original_weights)
                clean_model.train(mode = training_mode)
                
                optimizer_clean = torch.optim.SGD(params=clean_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                optimizer_clean.load_state_dict(original_optimizer)
                
                clean_loss = nn.CrossEntropyLoss()
                optimizer_clean.zero_grad()
                
                if not sample_from_test: 
                    clean_batch = [images[idx] for idx in range(len(images)) if indices[idx] in clean_indices + remaining_indices]
                    clean_batch += [target_images[i].to(device) for i, idx in enumerate(target_indices) if idx in extra_clean_indices]
                    clean_batch = torch.stack(clean_batch)
                    
                    clean_labels = [labels[idx] for idx in range(len(labels)) if indices[idx] in clean_indices + remaining_indices]
                    clean_labels += [torch.tensor(target_class).to(device)] * len(extra_clean_indices)
                    
                    clean_labels = torch.stack(clean_labels)
                else:
                    clean_batch = [images[idx] for idx in range(len(images)) if indices[idx].item() in remaining_indices]
                    clean_batch += [target_images[idx].to(device) for idx in clean_indices]
                    clean_batch = torch.stack(clean_batch)
                    
                    clean_labels = [labels[idx] for idx in range(len(labels)) if indices[idx].item() in remaining_indices]
                    clean_labels += [torch.tensor(target_class).to(device)] * len(clean_indices)
                    clean_labels = torch.stack(clean_labels)
                    
                    
                output = clean_model(clean_batch)
                pred_labels = output.argmax(dim=1)
                
                loss = clean_loss(output, clean_labels)
                loss.backward()
                optimizer_clean.step()
                
                temp_clean = {name: (clean_model.state_dict()[name] - original_weights[name]).cpu() for name in clean_model.state_dict().keys()}
                    
                original_weights = torch.load(f"./temp_folder/temp_weights_batch_{random_num}.pth")
                original_optimizer = torch.load(f"./temp_folder/temp_optimizer_batch_{random_num}.pth")
                
                clean_model_2 = copy.deepcopy(orig_model)
                clean_model_2.to(device)
                clean_model_2.load_state_dict(original_weights)
                clean_model_2.train(mode = training_mode)
                
                optimizer_clean_2 = torch.optim.SGD(params=clean_model_2.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                optimizer_clean_2.load_state_dict(original_optimizer)
                
                clean_loss_2 = nn.CrossEntropyLoss()
                optimizer_clean_2.zero_grad()
                
                if not sample_from_test:
                    clean_batch_2 = [images[idx] for idx in range(len(images)) if indices[idx] in clean_indices_2 + remaining_indices]
                    clean_batch_2 += [target_images[i].to(device) for i, idx in enumerate(target_indices) if idx in extra_clean_indices_2]
                    clean_batch_2 = torch.stack(clean_batch_2)
                    
                    clean_labels_2 = [labels[idx] for idx in range(len(labels)) if indices[idx] in clean_indices_2 + remaining_indices]
                    clean_labels_2 += [torch.tensor(target_class).to(device)] * len(extra_clean_indices_2)
                    clean_labels_2 = torch.stack(clean_labels_2)
                else:
                    clean_batch_2 = [images[idx] for idx in range(len(images)) if indices[idx].item() in remaining_indices]
                    clean_batch_2 += [target_images[idx].to(device) for idx in clean_indices_2]
                    clean_batch_2 = torch.stack(clean_batch_2)
                    
                    clean_labels_2 = [labels[idx] for idx in range(len(labels)) if indices[idx].item() in remaining_indices]
                    clean_labels_2 += [torch.tensor(target_class).to(device)] * len(clean_indices_2)
                    clean_labels_2 = torch.stack(clean_labels_2)
                
                output = clean_model_2(clean_batch_2)
                pred_labels = output.argmax(dim=1)
                
                loss = clean_loss_2(output, clean_labels_2)
                loss.backward()
                optimizer_clean_2.step()
                
                temp_clean_2 = {name: (clean_model_2.state_dict()[name] - original_weights[name]).cpu() for name in clean_model_2.state_dict().keys()}

                
                sus_diff[(epoch, tuple(pos_indices))] = np.concatenate([np.array(temp_sus[layer]).flatten() for layer in temp_sus]) - np.concatenate([np.array(temp_clean_2[layer]).flatten() for layer in temp_clean_2])
                if not sample_from_test:
                    clean_diff[(epoch, tuple(clean_indices+extra_clean_indices))] = np.concatenate([np.array(temp_clean[layer]).flatten() for layer in temp_clean]) - np.concatenate([np.array(temp_clean_2[layer]).flatten() for layer in temp_clean_2])
                else:
                    clean_diff[(epoch, tuple(np.array(clean_indices) + 50000))] = np.concatenate([np.array(temp_clean[layer]).flatten() for layer in temp_clean]) - np.concatenate([np.array(temp_clean_2[layer]).flatten() for layer in temp_clean_2])
                
                     
                del temp_sus, temp_clean, temp_clean_2 

        torch.cuda.empty_cache()       
                
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

        
        # Testing attack effect
        model.eval()
        correct, total = 0, 0
        for i, (images, labels) in enumerate(poisoned_test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        test_ACC.append(acc)
        print('\nAttack success rate %.2f' % (acc*100))
        print('Test_loss:',out_loss)
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:',out_loss)
    
    sus_inds = [ind for epoch, ind in sus_diff]
    clean_inds = [ind for epoch, ind in clean_diff]
    sus_diff = np.array([sus_diff[key] for key in sus_diff])
    clean_diff = np.array([clean_diff[key] for key in clean_diff])
    
    
    differences = np.abs(np.max(sus_diff, axis=0) - np.max(clean_diff, axis=0))
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    z_scores = (differences - mean_diff) / std_diff
    cut_num = len(np.where(np.abs(z_scores) > k)[0])
    important_features = np.argsort(np.abs(z_scores))[::-1][:cut_num]
    print("Number of important features:", len(important_features))
    plt.figure()
    plt.scatter(range(sus_diff.shape[1]), z_scores, label='Z Scores', alpha=0.5, color='blue')
    plt.savefig(figure_path + f"Max_diff.png")
    
    os.remove(f"./temp_folder/temp_weights_batch_{random_num}.pth")
    os.remove(f"./temp_folder/temp_optimizer_batch_{random_num}.pth")
    
    sus_diff_rel = sus_diff[:, important_features]
    del sus_diff
    clean_diff_rel = clean_diff[:, important_features]
    del clean_diff
    return sus_diff_rel, clean_diff_rel, sus_inds, clean_inds, important_features


def capture_batch_level_weight_updates_2(random_sus_idx, random_poison_idx, model, orig_model, optimizer, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class,sample_from_test, important_features, device, seed, figure_path, training_mode = True, k=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    sus_diff = {}
    clean_diff = {}
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    
    if not sample_from_test:
        target_images = [img for imgs, lbls, idxs in poisoned_train_loader for img, lbl in zip(imgs, lbls) if lbl == target_class]
        target_indices = [idx for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl == target_class]
    else:
        target_images = [img for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl == target_class]
        
    
    
    
    for epoch in tqdm(range(training_epochs)):
        # Train
        model.to(device) 
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device), labels.to(device), indices
            
            torch.save(model.state_dict(), f"./temp_folder/temp_weights_batch_{random_num}.pth")
            torch.save(optimizer.state_dict(), f"./temp_folder/temp_optimizer_batch_{random_num}.pth")
            
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
                
            pos_indices = [ind.item() for ind in indices if ind.item() in random_sus_idx]
            if len(pos_indices) > 0:
                if not sample_from_test:
                    available_indices = list(set(indices[labels.cpu().numpy() == target_class].numpy()) - set(pos_indices))
                    remaining_target_indices = list(set(target_indices) - set(pos_indices) - set(available_indices))
                    if len(available_indices) < len(pos_indices):
                        extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices = available_indices 
                    else:
                        extra_clean_indices = []
                        clean_indices = random.sample(available_indices, len(pos_indices)) 
                    ignore_set = set(clean_indices) | set(pos_indices)
                    available_indices = list(set(indices[labels.cpu().numpy() == target_class].numpy()) - ignore_set)
                    if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices_2 = extra_clean_indices_2
                    elif len(available_indices) == 0:
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
                        clean_indices_2 = []
                    else:
                        extra_clean_indices_2 = []
                        clean_indices_2 = random.sample(available_indices, len(pos_indices))
                
                    remaining_indices = [ind for ind in indices if ind not in pos_indices and ind not in clean_indices and ind not in clean_indices_2]
                    
                    # if not (set(remaining_indices) & set(random_poison_idx) == set() and set(clean_indices) & set(random_poison_idx) == set() and set(clean_indices_2) & set(random_poison_idx) == set()):
                    #     print(set(remaining_indices) & set(random_poison_idx), set(clean_indices) & set(random_poison_idx), set(clean_indices_2) & set(random_poison_idx))

                    assert set(remaining_indices) & set(random_poison_idx) == set() and set(clean_indices) & set(random_poison_idx) == set() and set(clean_indices_2) & set(random_poison_idx) == set()
                else:
                    clean_indices = random.sample(range(len(target_images)), len(pos_indices))
                    ignore_set = set(clean_indices) 
                    clean_indices_2 = random.sample(set(range(len(target_images))) - ignore_set, len(pos_indices))
                    remaining_indices = [ind for ind in indices if ind not in pos_indices]
            
                    assert set(remaining_indices) & set(random_poison_idx) == set()
                
                
                
                
                
                
                original_weights = torch.load(f"./temp_folder/temp_weights_batch_{random_num}.pth")
                original_optimizer = torch.load(f"./temp_folder/temp_optimizer_batch_{random_num}.pth")  
                
                pos_model = copy.deepcopy(orig_model)
                pos_model.to(device)
                pos_model.load_state_dict(original_weights)
                pos_model.train(mode = training_mode)
                
                optimizer_pos = torch.optim.SGD(params=pos_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                optimizer_pos.load_state_dict(original_optimizer)
                
                pos_loss = nn.CrossEntropyLoss()
                optimizer_pos.zero_grad()
                
                pos_batch = [images[idx] for idx in range(len(images)) if indices[idx] in pos_indices + remaining_indices] 
                pos_batch = torch.stack(pos_batch)
                
                poison_labels = [labels[idx] for idx in range(len(labels)) if indices[idx] in pos_indices + remaining_indices]
                poison_labels = torch.stack(poison_labels)
                
                output = pos_model(pos_batch)
                pred_labels = output.argmax(dim=1)
                
                loss = pos_loss(output, poison_labels)
                loss.backward()
                optimizer_pos.step()
                
                temp_sus = {name: (pos_model.state_dict()[name] - original_weights[name]).cpu() for name in pos_model.state_dict().keys()}
                
                
                
                
                
                original_weights = torch.load(f"./temp_folder/temp_weights_batch_{random_num}.pth")
                original_optimizer = torch.load(f"./temp_folder/temp_optimizer_batch_{random_num}.pth")
                
                clean_model = copy.deepcopy(orig_model)
                clean_model.to(device)
                clean_model.load_state_dict(original_weights)
                clean_model.train(mode = training_mode)
                
                optimizer_clean = torch.optim.SGD(params=clean_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                optimizer_clean.load_state_dict(original_optimizer)
                
                clean_loss = nn.CrossEntropyLoss()
                optimizer_clean.zero_grad()
                
                if not sample_from_test: 
                    clean_batch = [images[idx] for idx in range(len(images)) if indices[idx] in clean_indices + remaining_indices]
                    clean_batch += [target_images[i].to(device) for i, idx in enumerate(target_indices) if idx in extra_clean_indices]
                    clean_batch = torch.stack(clean_batch)
                    
                    clean_labels = [labels[idx] for idx in range(len(labels)) if indices[idx] in clean_indices + remaining_indices]
                    clean_labels += [torch.tensor(target_class).to(device)] * len(extra_clean_indices)
                    
                    clean_labels = torch.stack(clean_labels)
                else:
                    clean_batch = [images[idx] for idx in range(len(images)) if idx in remaining_indices]
                    clean_batch += [target_images[idx].to(device) for idx in clean_indices]
                    clean_batch = torch.stack(clean_batch)
                    
                    clean_labels = [labels[idx] for idx in range(len(labels)) if idx in remaining_indices]
                    clean_labels += [torch.tensor(target_class).to(device)] * len(clean_indices)
                    clean_labels = torch.stack(clean_labels)
                    
                    
                output = clean_model(clean_batch)
                pred_labels = output.argmax(dim=1)
                
                loss = clean_loss(output, clean_labels)
                loss.backward()
                optimizer_clean.step()
                
                temp_clean = {name: (clean_model.state_dict()[name] - original_weights[name]).cpu() for name in clean_model.state_dict().keys()}
                    
                original_weights = torch.load(f"./temp_folder/temp_weights_batch_{random_num}.pth")
                original_optimizer = torch.load(f"./temp_folder/temp_optimizer_batch_{random_num}.pth")
                
                clean_model_2 = copy.deepcopy(orig_model)
                clean_model_2.to(device)
                clean_model_2.load_state_dict(original_weights)
                clean_model_2.train(mode = training_mode)
                
                optimizer_clean_2 = torch.optim.SGD(params=clean_model_2.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                optimizer_clean_2.load_state_dict(original_optimizer)
                
                clean_loss_2 = nn.CrossEntropyLoss()
                optimizer_clean_2.zero_grad()
                
                if not sample_from_test:
                    clean_batch_2 = [images[idx] for idx in range(len(images)) if indices[idx] in clean_indices_2 + remaining_indices]
                    clean_batch_2 += [target_images[i].to(device) for i, idx in enumerate(target_indices) if idx in extra_clean_indices_2]
                    clean_batch_2 = torch.stack(clean_batch_2)
                    
                    clean_labels_2 = [labels[idx] for idx in range(len(labels)) if indices[idx] in clean_indices_2 + remaining_indices]
                    clean_labels_2 += [torch.tensor(target_class).to(device)] * len(extra_clean_indices_2)
                    clean_labels_2 = torch.stack(clean_labels_2)
                else:
                    clean_batch_2 = [images[idx] for idx in range(len(images)) if idx in remaining_indices]
                    clean_batch_2 += [target_images[idx].to(device) for idx in clean_indices_2]
                    clean_batch_2 = torch.stack(clean_batch_2)
                    
                    clean_labels_2 = [labels[idx] for idx in range(len(labels)) if idx in remaining_indices]
                    clean_labels_2 += [torch.tensor(target_class).to(device)] * len(clean_indices_2)
                    clean_labels_2 = torch.stack(clean_labels_2)
                
                output = clean_model_2(clean_batch_2)
                pred_labels = output.argmax(dim=1)
                
                loss = clean_loss_2(output, clean_labels_2)
                loss.backward()
                optimizer_clean_2.step()
                
                temp_clean_2 = {name: (clean_model_2.state_dict()[name] - original_weights[name]).cpu() for name in clean_model_2.state_dict().keys()}

                
                sus_diff[(epoch, tuple(pos_indices))] = np.concatenate([np.array(temp_sus[layer]).flatten() for layer in temp_sus])[important_features] - np.concatenate([np.array(temp_clean_2[layer]).flatten() for layer in temp_clean_2])[important_features]
                if not sample_from_test:
                    clean_diff[(epoch, tuple(clean_indices+extra_clean_indices))] = np.concatenate([np.array(temp_clean[layer]).flatten() for layer in temp_clean])[important_features] - np.concatenate([np.array(temp_clean_2[layer]).flatten() for layer in temp_clean_2])[important_features]
                else:
                    clean_diff[(epoch, tuple(np.array(clean_indices) + 50000))] = np.concatenate([np.array(temp_clean[layer]).flatten() for layer in temp_clean])[important_features] - np.concatenate([np.array(temp_clean_2[layer]).flatten() for layer in temp_clean_2])[important_features]
                
                     
                del temp_sus, temp_clean, temp_clean_2 

        torch.cuda.empty_cache()       
                
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

        
        # Testing attack effect
        model.eval()
        correct, total = 0, 0
        for i, (images, labels) in enumerate(poisoned_test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        test_ACC.append(acc)
        print('\nAttack success rate %.2f' % (acc*100))
        print('Test_loss:',out_loss)
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:',out_loss)
    
    sus_inds = [ind for epoch, ind in sus_diff]
    clean_inds = [ind for epoch, ind in clean_diff]
    sus_diff = np.array([sus_diff[key] for key in sus_diff])
    clean_diff = np.array([clean_diff[key] for key in clean_diff])
    
    os.remove(f"./temp_folder/temp_weights_batch_{random_num}.pth")
    os.remove(f"./temp_folder/temp_optimizer_batch_{random_num}.pth")

    return sus_diff, clean_diff, sus_inds, clean_inds


def capture_first_level_batch_sample_weight_updates(random_sus_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    random_sus_idx = set(random_sus_idx)
    
    # target_indices_class = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl == target_class]
    # print(len(set(target_indices_class) & random_sus_idx))
    # print(len(random_sus_idx))
    assert set(target_indices_class) & random_sus_idx == random_sus_idx
    if not sample_from_test:
        target_images = [img for imgs, lbls, idxs in poisoned_train_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
        target_images = torch.stack(target_images).to(device)
        target_indices = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl.item() == target_class and idx.item() not in random_sus_idx]
    else:
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ])
        transform_sa = Compose([
            RandomHorizontalFlip(),
        ])
        
        if attack == 'narcissus' or attack == 'lc':
            target_images = [transform_train(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
            print("attack: ", attack, "Length of target images: ", len(target_images), "with transform")
        elif attack == 'sa':
            target_images = [transform_sa(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
            print("attack: ", attack, "Length of target images: ", len(target_images), "with transform")
        else:
            target_images = [img for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
            print("attack: ", attack, "Length of target images: ", len(target_images), "without transform")
        print(len(target_images))
        target_images = torch.stack(target_images).to(device)
    
    
    sur_model = copy.deepcopy(model)
    if opt == 'sgd':
        sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt == 'adam':
        sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
    
    # if attack != 'ht':
    len_model_params = len(np.concatenate([model.state_dict()[layer].cpu().numpy().flatten() for layer in model.state_dict()]))
    # else:
    #     len_model_params = len(np.concatenate([model.state_dict()[layer].cpu().numpy().flatten() for layer in model.state_dict() if '20' in layer]))
    
    clean_params = np.array([-np.inf] * len_model_params)
    sus_params = np.array([-np.inf] * len_model_params)
    print("Length of clean params: ", len(clean_params))
    
    def optimize_weight_differences(pos_model_state_dict, original_weights, important_features):
        # Extract the keys once to avoid repeated dict key access
        state_dict_keys = list(pos_model_state_dict.keys())
        
        # Preallocate a list for the differences
        differences = []
        
        # Compute the differences and flatten them in a single loop
        for name in state_dict_keys:
            diff = (pos_model_state_dict[name] - original_weights[name]).cpu().numpy().flatten()
            differences.append(diff)
        
        # Concatenate all differences into a single array
        differences_array = np.concatenate(differences)
        
        # Select important features
        optimized_result = differences_array[important_features]
        
        return optimized_result

    
    
    
    for epoch in tqdm(range(training_epochs)):
        # Train
        model.to(device) 
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        
        step1_time_avg = 0
        step2_time_avg = 0
        step3_time_avg = 0
        step_4_1_time_avg = 0
        step_4_2_time_avg = 0
        step_4_3_time_avg = 0
        step_4_4_time_avg = 0
        step_4_5_time_avg = 0
        step5_time_avg = 0
        step5_1_time_avg = 0
        step6_time_avg = 0
        step7_time_avg = 0
        
    
        
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device), labels.to(device), indices                       
            
            start_time = time.time()
            
            # original_weights = copy.deepcopy(model.state_dict())
            if opt == 'sgd':
                torch.save(optimizer.state_dict(), f'./temp_folder/temp_optimizer_{random_num}.pth')
                torch.save(model.state_dict(), f'./temp_folder/temp_weights_{random_num}.pth')
            elif opt == 'adam':
                torch.save(model.state_dict(), f'./temp_folder/temp_weights_{random_num}.pth')
                                 
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
            step1_time = time.time() - start_time
            start_time = time.time()
            step1_time_avg += step1_time
            

            pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
            if len(pos_indices) > 0:
                step2_time = time.time() - start_time
                start_time = time.time()
                step2_time_avg += step2_time
                
                if not sample_from_test:
                    target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
                    available_indices = list(set(target_indices_batch) - set(pos_indices))
                    # print("Length of available indices: ", len(available_indices), "Length of pos indices: ", len(pos_indices), "Length of target indices: ", len(target_indices))
                    if len(available_indices) < len(pos_indices):
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices = available_indices 
                    else:
                        extra_clean_indices = []
                        clean_indices = random.sample(available_indices, len(pos_indices)) 
                    available_indices = list(set(target_indices_batch) - set(pos_indices) - set(clean_indices))
                    if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices_2 = available_indices
                    elif len(available_indices) == 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
                        clean_indices_2 = []
                    else:
                        extra_clean_indices_2 = []
                        clean_indices_2 = random.sample(available_indices, len(pos_indices))
                    
                    remaining_indices = list(set(range(len(indices))) - set(pos_indices) - set(clean_indices) - set(clean_indices_2))
                    assert set(indices[remaining_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices_2]) & set(random_sus_idx) == set()

                    assert set(indices[pos_indices].cpu().numpy()) & set(random_sus_idx) == set(indices[pos_indices].cpu().numpy())
                    assert np.all(labels[clean_indices].cpu().numpy() == target_class)
                    assert np.all(labels[clean_indices_2].cpu().numpy() == target_class)
                    
                else:
                    clean_indices = list(random.sample(range(len(target_images)), len(pos_indices)))
                    clean_indices_2 = list(random.sample(set(range(len(target_images))) - set(clean_indices), len(pos_indices)))
                    remaining_indices =  list(set(range(len(indices))) - set(pos_indices))
                    assert set(indices[remaining_indices].cpu().numpy()) | set(indices[pos_indices].cpu().numpy()) == set(indices.cpu().numpy()) and set(indices[remaining_indices].cpu().numpy()) & set(indices[pos_indices].cpu().numpy()) == set()
                
                
                step3_time = time.time() - start_time
                start_time = time.time()
                step3_time_avg += step3_time
                
                
                step_4_1_time = time.time() - start_time
                step_4_1_time_avg += step_4_1_time
                start_time = time.time()
                
                if opt == 'sgd':
                    original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                    original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    sur_optimizer.load_state_dict(original_optimizer)
                elif opt == 'adam':
                    original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
                
                sur_model.load_state_dict(copy.deepcopy(original_weights))
            
                
                sur_optimizer.zero_grad()
                
                step_4_2_time = time.time() - start_time
                step_4_2_time_avg += step_4_2_time
                start_time = time.time()
                
                
                output = sur_model(images[pos_indices + remaining_indices])
                pred_labels = output.argmax(dim=1)
                
                loss = criterion(output, labels[pos_indices + remaining_indices])
                loss.backward()
                sur_optimizer.step()
                
                step_4_3_time = time.time() - start_time 
                step_4_3_time_avg += step_4_3_time
                start_time = time.time()
                
                
                
                sur_model_state_dict = sur_model.state_dict()
                # if attack != 'ht':
                temp_sus = torch.cat([diff.view(-1) for diff in sur_model_state_dict.values()]).cpu().numpy() - torch.cat([diff.view(-1) for diff in original_weights.values()]).cpu().numpy()
                # else:
                #     temp_sus = torch.cat([diff.view(-1) for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy() - torch.cat([diff.view(-1) for name, diff in original_weights.items() if '20' in name]).cpu().numpy()
                
                
                step4_4_time = time.time() - start_time
                step_4_4_time_avg += step4_4_time
                start_time = time.time()
                
                
                if opt == 'sgd':
                    original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                    original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    sur_optimizer.load_state_dict(original_optimizer)
                elif opt == 'adam':
                    original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
                
            
                sur_model.load_state_dict(copy.deepcopy(original_weights))
                sur_model.train(mode = training_mode)
                
                
                sur_optimizer.zero_grad()

                if not sample_from_test: 
                    if len(extra_clean_indices) > 0:
                        clean_batch = torch.cat([images[clean_indices + remaining_indices], target_images[extra_clean_indices]])
                        clean_labels = torch.cat([labels[clean_indices + remaining_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                    else:
                        clean_batch = images[clean_indices + remaining_indices]
                        clean_labels = labels[clean_indices + remaining_indices]
                else:
                    clean_batch = torch.cat([images[remaining_indices], target_images[clean_indices]])
                    clean_labels = torch.cat([labels[remaining_indices], torch.tensor([target_class] * len(clean_indices)).to(device)])

                # if not sample_from_test: 
                #     if len(extra_clean_indices) > 0:
                #         clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
                #         clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                #     else:
                #         clean_batch = images[clean_indices]
                #         clean_labels = labels[clean_indices]
                # else:
                #     clean_batch = target_images[clean_indices]
                #     clean_batch = torch.tensor([target_class] * len(clean_indices)).to(device)

                output = sur_model(clean_batch) 
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
                sur_model_state_dict = sur_model.state_dict()
                # if attack != 'ht':
                temp_clean = torch.cat([diff.view(-1) for diff in sur_model_state_dict.values()]).cpu().numpy() - torch.cat([diff.view(-1) for diff in original_weights.values()]).cpu().numpy()
                # else:
                #     temp_clean = torch.cat([diff.view(-1) for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy()  - torch.cat([diff.view(-1) for name, diff in original_weights.items() if '20' in name]).cpu().numpy()



                step5_time = time.time() - start_time
                start_time = time.time()
                step5_time_avg += step5_time
                
                if opt == 'sgd':
                    original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                    original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    sur_optimizer.load_state_dict(original_optimizer)
                elif opt == 'adam':
                    original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
                
                sur_model.load_state_dict(copy.deepcopy(original_weights))
                sur_model.train(mode = training_mode)
                
                
                
                sur_optimizer.zero_grad()

                start_time_2 = time.time()                
                if not sample_from_test: 
                    if len(extra_clean_indices_2) > 0:
                        clean_batch = torch.cat([images[clean_indices_2 + remaining_indices], target_images[extra_clean_indices_2]])
                        clean_labels = torch.cat([labels[clean_indices_2 + remaining_indices], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                    else:
                        clean_batch = images[clean_indices_2 + remaining_indices]
                        clean_labels = labels[clean_indices_2 + remaining_indices]
                else:
                    clean_batch = torch.cat([images[remaining_indices], target_images[clean_indices_2]])
                    clean_labels = torch.cat([labels[remaining_indices], torch.tensor([target_class] * len(clean_indices_2)).to(device)])

                # if not sample_from_test: 
                #     if len(extra_clean_indices_2) > 0:
                #         clean_batch = torch.cat([images[clean_indices_2], target_images[extra_clean_indices_2]])
                #         clean_labels = torch.cat([labels[clean_indices_2], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                #     else:
                #         clean_batch = images[clean_indices_2]
                #         clean_labels = labels[clean_indices_2]
                # else:
                #     clean_batch = target_images[clean_indices_2]
                #     clean_labels = torch.tensor([target_class] * len(clean_indices_2)).to(device)

                step5_1_time = time.time() - start_time_2
                step5_1_time_avg += step5_1_time
                
                output = sur_model(clean_batch)
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
                sur_model_state_dict = sur_model.state_dict()
                # if attack != 'ht':
                temp_clean_2 = torch.cat([diff.view(-1) for diff in sur_model_state_dict.values()]).cpu().numpy() - torch.cat([diff.view(-1) for diff in original_weights.values()]).cpu().numpy()
                # else:
                #     temp_clean_2 = torch.cat([diff.view(-1) for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy() - torch.cat([diff.view(-1) for name, diff in original_weights.items() if '20' in name]).cpu().numpy()
            
                
                step6_time = time.time() - start_time
                step6_time_avg += step6_time
                start_time = time.time()
                
                sus_params = np.maximum(sus_params, np.abs(temp_sus - temp_clean_2))
                clean_params = np.maximum(clean_params, np.abs(temp_clean - temp_clean_2))
                
                step7_time = time.time() - start_time
                step7_time_avg += step7_time
                step7_time_avg += step7_time
                
                del temp_sus, temp_clean, temp_clean_2
                
                

        torch.cuda.empty_cache()       
        
        
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

        
        start_time = time.time()
        
        
        
        step7_time = time.time() - start_time
        step7_time_avg += step7_time
        
        print("Step 1 time:", step1_time_avg)
        print("Step 2 time:", step2_time_avg)
        print("Step 3 time:", step3_time_avg)
        print("Step 4_1 time:", step_4_1_time_avg)
        print("Step 4_2 time:", step_4_2_time_avg)
        print("Step 4_3 time:", step_4_3_time_avg)
        print("Step 4_4 time:", step_4_4_time_avg)
        print("Step 4_5 time:", step_4_5_time_avg)
        print("Step 5 time:", step5_time_avg)
        print("Step 5_1 time:", step5_1_time_avg)
        print("Step 6 time:", step6_time_avg)
        print("Step 7 time:", step7_time_avg)
        
        
        
        if type(poisoned_test_loader) == dict:
            for attack_name in poisoned_test_loader:
                print(f"Testing attack effect for {attack_name}")
                model.eval()
                correct, total = 0, 0
                for i, (images, labels) in enumerate(poisoned_test_loader[attack_name]):
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        logits = model(images)
                        out_loss = criterion(logits,labels)
                        _, predicted = torch.max(logits.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                acc = correct / total
                test_ACC.append(acc)
                print('\nAttack success rate %.2f' % (acc*100))
                print('Test_loss:',out_loss)
        else:
            # Testing attack effect
            model.eval()
            correct, total = 0, 0
            for i, (images, labels) in enumerate(poisoned_test_loader):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    logits = model(images)
                    out_loss = criterion(logits,labels)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            test_ACC.append(acc)
            print('\nAttack success rate %.2f' % (acc*100))
            print('Test_loss:',out_loss)
        
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:',out_loss)
    
    
    differences = np.abs(sus_params - clean_params)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    z_scores = (differences - mean_diff) / std_diff
    cut_num = len(np.where(np.abs(z_scores) > k)[0])
    important_features = np.argsort(np.abs(z_scores))[::-1][:cut_num]
    print("Number of important features:", len(important_features))
    plt.figure()
    plt.scatter(range(sus_params.shape[0]), z_scores, label='Z Scores', alpha=0.5, color='blue')
    plt.savefig(figure_path + f"Max_diff.png")
    
    plt.figure()
    plt.scatter(range(sus_params.shape[0]), differences, label='Differences', alpha=0.5, color='red')
    plt.savefig(figure_path + f"differences.png")
    
    if opt == 'sgd':
        os.remove(f'./temp_folder/temp_optimizer_{random_num}.pth')

    
    return important_features

# def capture_first_level_multi_epoch_batch_sample_weight_updates(random_sus_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
    
#     train_ACC = []
#     test_ACC = []
#     clean_ACC = []
#     target_ACC = []

#     separate_rng = np.random.default_rng()
#     random_num = separate_rng.integers(1, 10000)
#     random_sus_idx = set(random_sus_idx)
    
#     target_indices_class = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl == target_class]
#     print(len(set(target_indices_class) & random_sus_idx))
#     print(len(random_sus_idx))
#     assert set(target_indices_class) & random_sus_idx == random_sus_idx
#     if not sample_from_test:
#         target_images = [img for imgs, lbls, idxs in poisoned_train_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
#         target_images = torch.stack(target_images).to(device)
#         target_indices = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl.item() == target_class and idx.item() not in random_sus_idx]
#     else:
#         transform_train = Compose([
#             RandomCrop(32, padding=4),
#             RandomHorizontalFlip(),
#         ])
#         transform_sa = Compose([
#             RandomHorizontalFlip(),
#         ])
        
#         if attack == 'narcissus' or attack == 'lc':
#             target_images = [transform_train(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
#             print("attack: ", attack, "Length of target images: ", len(target_images), "with transform")
#         elif attack == 'sa':
#             target_images = [transform_sa(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
#             print("attack: ", attack, "Length of target images: ", len(target_images), "with transform")
#         else:
#             target_images = [img for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
#             print("attack: ", attack, "Length of target images: ", len(target_images), "without transform")
#         print(len(target_images))
#         target_images = torch.stack(target_images).to(device)
    
    
#     sur_model = copy.deepcopy(model)
#     if opt == 'sgd':
#         sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#     elif opt == 'adam':
#         sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
    
#     len_model_params = len(np.concatenate([model.state_dict()[layer].cpu().numpy().flatten() for layer in model.state_dict()]))
    
    
#     flattened_weights = torch.cat([param.flatten() for param in sur_model.state_dict().values()]) 
    
#     clean_params_dict = {}
#     sus_params_dict = {}
    
    
    
    
    
    
        
#     def optimize_weight_differences(pos_model_state_dict, original_weights, important_features):
#         # Extract the keys once to avoid repeated dict key access
#         state_dict_keys = list(pos_model_state_dict.keys())
        
#         # Preallocate a list for the differences
#         differences = []
        
#         # Compute the differences and flatten them in a single loop
#         for name in state_dict_keys:
#             diff = (pos_model_state_dict[name] - original_weights[name]).cpu().numpy().flatten()
#             differences.append(diff)
        
#         # Concatenate all differences into a single array
#         differences_array = np.concatenate(differences)
        
#         # Select important features
#         optimized_result = differences_array[important_features]
        
#         return optimized_result
    
#     torch.save(optimizer.state_dict(), f'./temp_folder/temp_optimizer_{random_num}.pth')
#     optimizer_state_dict = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
#     sur_optimizer.load_state_dict(optimizer_state_dict)
    
#     os.remove(f'./temp_folder/temp_optimizer_{random_num}.pth')
    
#     original_model = copy.deepcopy(model)
    
#     for epoch in tqdm(range(training_epochs)):
#         clean_params = np.array([-np.inf] * len_model_params)
#         sus_params = np.array([-np.inf] * len_model_params)
#         print("Length of clean params: ", len(clean_params))

#         # Train
#         model.to(device) 
#         model.train(mode = training_mode)
#         acc_meter = AverageMeter()
#         loss_meter = AverageMeter()
#         pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
#         step1_time_avg = 0
#         step2_time_avg = 0
#         step3_time_avg = 0
#         step_4_1_time_avg = 0
#         step_4_2_time_avg = 0
#         step_4_3_time_avg = 0
#         step_4_4_time_avg = 0
#         step_4_5_time_avg = 0
#         step5_time_avg = 0
#         step5_1_time_avg = 0
#         step6_time_avg = 0
#         step7_time_avg = 0
        

        
#         for images, labels, indices in pbar:
#             images, labels, indices = images.to(device), labels.to(device), indices                       
            
#             start_time = time.time()
            
#             original_weights = copy.deepcopy(model.state_dict())

                                 
#             model.zero_grad()
#             optimizer.zero_grad()
#             logits = model(images)
#             loss = criterion(logits, labels)
#             loss.backward()
#             optimizer.step()
            
#             _, predicted = torch.max(logits.data, 1)
#             acc = (predicted == labels).sum().item()/labels.size(0)
#             acc_meter.update(acc)
#             loss_meter.update(loss.item())
#             pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
#             step1_time = time.time() - start_time
#             start_time = time.time()
#             step1_time_avg += step1_time
            

#             pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
#             if len(pos_indices) > 0:
#                 step2_time = time.time() - start_time
#                 start_time = time.time()
#                 step2_time_avg += step2_time
                
#                 if not sample_from_test:
#                     target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
#                     available_indices = list(set(target_indices_batch) - set(pos_indices))
#                     # print("Length of available indices: ", len(available_indices), "Length of pos indices: ", len(pos_indices), "Length of target indices: ", len(target_indices))
#                     if len(available_indices) < len(pos_indices):
#                         indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
#                         remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
#                         extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
#                         clean_indices = available_indices 
#                     else:
#                         extra_clean_indices = []
#                         clean_indices = random.sample(available_indices, len(pos_indices)) 
#                     available_indices = list(set(target_indices_batch) - set(pos_indices) - set(clean_indices))
#                     if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
#                         indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
#                         remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
#                         extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
#                         clean_indices_2 = available_indices
#                     elif len(available_indices) == 0:
#                         indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
#                         remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
#                         extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
#                         clean_indices_2 = []
#                     else:
#                         extra_clean_indices_2 = []
#                         clean_indices_2 = random.sample(available_indices, len(pos_indices))
                    
#                     remaining_indices = list(set(range(len(indices))) - set(pos_indices) - set(clean_indices) - set(clean_indices_2))
#                     assert set(indices[remaining_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices_2]) & set(random_sus_idx) == set()

#                     assert set(indices[pos_indices].cpu().numpy()) & set(random_sus_idx) == set(indices[pos_indices].cpu().numpy())
#                     assert np.all(labels[clean_indices].cpu().numpy() == target_class)
#                     assert np.all(labels[clean_indices_2].cpu().numpy() == target_class)
                    
#                 else:
#                     clean_indices = list(random.sample(range(len(target_images)), len(pos_indices)))
#                     clean_indices_2 = list(random.sample(set(range(len(target_images))) - set(clean_indices), len(pos_indices)))
#                     remaining_indices =  list(set(range(len(indices))) - set(pos_indices))
#                     assert set(indices[remaining_indices].cpu().numpy()) | set(indices[pos_indices].cpu().numpy()) == set(indices.cpu().numpy()) and set(indices[remaining_indices].cpu().numpy()) & set(indices[pos_indices].cpu().numpy()) == set()
                
                
#                 step3_time = time.time() - start_time
#                 start_time = time.time()
#                 step3_time_avg += step3_time
            
    
                
                
#                 step_4_1_time = time.time() - start_time
#                 step_4_1_time_avg += step_4_1_time
#                 start_time = time.time()
                
                
#                 sur_model.load_state_dict(original_weights)
            
                
#                 sur_optimizer.zero_grad()
                
#                 step_4_2_time = time.time() - start_time
#                 step_4_2_time_avg += step_4_2_time
#                 start_time = time.time()
                
                
#                 try:
#                     output = sur_model(images[pos_indices])
#                 except:
#                     sur_model.eval()  
#                     output = sur_model(images[pos_indices])
#                     sur_model.train(mode = training_mode)                      
                    
#                 pred_labels = output.argmax(dim=1)
                
#                 loss = criterion(output, labels[pos_indices])
#                 loss.backward()
#                 sur_optimizer.step()
                
#                 step_4_3_time = time.time() - start_time 
#                 step_4_3_time_avg += step_4_3_time
#                 start_time = time.time()
                
                
                
#                 sur_model_state_dict = sur_model.state_dict()
#                 # if attack != 'ht':
#                 temp_sus = torch.cat([diff.view(-1) for diff in sur_model_state_dict.values()]).cpu().numpy() - torch.cat([diff.view(-1) for diff in original_weights.values()]).cpu().numpy()
#                 # else:
#                     # temp_sus = torch.cat([diff.view(-1) for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy() - torch.cat([diff.view(-1) for name, diff in original_weights.items() if '20' in name]).cpu().numpy()
                
                
#                 step4_4_time = time.time() - start_time
#                 step_4_4_time_avg += step4_4_time
#                 start_time = time.time()
                
                
                
#                 sur_model.load_state_dict(original_weights)
#                 sur_model.train(mode = training_mode)
                
                
#                 sur_optimizer.zero_grad()

#                 if not sample_from_test: 
#                     if len(extra_clean_indices) > 0:
#                         clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
#                         clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
#                     else:
#                         clean_batch = images[clean_indices]
#                         clean_labels = labels[clean_indices ]
#                 else:
#                     clean_batch = target_images[clean_indices]
#                     clean_labels =  torch.tensor([target_class] * len(clean_indices)).to(device)

#                 # if not sample_from_test: 
#                 #     if len(extra_clean_indices) > 0:
#                 #         clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
#                 #         clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
#                 #     else:
#                 #         clean_batch = images[clean_indices]
#                 #         clean_labels = labels[clean_indices]
#                 # else:
#                 #     clean_batch = target_images[clean_indices]
#                 #     clean_batch = torch.tensor([target_class] * len(clean_indices)).to(device)

#                 try:
#                     output = sur_model(clean_batch)
#                 except:
#                     sur_model.eval()  
#                     output = sur_model(clean_batch)
#                     sur_model.train(mode = training_mode)  
               
                    
                    
#                 clean_labels = clean_labels.long()
#                 loss = criterion(output, clean_labels)
#                 loss.backward()
#                 sur_optimizer.step()
                
#                 sur_model_state_dict = sur_model.state_dict()
#                 # if attack != 'ht':
#                 temp_clean = torch.cat([diff.view(-1) for diff in sur_model_state_dict.values()]).cpu().numpy() - torch.cat([diff.view(-1) for diff in original_weights.values()]).cpu().numpy()
#                 # else:
#                 #     temp_clean = torch.cat([diff.view(-1) for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy()  - torch.cat([diff.view(-1) for name, diff in original_weights.items() if '20' in name]).cpu().numpy()



#                 step5_time = time.time() - start_time
#                 start_time = time.time()
#                 step5_time_avg += step5_time
                
                
#                 sur_model.load_state_dict(original_weights)
#                 sur_model.train(mode = training_mode)
                
#                 sur_optimizer.zero_grad()

#                 start_time_2 = time.time()                
#                 if not sample_from_test: 
#                     if len(extra_clean_indices_2) > 0:
#                         clean_batch = torch.cat([images[clean_indices_2], target_images[extra_clean_indices_2]])
#                         clean_labels = torch.cat([labels[clean_indices_2], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
#                     else:
#                         clean_batch = images[clean_indices_2]
#                         clean_labels = labels[clean_indices_2]
#                 else:
#                     clean_batch =  target_images[clean_indices_2]
#                     clean_labels = torch.tensor([target_class] * len(clean_indices_2)).to(device)
                
#                 # if not sample_from_test: 
#                 #     if len(extra_clean_indices_2) > 0:
#                 #         clean_batch = torch.cat([images[clean_indices_2], target_images[extra_clean_indices_2]])
#                 #         clean_labels = torch.cat([labels[clean_indices_2], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
#                 #     else:
#                 #         clean_batch = images[clean_indices_2]
#                 #         clean_labels = labels[clean_indices_2]
#                 # else:
#                 #     clean_batch = target_images[clean_indices_2]
#                 #     clean_labels = torch.tensor([target_class] * len(clean_indices_2)).to(device)

#                 step5_1_time = time.time() - start_time_2
#                 step5_1_time_avg += step5_1_time
                
#                 try:
#                     output = sur_model(clean_batch)
#                 except:
#                     sur_model.eval()  
#                     output = sur_model(clean_batch)
#                     sur_model.train(mode = training_mode)  
                    
                    
#                 clean_labels = clean_labels.long()
#                 loss = criterion(output, clean_labels)
#                 loss.backward()
#                 sur_optimizer.step()
                
#                 sur_model_state_dict = sur_model.state_dict()
#                 # if attack != 'ht':
#                 temp_clean_2 = torch.cat([diff.view(-1) for diff in sur_model_state_dict.values()]).cpu().numpy() - torch.cat([diff.view(-1) for diff in original_weights.values()]).cpu().numpy()
#                 # else:
#                 #     temp_clean_2 = torch.cat([diff.view(-1) for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy() - torch.cat([diff.view(-1) for name, diff in original_weights.items() if '20' in name]).cpu().numpy()
            
                
#                 step6_time = time.time() - start_time   
#                 step6_time_avg += step6_time 
#                 start_time = time.time()
                
#                 sus_params = np.maximum(sus_params, np.abs(temp_sus - temp_clean_2))
#                 clean_params = np.maximum(clean_params, np.abs(temp_clean - temp_clean_2))
                
#                 step7_time = time.time() - start_time
#                 step7_time_avg += step7_time
#                 step7_time_avg += step7_time
                
#                 del temp_sus, temp_clean, temp_clean_2
                
                

#         torch.cuda.empty_cache()       
        
        
#         train_ACC.append(acc_meter.avg)
#         print('Train_loss:',loss)
#         if opt == 'sgd':
#             scheduler.step()

        
#         start_time = time.time()
        
        
        
#         step7_time = time.time() - start_time
#         step7_time_avg += step7_time
        
#         print("Step 1 time:", step1_time_avg)
#         print("Step 2 time:", step2_time_avg)
#         print("Step 3 time:", step3_time_avg)
#         print("Step 4_1 time:", step_4_1_time_avg)
#         print("Step 4_2 time:", step_4_2_time_avg)
#         print("Step 4_3 time:", step_4_3_time_avg)
#         print("Step 4_4 time:", step_4_4_time_avg)
#         print("Step 4_5 time:", step_4_5_time_avg)
#         print("Step 5 time:", step5_time_avg)
#         print("Step 5_1 time:", step5_1_time_avg)
#         print("Step 6 time:", step6_time_avg)
#         print("Step 7 time:", step7_time_avg)
        
        
        
#         if type(poisoned_test_loader) == dict:
#             for attack_name in poisoned_test_loader:
#                 print(f"Testing attack effect for {attack_name}")
#                 model.eval()
#                 correct, total = 0, 0
#                 for i, (images, labels) in enumerate(poisoned_test_loader[attack_name]):
#                     images, labels = images.to(device), labels.to(device)
#                     with torch.no_grad():
#                         logits = model(images)
#                         out_loss = criterion(logits,labels)
#                         _, predicted = torch.max(logits.data, 1)
#                         total += labels.size(0)
#                         correct += (predicted == labels).sum().item()
#                 acc = correct / total
#                 test_ACC.append(acc)
#                 print('\nAttack success rate %.2f' % (acc*100))
#                 print('Test_loss:',out_loss)
#         else:
#             # Testing attack effect
#             model.eval()
#             correct, total = 0, 0
#             for i, (images, labels) in enumerate(poisoned_test_loader):
#                 images, labels = images.to(device), labels.to(device)
#                 with torch.no_grad():
#                     logits = model(images)
#                     out_loss = criterion(logits,labels)
#                     _, predicted = torch.max(logits.data, 1)
#                     total += labels.size(0)
#                     correct += (predicted == labels).sum().item()
#             acc = correct / total
#             test_ACC.append(acc)
#             print('\nAttack success rate %.2f' % (acc*100))
#             print('Test_loss:',out_loss)
        
        
        
#         correct_clean, total_clean = 0, 0
#         for i, (images, labels) in enumerate(test_loader):
#             images, labels = images.to(device), labels.to(device)
#             with torch.no_grad():
#                 logits = model(images)
#                 out_loss = criterion(logits,labels)
#                 _, predicted = torch.max(logits.data, 1)
#                 total_clean += labels.size(0)
#                 correct_clean += (predicted == labels).sum().item()
#         acc_clean = correct_clean / total_clean
#         clean_ACC.append(acc_clean)
#         print('\nTest clean Accuracy %.2f' % (acc_clean*100))
#         print('Test_loss:',out_loss)

#         differences = np.abs(sus_params - clean_params)
#         mean_diff = np.mean(differences)
#         std_diff = np.std(differences)
#         z_scores = (differences - mean_diff) / std_diff
#         cut_num = len(np.where(np.abs(z_scores) > k)[0])
#         important_features = np.argsort(np.abs(z_scores))[::-1][:cut_num]
#         print("Number of important features:", len(important_features), "Epoch:", epoch)
#         if epoch == 0:
#             important_features_avg = important_features
#         else:
#             important_features_avg = np.intersect1d(important_features_avg, important_features)
#             print("Number of important features after intersection:", len(important_features_avg), "Epoch:", epoch)
#         plt.figure()
#         plt.scatter(range(sus_params.shape[0]), z_scores, label='Z Scores', alpha=0.5, color='blue')
#         plt.savefig(figure_path + f"Max_diff.png")
        
#         plt.figure()
#         plt.scatter(range(sus_params.shape[0]), differences, label='Differences', alpha=0.5, color='red')
#         plt.savefig(figure_path + f"differences.png")

#     return important_features_avg



def capture_first_level_multi_epoch_batch_sample_weight_updates(random_sus_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    random_sus_idx = set(random_sus_idx)
    
    dataset = poisoned_train_loader.dataset

    # target_indices_class = [dataset[i][2].item() for i in range(len(dataset)) if dataset[i][1] == target_class]
    # print("Number of target indices in the dataset: ", len(target_indices_class))
    # print("Number of random sus indices: ", len(random_sus_idx))
    # print(len(set(target_indices_class) & random_sus_idx))
    # assert set(target_indices_class) & random_sus_idx == random_sus_idx 
    
    if not sample_from_test:
        target_images = [dataset[i][0] for i in range(len(dataset)) if dataset[i][1] == target_class]
        target_images = torch.stack(target_images).to(device)
        target_indices = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] == target_class and dataset[i][2] not in random_sus_idx]
    else:
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ])
        
        transform_sa = Compose([
            RandomHorizontalFlip(),
        ])
        
        test_dataset = test_loader.dataset
        target_images = [test_dataset[i][0].clone() for i in range(len(test_dataset)) if test_dataset[i][1] == target_class]
        
        if attack == 'narcissus' or attack == 'lc':
            target_images = [transform_train(img) for img in target_images]
        elif attack == 'sa':
            target_images = [transform_sa(img) for img in target_images]
               
        print(len(target_images))
        target_images = torch.stack(target_images).to(device)
        print("target images shape:", target_images.shape, "attack: ", attack)
        
    
    # target_indices_class = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl == target_class]
    # print(len(set(target_indices_class) & random_sus_idx))
    # print(len(random_sus_idx))
    # assert set(target_indices_class) & random_sus_idx == random_sus_idx
    # if not sample_from_test:
    #     target_images = [img for imgs, lbls, idxs in poisoned_train_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
    #     target_images = torch.stack(target_images).to(device)
    #     target_indices = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl.item() == target_class and idx.item() not in random_sus_idx]
    # else:
    #     transform_train = Compose([
    #         RandomCrop(32, padding=4),
    #         RandomHorizontalFlip(),
    #     ])
    #     transform_sa = Compose([
    #         RandomHorizontalFlip(),
    #     ])
        
    #     if attack == 'narcissus' or attack == 'lc':
    #         target_images = [transform_train(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
    #         print("attack: ", attack, "Length of target images: ", len(target_images), "with transform")
    #     elif attack == 'sa':
    #         target_images = [transform_sa(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
    #         print("attack: ", attack, "Length of target images: ", len(target_images), "with transform")
    #     else:
    #         target_images = [img for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
    #         print("attack: ", attack, "Length of target images: ", len(target_images), "without transform")
    #     print(len(target_images))
    #     target_images = torch.stack(target_images).to(device)
    
    
    sur_model = copy.deepcopy(model).to(device)
    if opt == 'sgd':
        sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt == 'adam':
        sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
    
    len_model_params = len(np.concatenate([model.state_dict()[layer].cpu().numpy().flatten() for layer in model.state_dict()]))
    
    
    flattened_weights = torch.cat([param.flatten() for param in sur_model.state_dict().values()]) 
    
    clean_params_dict = {}
    sus_params_dict = {}
    
    torch.save(optimizer.state_dict(), f"temp_folder/temp_optimizer_state_dict_{random_num}.pth")
    
    sur_optimizer_state = torch.load(f"temp_folder/temp_optimizer_state_dict_{random_num}.pth")
    sur_optimizer.load_state_dict(sur_optimizer_state)
    
    os.remove(f"temp_folder/temp_optimizer_state_dict_{random_num}.pth")
        
    def optimize_weight_differences(pos_model_state_dict, original_weights, important_features):
        # Extract the keys once to avoid repeated dict key access
        state_dict_keys = list(pos_model_state_dict.keys())
        
        # Preallocate a list for the differences
        differences = []
        
        # Compute the differences and flatten them in a single loop
        for name in state_dict_keys:
            diff = (pos_model_state_dict[name] - original_weights[name]).cpu().numpy().flatten()
            differences.append(diff)
        
        # Concatenate all differences into a single array
        differences_array = np.concatenate(differences)
        
        # Select important features
        optimized_result = differences_array[important_features]
        
        return optimized_result
    
    
    
    
    for epoch in tqdm(range(training_epochs)):
        clean_params = np.array([-np.inf] * len_model_params)
        sus_params = np.array([-np.inf] * len_model_params)
        print("Length of clean params: ", len(clean_params))

        # Train
        model.to(device) 
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        step1_time_avg = 0
        step2_time_avg = 0
        step3_time_avg = 0
        step_4_1_time_avg = 0
        step_4_2_time_avg = 0
        step_4_3_time_avg = 0
        step_4_4_time_avg = 0
        step_4_5_time_avg = 0
        step5_time_avg = 0
        step5_1_time_avg = 0
        step6_time_avg = 0
        step7_time_avg = 0
        

        
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device), labels.to(device), indices                       
            
            start_time = time.time()
            
            original_weights = copy.deepcopy(model.state_dict())
            
            model.eval()                     
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            model.train(mode = training_mode)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
            step1_time = time.time() - start_time
            start_time = time.time()
            step1_time_avg += step1_time
            
            # torch_rng_state = torch.get_rng_state()
            # np_rng_state = np.random.get_state()
            # python_rng_state = random.getstate()
            
            images = images.clone()
            labels = labels.clone()
            indices = indices.clone()
    

            
            pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
            if len(pos_indices) > 0:
                step2_time = time.time() - start_time
                start_time = time.time()
                step2_time_avg += step2_time
                
                if not sample_from_test:
                    target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
                    available_indices = list(set(target_indices_batch) - set(pos_indices))
                    # print("Length of available indices: ", len(available_indices), "Length of pos indices: ", len(pos_indices), "Length of target indices: ", len(target_indices))
                    if len(available_indices) < len(pos_indices):
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices = available_indices 
                    else:
                        extra_clean_indices = []
                        clean_indices = random.sample(available_indices, len(pos_indices)) 
                    available_indices = list(set(target_indices_batch) - set(pos_indices) - set(clean_indices))
                    if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices_2 = available_indices
                    elif len(available_indices) == 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
                        clean_indices_2 = []
                    else:
                        extra_clean_indices_2 = []
                        clean_indices_2 = random.sample(available_indices, len(pos_indices))
                    
                    remaining_indices = list(set(range(len(indices))) - set(pos_indices) - set(clean_indices) - set(clean_indices_2))
                    assert set(indices[remaining_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices_2]) & set(random_sus_idx) == set()

                    assert set(indices[pos_indices].cpu().numpy()) & set(random_sus_idx) == set(indices[pos_indices].cpu().numpy())
                    assert np.all(labels[clean_indices].cpu().numpy() == target_class)
                    assert np.all(labels[clean_indices_2].cpu().numpy() == target_class)
                    
                else:
                    clean_indices = list(random.sample(range(len(target_images)), len(pos_indices)))
                    clean_indices_2 = list(random.sample(set(range(len(target_images))) - set(clean_indices), len(pos_indices)))
                    remaining_indices =  list(set(range(len(indices))) - set(pos_indices))
                    assert set(indices[remaining_indices].cpu().numpy()) | set(indices[pos_indices].cpu().numpy()) == set(indices.cpu().numpy()) and set(indices[remaining_indices].cpu().numpy()) & set(indices[pos_indices].cpu().numpy()) == set()
                
                
                step3_time = time.time() - start_time
                start_time = time.time()
                step3_time_avg += step3_time
                
                
                step_4_1_time = time.time() - start_time
                step_4_1_time_avg += step_4_1_time
                start_time = time.time()
                sur_model.train(mode = training_mode)                   

                
                sur_model.load_state_dict(original_weights)
                sur_optimizer.load_state_dict(sur_optimizer_state)
            
                
                sur_optimizer.zero_grad()
                
                step_4_2_time = time.time() - start_time
                step_4_2_time_avg += step_4_2_time
                start_time = time.time()
                
                
                output = sur_model(images[pos_indices])
                    
                pred_labels = output.argmax(dim=1)
                
                loss = criterion(output, labels[pos_indices])
                loss.backward()
                sur_optimizer.step()
                
                step_4_3_time = time.time() - start_time 
                step_4_3_time_avg += step_4_3_time
                start_time = time.time()
                
                
                
                sur_model_state_dict = sur_model.state_dict()
                # if attack != 'ht':
                temp_sus = torch.cat([
                    (sur_model_state_dict[key] - original_weights[key]).view(-1)
                    for key in sur_model_state_dict
                ]).cpu().numpy()
                # else:
                    # temp_sus = torch.cat([diff.view(-1) for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy() - torch.cat([diff.view(-1) for name, diff in original_weights.items() if '20' in name]).cpu().numpy()
                
                
                step4_4_time = time.time() - start_time
                step_4_4_time_avg += step4_4_time
                start_time = time.time()
                
                
                
                sur_model.load_state_dict(original_weights)
                sur_optimizer.load_state_dict(sur_optimizer_state)
                
                
                sur_optimizer.zero_grad()

                if not sample_from_test: 
                    if len(extra_clean_indices) > 0:
                        clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
                        clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                    else:
                        clean_batch = images[clean_indices]
                        clean_labels = labels[clean_indices ]
                else:
                    clean_batch = target_images[clean_indices]
                    clean_labels =  torch.tensor([target_class] * len(clean_indices)).to(device)

                # if not sample_from_test: 
                #     if len(extra_clean_indices) > 0:
                #         clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
                #         clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                #     else:
                #         clean_batch = images[clean_indices]
                #         clean_labels = labels[clean_indices]
                # else:
                #     clean_batch = target_images[clean_indices]
                #     clean_batch = torch.tensor([target_class] * len(clean_indices)).to(device)


                sur_model.eval()  
                output = sur_model(clean_batch)
                sur_model.train(mode = training_mode)  
            
                    
                    
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
                sur_model_state_dict = sur_model.state_dict()
                # if attack != 'ht':
                temp_clean = torch.cat([
                    (sur_model_state_dict[key] - original_weights[key]).view(-1)
                    for key in sur_model_state_dict
                ]).cpu().numpy()
                # else:
                #     temp_clean = torch.cat([diff.view(-1) for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy()  - torch.cat([diff.view(-1) for name, diff in original_weights.items() if '20' in name]).cpu().numpy()



                step5_time = time.time() - start_time
                start_time = time.time()
                step5_time_avg += step5_time
                
                
                sur_model.load_state_dict(original_weights)
                sur_optimizer.load_state_dict(sur_optimizer_state)
                
                sur_optimizer.zero_grad()

                start_time_2 = time.time()                
                if not sample_from_test: 
                    if len(extra_clean_indices_2) > 0:
                        clean_batch = torch.cat([images[clean_indices_2], target_images[extra_clean_indices_2]])
                        clean_labels = torch.cat([labels[clean_indices_2], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                    else:
                        clean_batch = images[clean_indices_2]
                        clean_labels = labels[clean_indices_2]
                else:
                    clean_batch =  target_images[clean_indices_2]
                    clean_labels = torch.tensor([target_class] * len(clean_indices_2)).to(device)
                
                # if not sample_from_test: 
                #     if len(extra_clean_indices_2) > 0:
                #         clean_batch = torch.cat([images[clean_indices_2], target_images[extra_clean_indices_2]])
                #         clean_labels = torch.cat([labels[clean_indices_2], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                #     else:
                #         clean_batch = images[clean_indices_2]
                #         clean_labels = labels[clean_indices_2]
                # else:
                #     clean_batch = target_images[clean_indices_2]
                #     clean_labels = torch.tensor([target_class] * len(clean_indices_2)).to(device)

                step5_1_time = time.time() - start_time_2
                step5_1_time_avg += step5_1_time
                
                sur_model.train(mode = training_mode)    
                output = sur_model(clean_batch)
                
                    
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
                sur_model_state_dict = sur_model.state_dict()
                # if attack != 'ht':
                temp_clean_2 = torch.cat([
                        (sur_model_state_dict[key] - original_weights[key]).view(-1)
                        for key in sur_model_state_dict.keys()
                ]).cpu().numpy()

                # else:
                #     temp_clean_2 = torch.cat([diff.view(-1) for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy() - torch.cat([diff.view(-1) for name, diff in original_weights.items() if '20' in name]).cpu().numpy()
            
                
                step6_time = time.time() - start_time   
                step6_time_avg += step6_time 
                start_time = time.time()
                
                sus_params = np.maximum(sus_params, np.abs(temp_sus - temp_clean_2))
                clean_params = np.maximum(clean_params, np.abs(temp_clean - temp_clean_2))
                
                step7_time = time.time() - start_time
                step7_time_avg += step7_time
                step7_time_avg += step7_time
                
                del temp_sus, temp_clean, temp_clean_2
                
                

        # torch.set_rng_state(torch_rng_state)
        # np.random.set_state(np_rng_state)
        # random.setstate(python_rng_state)

        
        
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        # if opt == 'sgd':
        #     scheduler.step()

        
        start_time = time.time()
        
        
        
        step7_time = time.time() - start_time
        step7_time_avg += step7_time
        
        print("Step 1 time:", step1_time_avg)
        print("Step 2 time:", step2_time_avg)
        print("Step 3 time:", step3_time_avg)
        print("Step 4_1 time:", step_4_1_time_avg)
        print("Step 4_2 time:", step_4_2_time_avg)
        print("Step 4_3 time:", step_4_3_time_avg)
        print("Step 4_4 time:", step_4_4_time_avg)
        print("Step 4_5 time:", step_4_5_time_avg)
        print("Step 5 time:", step5_time_avg)
        print("Step 5_1 time:", step5_1_time_avg)
        print("Step 6 time:", step6_time_avg)
        print("Step 7 time:", step7_time_avg)
        
        
        
        if type(poisoned_test_loader) == dict:
            for attack_name in poisoned_test_loader:
                print(f"Testing attack effect for {attack_name}")
                model.eval()
                correct, total = 0, 0
                for i, (images, labels) in enumerate(poisoned_test_loader[attack_name]):
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        logits = model(images)
                        out_loss = criterion(logits,labels)
                        _, predicted = torch.max(logits.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                acc = correct / total
                test_ACC.append(acc)
                print('\nAttack success rate %.2f' % (acc*100))
                print('Test_loss:',out_loss)
        else:
            # Testing attack effect
            model.eval()
            correct, total = 0, 0
            for i, (images, labels) in enumerate(poisoned_test_loader):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    logits = model(images)
                    out_loss = criterion(logits,labels)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            test_ACC.append(acc)
            print('\nAttack success rate %.2f' % (acc*100))
            print('Test_loss:',out_loss)
        
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:',out_loss)

        differences = np.abs(sus_params - clean_params)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        z_scores = (differences - mean_diff) / std_diff
        cut_num = len(np.where(np.abs(z_scores) > k)[0])
        important_features = np.argsort(np.abs(z_scores))[::-1][:cut_num]
        print("Number of important features:", len(important_features), "Epoch:", epoch)
        if epoch == 0:
            important_features_avg = important_features
        else:
            important_features_avg = np.intersect1d(important_features_avg, important_features)
            print("Number of important features after intersection:", len(important_features_avg), "Epoch:", epoch)
        plt.figure()
        plt.scatter(range(sus_params.shape[0]), z_scores, label='Z Scores', alpha=0.5, color='blue')
        plt.savefig(figure_path + f"Max_diff.png")
        
        plt.figure()
        plt.scatter(range(sus_params.shape[0]), differences, label='Differences', alpha=0.5, color='red')
        plt.savefig(figure_path + f"differences.png")
    return important_features_avg




def capture_batch_sample_level_weight_updates(random_sus_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, important_features, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    sus_diff = {}
    clean_diff = {}
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    random_sus_idx = set(random_sus_idx)
    
    target_indices_class = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl == target_class]
    print(len(set(target_indices_class) & random_sus_idx))
    assert set(target_indices_class) & random_sus_idx == random_sus_idx
    if not sample_from_test:
        target_images = [img for imgs, lbls, idxs in poisoned_train_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
        target_images = torch.stack(target_images).to(device)
        target_indices = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl.item() == target_class and idx.item() not in random_sus_idx]
    else:
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ])
        
        transform_sa = Compose([
            RandomHorizontalFlip(),
        ])
        
        
        if attack == 'narcissus' or attack == 'lc':
            target_images = [transform_train(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]   
        # elif attack == 'sa':
        #     target_images = [transform_sa(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
        else:
            target_images = [transform_sa(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]   
            
        print(len(target_images))
        target_images = torch.stack(target_images).to(device)
    
    
    sur_model = copy.deepcopy(model)
    if opt == 'sgd':
        sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt == 'adam':
        sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
        
    flattened_weights = torch.cat([param.flatten() for param in sur_model.state_dict().values()])
    
    layer_mapping = {}
    current_index = 0

    for layer_name, param in sur_model.state_dict().items():
        param_size = param.numel()
        # Identify important features within the current parameter's range
        relevant_indices = [idx - current_index for idx in important_features if current_index <= idx < current_index + param_size]

        if relevant_indices:
            layer_mapping[layer_name] = relevant_indices

        current_index += param_size
        
    
    def optimize_weight_differences(pos_model_state_dict, original_weights, important_features):
        # Extract the keys once to avoid repeated dict key access
        state_dict_keys = list(pos_model_state_dict.keys())
        
        # Preallocate a list for the differences
        differences = []
        
        # Compute the differences and flatten them in a single loop
        for name in state_dict_keys:
            diff = (pos_model_state_dict[name] - original_weights[name]).cpu().numpy().flatten()
            differences.append(diff)
        
        # Concatenate all differences into a single array
        differences_array = np.concatenate(differences)
        
        # Select important features
        optimized_result = differences_array[important_features]
        
        return optimized_result
    
    
    for epoch in tqdm(range(training_epochs)):
        # Train
        model.to(device)  
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        
        step1_time_avg = 0
        step1_2_time_avg = 0
        step2_time_avg = 0
        step3_time_avg = 0
        step_4_1_time_avg = 0
        step_4_2_time_avg = 0
        step_4_3_time_avg = 0
        step_4_4_time_avg = 0
        step_4_5_time_avg = 0
        step5_time_avg = 0
        step5_1_time_avg = 0
        step6_time_avg = 0
        step7_time_avg = 0
    
        
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device), labels.to(device), indices                       
            
            start_time = time.time()
            
            original_weights = copy.deepcopy(model.state_dict())
            if opt == 'sgd':
                # torch.save(optimizer.state_dict(), f'./temp_folder/temp_optimizer_{random_num}.pth')
                current_lr = optimizer.param_groups[0]['lr']
                momentum = optimizer.param_groups[0]['momentum']
                weight_decay = optimizer.param_groups[0]['weight_decay']
                sur_optimizer.param_groups[0]['lr'] = current_lr
                sur_optimizer.param_groups[0]['momentum'] = momentum
                sur_optimizer.param_groups[0]['weight_decay'] = weight_decay
                
                
                # torch.save(model.state_dict(), f'./temp_folder/temp_weights_{random_num}.pth')
            # elif opt == 'adam':
            #     torch.save(model.state_dict(), f'./temp_folder/temp_weights_{random_num}.pth')
            
            step1_time = time.time() - start_time
            start_time = time.time()
            step1_time_avg += step1_time
            
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
            step1_2_time = time.time() - start_time
            step1_2_time_avg  += step1_2_time
            start_time = time.time()

            pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
            if len(pos_indices) > 0:
                step2_time = time.time() - start_time
                start_time = time.time()
                step2_time_avg += step2_time
                
                if not sample_from_test:
                    target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
                    available_indices = list(set(target_indices_batch) - set(pos_indices))
                    # print("Length of available indices: ", len(available_indices), "Length of pos indices: ", len(pos_indices), "Length of target indices: ", len(target_indices))
                    if len(available_indices) < len(pos_indices):
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices = available_indices 
                    else:
                        extra_clean_indices = []
                        clean_indices = random.sample(available_indices, len(pos_indices)) 
                    available_indices = list(set(target_indices_batch) - set(pos_indices) - set(clean_indices))
                    if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices_2 = available_indices
                    elif len(available_indices) == 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
                        clean_indices_2 = []
                    else:
                        extra_clean_indices_2 = []
                        clean_indices_2 = random.sample(available_indices, len(pos_indices))
                    
                    remaining_indices = list(set(range(len(indices))) - set(pos_indices) - set(clean_indices) - set(clean_indices_2))
                    assert set(indices[remaining_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices_2]) & set(random_sus_idx) == set()

                    assert set(indices[pos_indices].cpu().numpy()) & set(random_sus_idx) == set(indices[pos_indices].cpu().numpy())
                    assert np.all(labels[clean_indices].cpu().numpy() == target_class)
                    assert np.all(labels[clean_indices_2].cpu().numpy() == target_class)
                    
                else:
                    clean_indices = list(random.sample(range(len(target_images)), len(pos_indices)))
                    clean_indices_2 = list(random.sample(set(range(len(target_images))) - set(clean_indices), len(pos_indices)))
                    remaining_indices = [i for i, ind in enumerate(indices) if ind.item() not in random_sus_idx]
                    assert set(indices[remaining_indices].cpu().numpy()) | set(indices[pos_indices].cpu().numpy()) == set(indices.cpu().numpy()) and set(indices[remaining_indices].cpu().numpy()) & set(indices[pos_indices].cpu().numpy()) == set()
                
                
                step3_time = time.time() - start_time
                start_time = time.time()
                step3_time_avg += step3_time
                # if opt == 'sgd':
                    # original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                    # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    # sur_optimizer.load_state_dict(original_optimizer)
                # elif opt == 'adam':
                #     original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                #     sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
                
                step_4_1_time = time.time() - start_time
                step_4_1_time_avg += step_4_1_time
                start_time = time.time()
                # original_weights = copy.deepcopy(original_weights_)
                sur_model.load_state_dict(original_weights)
            
                sur_optimizer.zero_grad()
                
                step_4_2_time = time.time() - start_time
                step_4_2_time_avg += step_4_2_time
                start_time = time.time()
                
                
                output = sur_model(images[pos_indices])
                pred_labels = output.argmax(dim=1)
                
                loss = criterion(output, labels[pos_indices])
                loss.backward()
                sur_optimizer.step()
                
                step_4_3_time = time.time() - start_time 
                step_4_3_time_avg += step_4_3_time
                start_time = time.time()
                
                
                
                sur_model_state_dict = sur_model.state_dict()

                if attack != "ht":
                    temp_sus = []
                    for layer_name, indices_ in layer_mapping.items():
                        # Get the layer weights for both current and original models
                        sur_layer_weights = sur_model_state_dict[layer_name]
                        orig_layer_weights = original_weights[layer_name]

                        # Access the important indices and compute the difference
                        important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                        temp_sus.append(important_diff)

                    # Concatenate all the differences and convert to numpy
                    temp_sus = torch.cat(temp_sus).cpu().numpy()

                else:
                    temp_sus = []
                    for layer_name, indices_ in layer_mapping.items():
                        if '20' in layer_name:  # Filter tensors with '20' in their names
                            sur_layer_weights = sur_model_state_dict[layer_name]
                            orig_layer_weights = original_weights[layer_name]

                            # Access the important indices and compute the difference
                            important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                            temp_sus.append(important_diff)

                    temp_sus = torch.cat(temp_sus).cpu().numpy()

                step4_4_time = time.time() - start_time
                step_4_4_time_avg += step4_4_time
                start_time = time.time()
                                
                
                # if opt == 'sgd':
                #     original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                #     # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                #     sur_optimizer.load_state_dict(original_optimizer)
                # elif opt == 'adam':
                #     # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                #     sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
                # original_weights = copy.deepcopy(original_weights_)
                sur_model.load_state_dict(original_weights)
                sur_model.train(mode = training_mode)
                
                sur_optimizer.zero_grad()

                # if not sample_from_test: 
                #     if len(extra_clean_indices) > 0:
                #         clean_batch = torch.cat([images[clean_indices + remaining_indices], target_images[extra_clean_indices]])
                #         clean_labels = torch.cat([labels[clean_indices + remaining_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                #     else:
                #         clean_batch = images[clean_indices + remaining_indices]
                #         clean_labels = labels[clean_indices + remaining_indices]
                # else:
                #     clean_batch = torch.cat([images[remaining_indices], target_images[clean_indices]])
                #     clean_labels = torch.cat([labels[remaining_indices], torch.tensor([target_class] * len(clean_indices)).to(device)])

                if not sample_from_test: 
                    if len(extra_clean_indices) > 0:
                        clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
                        clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                    else:
                        clean_batch = images[clean_indices]
                        clean_labels = labels[clean_indices]
                else:
                    clean_batch = target_images[clean_indices]
                    clean_labels = torch.tensor([target_class] * len(clean_indices)).to(device)
                
                output = sur_model(clean_batch) 
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
               # Obtain the state dictionary of the surrogate model
                sur_model_state_dict = sur_model.state_dict()

                if attack != "ht":
                    temp_clean = []
                    for layer_name, indices_ in layer_mapping.items():
                        sur_layer_weights = sur_model_state_dict[layer_name]
                        orig_layer_weights = original_weights[layer_name]

                        # Access the important indices and compute the difference
                        important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                        temp_clean.append(important_diff)

                    temp_clean = torch.cat(temp_clean).cpu().numpy()

                else:
                    temp_clean = []
                    for layer_name, indices_ in layer_mapping.items():
                        if '20' in layer_name:
                            sur_layer_weights = sur_model_state_dict[layer_name]
                            orig_layer_weights = original_weights[layer_name]

                            important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                            temp_clean.append(important_diff)

                    temp_clean = torch.cat(temp_clean).cpu().numpy()

                step5_time = time.time() - start_time
                start_time = time.time()
                step5_time_avg += step5_time
                
                
                sur_model.train(mode = training_mode)
                
                # if opt == 'sgd':
                #     original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                #     # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                #     sur_optimizer.load_state_dict(original_optimizer)
                # elif opt == 'adam':
                #     # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                #     sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
                # original_weights = copy.deepcopy(original_weights_)
                sur_model.load_state_dict(original_weights)
                sur_optimizer.zero_grad()

                # start_time_2 = time.time()                
                # if not sample_from_test: 
                #     if len(extra_clean_indices_2) > 0:
                #         clean_batch = torch.cat([images[clean_indices_2 + remaining_indices], target_images[extra_clean_indices_2]])
                #         clean_labels = torch.cat([labels[clean_indices_2 + remaining_indices], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                #     else:
                #         clean_batch = images[clean_indices_2 + remaining_indices]
                #         clean_labels = labels[clean_indices_2 + remaining_indices]
                # else:
                #     clean_batch = torch.cat([images[remaining_indices], target_images[clean_indices_2]])
                #     clean_labels = torch.cat([labels[remaining_indices], torch.tensor([target_class] * len(clean_indices_2)).to(device)])
                    

                start_time_2 = time.time()                
                if not sample_from_test: 
                    if len(extra_clean_indices_2) > 0:
                        clean_batch = torch.cat([images[clean_indices_2], target_images[extra_clean_indices_2]])
                        clean_labels = torch.cat([labels[clean_indices_2], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                    else:
                        clean_batch = images[clean_indices_2]
                        clean_labels = labels[clean_indices_2]
                else:
                    clean_batch =  target_images[clean_indices_2]
                    clean_labels = torch.tensor([target_class] * len(clean_indices_2)).to(device)
                    

                step5_1_time = time.time() - start_time_2
                step5_1_time_avg += step5_1_time
                
                output = sur_model(clean_batch)
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
                sur_model_state_dict = sur_model.state_dict()

                if attack != "ht":
                    temp_clean_2 = []
                    for layer_name, indices_ in layer_mapping.items():
                        sur_layer_weights = sur_model_state_dict[layer_name]
                        orig_layer_weights = original_weights[layer_name]

                        important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                        temp_clean_2.append(important_diff)

                    temp_clean_2 = torch.cat(temp_clean_2).cpu().numpy()

                else:
                    temp_clean_2 = []
                    for layer_name, indices_ in layer_mapping.items():
                        if '20' in layer_name:
                            sur_layer_weights = sur_model_state_dict[layer_name]
                            orig_layer_weights = original_weights[layer_name]

                            important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                            temp_clean_2.append(important_diff)

                    temp_clean_2 = torch.cat(temp_clean_2).cpu().numpy()

                step6_time = time.time() - start_time
                step6_time_avg += step6_time
                start_time = time.time()
                
                sus_diff[(epoch, tuple(indices[pos_indices].numpy()))] = temp_sus - temp_clean_2
                if not sample_from_test:
                    if len(extra_clean_indices) > 0:
                        clean_idxs = np.concatenate([indices[clean_indices].numpy() , np.array(target_indices)[extra_clean_indices]])
                        clean_diff[(epoch, tuple(clean_idxs))] = temp_clean - temp_clean_2
                    else:
                        clean_diff[(epoch, tuple(indices[clean_indices].numpy()))] = temp_clean - temp_clean_2
                else:
                    clean_diff[(epoch, tuple(np.array(range(len(target_images)))[clean_indices] + 50000))] = temp_clean - temp_clean_2
                
                
                step7_time = time.time() - start_time
                step7_time_avg += step7_time
                step7_time_avg += step7_time
                
                del temp_sus, temp_clean, temp_clean_2
                
                

        torch.cuda.empty_cache()       
        
        
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

        
        start_time = time.time()
        
        
        
        step7_time = time.time() - start_time
        step7_time_avg += step7_time
        
        print("Step 1 time:", step1_time_avg)
        print("Step 1_2 time:", step1_2_time_avg)
        print("Step 2 time:", step2_time_avg)
        print("Step 3 time:", step3_time_avg)
        print("Step 4_1 time:", step_4_1_time_avg)
        print("Step 4_2 time:", step_4_2_time_avg)
        print("Step 4_3 time:", step_4_3_time_avg)
        print("Step 4_4 time:", step_4_4_time_avg)
        print("Step 4_5 time:", step_4_5_time_avg)
        print("Step 5 time:", step5_time_avg)
        print("Step 5_1 time:", step5_1_time_avg)
        print("Step 6 time:", step6_time_avg)
        print("Step 7 time:", step7_time_avg)
        
        
         
        if type(poisoned_test_loader) == dict:
            for attack_name in poisoned_test_loader:
                print(f"Testing attack effect for {attack_name}")
                model.eval()
                correct, total = 0, 0
                for i, (images, labels) in enumerate(poisoned_test_loader[attack_name]):
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        logits = model(images)
                        out_loss = criterion(logits,labels)
                        _, predicted = torch.max(logits.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                acc = correct / total
                test_ACC.append(acc)
                print('\nAttack success rate %.2f' % (acc*100))
                print('Test_loss:',out_loss)
        else:
            # Testing attack effect
            model.eval()
            correct, total = 0, 0
            for i, (images, labels) in enumerate(poisoned_test_loader):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    logits = model(images)
                    out_loss = criterion(logits,labels)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            test_ACC.append(acc)
            print('\nAttack success rate %.2f' % (acc*100))
            print('Test_loss:',out_loss)
            
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:',out_loss)
    
    sus_inds = [ind for epoch, ind in sus_diff]
    clean_inds = [ind for epoch, ind in clean_diff]
    sus_diff = np.array([sus_diff[key] for key in sus_diff])
    clean_diff = np.array([clean_diff[key] for key in clean_diff])
    
    return sus_diff, clean_diff, sus_inds, clean_inds


def capture_batch_sample_level_weight_updates(random_sus_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, important_features, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    sus_diff = {}
    clean_diff = {}
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    random_sus_idx = set(random_sus_idx)
    
    target_indices_class = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl == target_class]
    print(len(set(target_indices_class) & random_sus_idx))
    assert set(target_indices_class) & random_sus_idx == random_sus_idx
    if not sample_from_test:
        target_images = [img for imgs, lbls, idxs in poisoned_train_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
        target_images = torch.stack(target_images).to(device)
        target_indices = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl.item() == target_class and idx.item() not in random_sus_idx]
    else:
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ])
        
        transform_sa = Compose([
            RandomHorizontalFlip(),
        ])
        
        
        if attack == 'narcissus' or attack == 'lc':
            target_images = [transform_train(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]   
        # elif attack == 'sa':
        #     target_images = [transform_sa(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
        else:
            target_images = [transform_sa(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]   
            
        print(len(target_images))
        target_images = torch.stack(target_images).to(device)
    
    
    sur_model = copy.deepcopy(model)
    if opt == 'sgd':
        sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt == 'adam':
        sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
        
    flattened_weights = torch.cat([param.flatten() for param in sur_model.state_dict().values()])
    
    layer_mapping = {}
    current_index = 0

    for layer_name, param in sur_model.state_dict().items():
        param_size = param.numel()
        # Identify important features within the current parameter's range
        relevant_indices = [idx - current_index for idx in important_features if current_index <= idx < current_index + param_size]

        if relevant_indices:
            layer_mapping[layer_name] = relevant_indices

        current_index += param_size
        
    
    def optimize_weight_differences(pos_model_state_dict, original_weights, important_features):
        # Extract the keys once to avoid repeated dict key access
        state_dict_keys = list(pos_model_state_dict.keys())
        
        # Preallocate a list for the differences
        differences = []
        
        # Compute the differences and flatten them in a single loop
        for name in state_dict_keys:
            diff = (pos_model_state_dict[name] - original_weights[name]).cpu().numpy().flatten()
            differences.append(diff)
        
        # Concatenate all differences into a single array
        differences_array = np.concatenate(differences)
        
        # Select important features
        optimized_result = differences_array[important_features]
        
        return optimized_result
    
    
    for epoch in tqdm(range(training_epochs)):
        # Train
        model.to(device)  
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        
        step1_time_avg = 0
        step1_2_time_avg = 0
        step2_time_avg = 0
        step3_time_avg = 0
        step_4_1_time_avg = 0
        step_4_2_time_avg = 0
        step_4_3_time_avg = 0
        step_4_4_time_avg = 0
        step_4_5_time_avg = 0
        step5_time_avg = 0
        step5_1_time_avg = 0
        step6_time_avg = 0
        step7_time_avg = 0
    
        
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device), labels.to(device), indices                       
            
            start_time = time.time()
            
            original_weights = copy.deepcopy(model.state_dict())
            if opt == 'sgd':
                # torch.save(optimizer.state_dict(), f'./temp_folder/temp_optimizer_{random_num}.pth')
                current_lr = optimizer.param_groups[0]['lr']
                momentum = optimizer.param_groups[0]['momentum']
                weight_decay = optimizer.param_groups[0]['weight_decay']
                sur_optimizer.param_groups[0]['lr'] = current_lr
                sur_optimizer.param_groups[0]['momentum'] = momentum
                sur_optimizer.param_groups[0]['weight_decay'] = weight_decay
                
                
                # torch.save(model.state_dict(), f'./temp_folder/temp_weights_{random_num}.pth')
            # elif opt == 'adam':
            #     torch.save(model.state_dict(), f'./temp_folder/temp_weights_{random_num}.pth')
            
            step1_time = time.time() - start_time
            start_time = time.time()
            step1_time_avg += step1_time
            
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
            step1_2_time = time.time() - start_time
            step1_2_time_avg  += step1_2_time
            start_time = time.time()

            pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
            if len(pos_indices) > 0:
                step2_time = time.time() - start_time
                start_time = time.time()
                step2_time_avg += step2_time
                
                if not sample_from_test:
                    target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
                    available_indices = list(set(target_indices_batch) - set(pos_indices))
                    # print("Length of available indices: ", len(available_indices), "Length of pos indices: ", len(pos_indices), "Length of target indices: ", len(target_indices))
                    if len(available_indices) < len(pos_indices):
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices = available_indices 
                    else:
                        extra_clean_indices = []
                        clean_indices = random.sample(available_indices, len(pos_indices)) 
                    available_indices = list(set(target_indices_batch) - set(pos_indices) - set(clean_indices))
                    if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices_2 = available_indices
                    elif len(available_indices) == 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
                        clean_indices_2 = []
                    else:
                        extra_clean_indices_2 = []
                        clean_indices_2 = random.sample(available_indices, len(pos_indices))
                    
                    remaining_indices = list(set(range(len(indices))) - set(pos_indices) - set(clean_indices) - set(clean_indices_2))
                    assert set(indices[remaining_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices_2]) & set(random_sus_idx) == set()

                    assert set(indices[pos_indices].cpu().numpy()) & set(random_sus_idx) == set(indices[pos_indices].cpu().numpy())
                    assert np.all(labels[clean_indices].cpu().numpy() == target_class)
                    assert np.all(labels[clean_indices_2].cpu().numpy() == target_class)
                    
                else:
                    clean_indices = list(random.sample(range(len(target_images)), len(pos_indices)))
                    clean_indices_2 = list(random.sample(set(range(len(target_images))) - set(clean_indices), len(pos_indices)))
                    remaining_indices = [i for i, ind in enumerate(indices) if ind.item() not in random_sus_idx]
                    assert set(indices[remaining_indices].cpu().numpy()) | set(indices[pos_indices].cpu().numpy()) == set(indices.cpu().numpy()) and set(indices[remaining_indices].cpu().numpy()) & set(indices[pos_indices].cpu().numpy()) == set()
                
                
                step3_time = time.time() - start_time
                start_time = time.time()
                step3_time_avg += step3_time
                # if opt == 'sgd':
                    # original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                    # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    # sur_optimizer.load_state_dict(original_optimizer)
                # elif opt == 'adam':
                #     original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                #     sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
                
                step_4_1_time = time.time() - start_time
                step_4_1_time_avg += step_4_1_time
                start_time = time.time()
                # original_weights = copy.deepcopy(original_weights_)
                sur_model.load_state_dict(original_weights)
            
                sur_optimizer.zero_grad()
                
                step_4_2_time = time.time() - start_time
                step_4_2_time_avg += step_4_2_time
                start_time = time.time()
                
                
                output = sur_model(images[pos_indices])
                pred_labels = output.argmax(dim=1)
                
                loss = criterion(output, labels[pos_indices])
                loss.backward()
                sur_optimizer.step()
                
                step_4_3_time = time.time() - start_time 
                step_4_3_time_avg += step_4_3_time
                start_time = time.time()
                
                
                
                sur_model_state_dict = sur_model.state_dict()

                if attack != "ht":
                    temp_sus = []
                    for layer_name, indices_ in layer_mapping.items():
                        # Get the layer weights for both current and original models
                        sur_layer_weights = sur_model_state_dict[layer_name]
                        orig_layer_weights = original_weights[layer_name]

                        # Access the important indices and compute the difference
                        important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                        temp_sus.append(important_diff)

                    # Concatenate all the differences and convert to numpy
                    temp_sus = torch.cat(temp_sus).cpu().numpy()

                else:
                    temp_sus = []
                    for layer_name, indices_ in layer_mapping.items():
                        if '20' in layer_name:  # Filter tensors with '20' in their names
                            sur_layer_weights = sur_model_state_dict[layer_name]
                            orig_layer_weights = original_weights[layer_name]

                            # Access the important indices and compute the difference
                            important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                            temp_sus.append(important_diff)

                    temp_sus = torch.cat(temp_sus).cpu().numpy()

                step4_4_time = time.time() - start_time
                step_4_4_time_avg += step4_4_time
                start_time = time.time()
                                
                
                # if opt == 'sgd':
                #     original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                #     # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                #     sur_optimizer.load_state_dict(original_optimizer)
                # elif opt == 'adam':
                #     # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                #     sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
                # original_weights = copy.deepcopy(original_weights_)
                sur_model.load_state_dict(original_weights)
                sur_model.train(mode = training_mode)
                
                sur_optimizer.zero_grad()

                # if not sample_from_test: 
                #     if len(extra_clean_indices) > 0:
                #         clean_batch = torch.cat([images[clean_indices + remaining_indices], target_images[extra_clean_indices]])
                #         clean_labels = torch.cat([labels[clean_indices + remaining_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                #     else:
                #         clean_batch = images[clean_indices + remaining_indices]
                #         clean_labels = labels[clean_indices + remaining_indices]
                # else:
                #     clean_batch = torch.cat([images[remaining_indices], target_images[clean_indices]])
                #     clean_labels = torch.cat([labels[remaining_indices], torch.tensor([target_class] * len(clean_indices)).to(device)])

                if not sample_from_test: 
                    if len(extra_clean_indices) > 0:
                        clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
                        clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                    else:
                        clean_batch = images[clean_indices]
                        clean_labels = labels[clean_indices]
                else:
                    clean_batch = target_images[clean_indices]
                    clean_labels = torch.tensor([target_class] * len(clean_indices)).to(device)
                
                output = sur_model(clean_batch) 
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
               # Obtain the state dictionary of the surrogate model
                sur_model_state_dict = sur_model.state_dict()

                if attack != "ht":
                    temp_clean = []
                    for layer_name, indices_ in layer_mapping.items():
                        sur_layer_weights = sur_model_state_dict[layer_name]
                        orig_layer_weights = original_weights[layer_name]

                        # Access the important indices and compute the difference
                        important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                        temp_clean.append(important_diff)

                    temp_clean = torch.cat(temp_clean).cpu().numpy()

                else:
                    temp_clean = []
                    for layer_name, indices_ in layer_mapping.items():
                        if '20' in layer_name:
                            sur_layer_weights = sur_model_state_dict[layer_name]
                            orig_layer_weights = original_weights[layer_name]

                            important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                            temp_clean.append(important_diff)

                    temp_clean = torch.cat(temp_clean).cpu().numpy()

                step5_time = time.time() - start_time
                start_time = time.time()
                step5_time_avg += step5_time
                
                
                sur_model.train(mode = training_mode)
                
                # if opt == 'sgd':
                #     original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                #     # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                #     sur_optimizer.load_state_dict(original_optimizer)
                # elif opt == 'adam':
                #     # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                #     sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
                # original_weights = copy.deepcopy(original_weights_)
                sur_model.load_state_dict(original_weights)
                sur_optimizer.zero_grad()

                # start_time_2 = time.time()                
                # if not sample_from_test: 
                #     if len(extra_clean_indices_2) > 0:
                #         clean_batch = torch.cat([images[clean_indices_2 + remaining_indices], target_images[extra_clean_indices_2]])
                #         clean_labels = torch.cat([labels[clean_indices_2 + remaining_indices], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                #     else:
                #         clean_batch = images[clean_indices_2 + remaining_indices]
                #         clean_labels = labels[clean_indices_2 + remaining_indices]
                # else:
                #     clean_batch = torch.cat([images[remaining_indices], target_images[clean_indices_2]])
                #     clean_labels = torch.cat([labels[remaining_indices], torch.tensor([target_class] * len(clean_indices_2)).to(device)])
                    

                start_time_2 = time.time()                
                if not sample_from_test: 
                    if len(extra_clean_indices_2) > 0:
                        clean_batch = torch.cat([images[clean_indices_2], target_images[extra_clean_indices_2]])
                        clean_labels = torch.cat([labels[clean_indices_2], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                    else:
                        clean_batch = images[clean_indices_2]
                        clean_labels = labels[clean_indices_2]
                else:
                    clean_batch =  target_images[clean_indices_2]
                    clean_labels = torch.tensor([target_class] * len(clean_indices_2)).to(device)
                    

                step5_1_time = time.time() - start_time_2
                step5_1_time_avg += step5_1_time
                
                output = sur_model(clean_batch)
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
                sur_model_state_dict = sur_model.state_dict()

                if attack != "ht":
                    temp_clean_2 = []
                    for layer_name, indices_ in layer_mapping.items():
                        sur_layer_weights = sur_model_state_dict[layer_name]
                        orig_layer_weights = original_weights[layer_name]

                        important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                        temp_clean_2.append(important_diff)

                    temp_clean_2 = torch.cat(temp_clean_2).cpu().numpy()

                else:
                    temp_clean_2 = []
                    for layer_name, indices_ in layer_mapping.items():
                        if '20' in layer_name:
                            sur_layer_weights = sur_model_state_dict[layer_name]
                            orig_layer_weights = original_weights[layer_name]

                            important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                            temp_clean_2.append(important_diff)

                    temp_clean_2 = torch.cat(temp_clean_2).cpu().numpy()

                step6_time = time.time() - start_time
                step6_time_avg += step6_time
                start_time = time.time()
                
                sus_diff[(epoch, tuple(indices[pos_indices].numpy()))] = temp_sus - temp_clean_2
                if not sample_from_test:
                    if len(extra_clean_indices) > 0:
                        clean_idxs = np.concatenate([indices[clean_indices].numpy() , np.array(target_indices)[extra_clean_indices]])
                        clean_diff[(epoch, tuple(clean_idxs))] = temp_clean - temp_clean_2
                    else:
                        clean_diff[(epoch, tuple(indices[clean_indices].numpy()))] = temp_clean - temp_clean_2
                else:
                    clean_diff[(epoch, tuple(np.array(range(len(target_images)))[clean_indices] + 50000))] = temp_clean - temp_clean_2
                
                
                step7_time = time.time() - start_time
                step7_time_avg += step7_time
                step7_time_avg += step7_time
                
                del temp_sus, temp_clean, temp_clean_2
                
                

        torch.cuda.empty_cache()  
        
        os.remove(f"temp_folder/temp_optimizer_state_dict_{random_num}.pth")     
        
        
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

        
        start_time = time.time()
        
        
        
        step7_time = time.time() - start_time
        step7_time_avg += step7_time
        
        print("Step 1 time:", step1_time_avg)
        print("Step 1_2 time:", step1_2_time_avg)
        print("Step 2 time:", step2_time_avg)
        print("Step 3 time:", step3_time_avg)
        print("Step 4_1 time:", step_4_1_time_avg)
        print("Step 4_2 time:", step_4_2_time_avg)
        print("Step 4_3 time:", step_4_3_time_avg)
        print("Step 4_4 time:", step_4_4_time_avg)
        print("Step 4_5 time:", step_4_5_time_avg)
        print("Step 5 time:", step5_time_avg)
        print("Step 5_1 time:", step5_1_time_avg)
        print("Step 6 time:", step6_time_avg)
        print("Step 7 time:", step7_time_avg)
        
        
         
        if type(poisoned_test_loader) == dict:
            for attack_name in poisoned_test_loader:
                print(f"Testing attack effect for {attack_name}")
                model.eval()
                correct, total = 0, 0
                for i, (images, labels) in enumerate(poisoned_test_loader[attack_name]):
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        logits = model(images)
                        out_loss = criterion(logits,labels)
                        _, predicted = torch.max(logits.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                acc = correct / total
                test_ACC.append(acc)
                print('\nAttack success rate %.2f' % (acc*100))
                print('Test_loss:',out_loss)
        else:
            # Testing attack effect
            model.eval()
            correct, total = 0, 0
            for i, (images, labels) in enumerate(poisoned_test_loader):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    logits = model(images)
                    out_loss = criterion(logits,labels)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            test_ACC.append(acc)
            print('\nAttack success rate %.2f' % (acc*100))
            print('Test_loss:',out_loss)
            
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:',out_loss)
    
    sus_inds = [ind for epoch, ind in sus_diff]
    clean_inds = [ind for epoch, ind in clean_diff]
    sus_diff = np.array([sus_diff[key] for key in sus_diff])
    clean_diff = np.array([clean_diff[key] for key in clean_diff])
    
    return sus_diff, clean_diff, sus_inds, clean_inds



class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    


# def capture_sample_level_weight_updates_idv(random_sus_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, important_features, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1): 
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
    
    
#     train_ACC = []
#     test_ACC = []
#     clean_ACC = []
#     target_ACC = []

#     sus_diff = {}
#     clean_diff = {}
#     separate_rng = np.random.default_rng()
#     random_num = separate_rng.integers(1, 10000)
#     random_sus_idx = set(random_sus_idx)
    
#     dataset = poisoned_train_loader.dataset

#     target_indices_class = [dataset[i][2].item() for i in range(len(dataset)) if dataset[i][1] == target_class]
#     print("Number of target indices in the dataset: ", len(target_indices_class))
#     print(len(set(target_indices_class) & random_sus_idx))
#     assert set(target_indices_class) & random_sus_idx == random_sus_idx 
    
    
#     print("shape of dataset: ", dataset[0][0].shape)
#     if not sample_from_test:
#         target_images = [dataset[i][0] for i in range(len(dataset)) if dataset[i][1] == target_class]
#         target_images = torch.stack(target_images).to(device)
#         target_indices = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] == target_class and dataset[i][2] not in random_sus_idx]
#     else:
#         transform_train = Compose([
#             RandomCrop(32, padding=4),
#             RandomHorizontalFlip(),
#         ])
        
#         transform_sa = Compose([
#             RandomHorizontalFlip(),
#         ])
        
#         test_dataset = test_loader.dataset
#         target_images = [test_dataset[i][0].clone().detach() for i in range(len(test_dataset)) if test_dataset[i][1] == target_class]
#         if attack == 'narcissus' or attack == 'lc':
#             target_images = [transform_train(img) for img in target_images]
#         elif attack == 'sa':
#             target_images = [transform_sa(img) for img in target_images]
               
            
#         print(len(target_images))
#         target_images = torch.stack(target_images).to(device)
#         print("target images shape:", target_images.shape)
    
    
#     def compare_weights(current_weights, reference_weights):
#         for key in current_weights:
#             if not torch.equal(current_weights[key], reference_weights[key]):
#                 print(f"Mismatch found in layer: {key}")
#                 return False
#         return True
    
#     separate_rng = np.random.default_rng()
#     random_num = separate_rng.integers(1, 10000)
    
#     sur_model = copy.deepcopy(model)
#     sur_model.to(device)
#     if opt == 'sgd':
#         sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#     elif opt == 'adam':
#         sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
    
#     torch.save(sur_optimizer.state_dict(), f'./temp_folder/temp_optimizer_state_dict_{random_num}.pth')
#     optimizer_state = torch.load(f'./temp_folder/temp_optimizer_state_dict_{random_num}.pth')
#     sur_optimizer.load_state_dict(optimizer_state)
    
#     os.remove(f"temp_folder/temp_optimizer_state_dict_{random_num}.pth")
    
    
#     flattened_weights = torch.cat([param.flatten() for name, param in sur_model.state_dict().items()])
    
#     model.to(device) 

    
#     for epoch in tqdm(range(training_epochs)):
#         # Train
#         model.train(mode = training_mode)
#         acc_meter = AverageMeter()
#         loss_meter = AverageMeter()
#         pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        
        
        
#         step1_time_avg = 0
#         step2_time_avg = 0
#         step3_time_avg = 0
#         step_4_1_time_avg = 0
#         step_4_2_time_avg = 0
#         step_4_3_time_avg = 0
#         step_4_4_time_avg = 0
#         step_4_5_time_avg = 0
#         step5_time_avg = 0
#         step5_1_time_avg = 0
#         step6_time_avg = 0
#         step7_time_avg = 0
    
        
#         for images, labels, indices in pbar:
#             images, labels, indices = images.to(device), labels.to(device), indices                       
            
#             start_time = time.time()
            
#             # torch.save(model.state_dict(), f'./temp_folder/temp_weights_{random_num}.pth')
#             # torch.save(optimizer.state_dict(), f'./temp_folder/temp_optimizer_{random_num}.pth')
#             original_weights = copy.deepcopy(model.state_dict())
#             # sur_optimizer.load_state_dict(optimizer.state_dict()) 

#             step1_time = time.time() - start_time
#             start_time = time.time()
#             step1_time_avg += step1_time
            
#             model.zero_grad()
#             optimizer.zero_grad()
#             logits = model(images)
#             loss = criterion(logits, labels)
#             loss.backward()
#             optimizer.step()
            
#             _, predicted = torch.max(logits.data, 1)
#             acc = (predicted == labels).sum().item()/labels.size(0)
#             acc_meter.update(acc)
#             loss_meter.update(loss.item())
#             pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
            
#             # indices = indices.clone().detach()
#             # images = images.clone().detach()
#             # labels = labels.clone().detach()
          
            
#             temp_sus = {}
#             temp_clean = {}
            

#             pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
#             assert np.all(labels[pos_indices].cpu().numpy() == target_class)
#             if len(pos_indices) > 0:
#                 step2_time = time.time() - start_time
#                 start_time = time.time()
#                 step2_time_avg += step2_time
#                 # org_w = np.concatenate([original_weights[name].cpu().numpy().flatten() for name in original_weights])[important_features]
                
#                 if not sample_from_test:
#                     target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
#                     available_indices = list(set(target_indices_batch) - set(pos_indices))
#                     # print("Length of available indices: ", len(available_indices), "Length of pos indices: ", len(pos_indices), "Length of target indices: ", len(target_indices))
#                     if len(available_indices) < len(pos_indices):
#                         indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
#                         remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
#                         extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
#                         clean_indices = available_indices 
#                     else:
#                         extra_clean_indices = []
#                         clean_indices = random.sample(available_indices, len(pos_indices)) 
#                     available_indices = list(set(target_indices_batch) - set(pos_indices) - set(clean_indices))
#                     if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
#                         indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
#                         remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
#                         extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
#                         clean_indices_2 = available_indices
#                     elif len(available_indices) == 0:
#                         indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
#                         remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
#                         extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
#                         clean_indices_2 = []
#                     else:
#                         extra_clean_indices_2 = []
#                         clean_indices_2 = random.sample(available_indices, len(pos_indices))
                    

#                     assert set(indices[pos_indices].cpu().numpy()) & set(random_sus_idx) == set(indices[pos_indices].cpu().numpy())
#                     assert np.all(labels[clean_indices].cpu().numpy() == target_class)
#                     assert np.all(labels[clean_indices_2].cpu().numpy() == target_class)
                    
#                 else:
#                     clean_indices = list(random.sample(range(len(target_images)), len(pos_indices)))
#                     clean_indices_2 = list(random.sample(set(range(len(target_images))) - set(clean_indices), len(pos_indices)))
                    
#                     assert set(clean_indices) & set(clean_indices_2) == set()
              
                
#                 step3_time = time.time() - start_time
#                 start_time = time.time()
#                 step3_time_avg += step3_time
                
#                 # temp_pos_loader = DataLoader(list(zip(images[pos_indices], labels[pos_indices], pos_indices)), batch_size=1, shuffle=False) 
#                 # for image, label, index in temp_pos_loader:
                    
                    
#                 #     step_4_1_time = time.time() - start_time
#                 #     step_4_1_time_avg += step_4_1_time
#                 #     start_time = time.time()
                
                        
                    
                    
                    
#                 #     sur_model.train(mode = training_mode)
                    
#                 #     sur_model.load_state_dict(original_weights)
                    
#                 #     step_4_2_time = time.time() - start_time
#                 #     step_4_2_time_avg += step_4_2_time
#                 #     start_time = time.time()
                    
#                 #     # true_original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
#                 #     # assert compare_weights(true_original_weights, sur_model.state_dict())
#                 #     sur_optimizer.zero_grad()
                    
#                 #     try:
#                 #         output = sur_model(image)
#                 #     except:
#                 #         sur_model.eval()
#                 #         output = sur_model(image)
#                 #         sur_model.train(mode = training_mode)
                                        
#                 #     loss = criterion(output, label)
#                 #     loss.backward()
#                 #     sur_optimizer.step()
                
#                 #     step_4_3_time = time.time() - start_time 
#                 #     step_4_3_time_avg += step_4_3_time
#                 #     start_time = time.time()
                    
#                 #     sur_model_state_dict = sur_model.state_dict()                    
#                 #     # if attack != 'ht':                        
#                 #     temp_sus[indices[index]] = optimize_weight_differences(sur_model_state_dict, original_weights, important_features)

#                 #     # else:
#                 #     #     temp_sus[indices[index]] = torch.cat([diff.view(-1).contiguous() for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy().copy()[important_features]
                     
#                 #     step4_4_time = time.time() - start_time
#                 #     step_4_4_time_avg += step4_4_time
#                 #     start_time = time.time()
                
                
                

                
#                 # if not sample_from_test: 
#                 #     if len(extra_clean_indices) > 0:
#                 #         clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
#                 #         clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
#                 #         clean_indexes = np.concatenate([indices[clean_indices].cpu().numpy(), np.array(target_indices)[extra_clean_indices]])
#                 #     else:
#                 #         clean_batch = images[clean_indices]
#                 #         clean_labels = labels[clean_indices]
#                 #         clean_indexes = indices[clean_indices].cpu().numpy()
#                 # else:
#                 #     clean_batch = target_images[clean_indices]
#                 #     clean_labels = torch.tensor([target_class] * len(clean_indices)).to(device)
#                 #     clean_indexes = np.array(clean_indices) + 50000
                
#                 # start_time = time.time()

#                 # temp_clean_1_loader = DataLoader(list(zip(clean_batch, clean_labels, clean_indexes)), batch_size=1, shuffle=False)
#                 # for image, label, index in temp_clean_1_loader:
                        
#                 #     sur_model.load_state_dict(original_weights)
#                 #     sur_model.train(mode = training_mode)
                    
#                 #     # true_original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
#                 #     # assert compare_weights(true_original_weights, sur_model.state_dict())
#                 #     sur_optimizer.zero_grad()
                    
                    
#                 #     try:
#                 #         output = sur_model(image)
#                 #     except:
#                 #         sur_model.eval()
#                 #         output = sur_model(image)
#                 #         sur_model.train(mode = training_mode)
                        
#                 #     clean_label = label.long()
#                 #     loss = criterion(output, clean_label)
#                 #     loss.backward()
#                 #     sur_optimizer.step()
                    
#                 #     sur_model_state_dict = sur_model.state_dict()                    
#                 #     # if attack != 'ht':
#                 #     temp_clean[index] = optimize_weight_differences(sur_model_state_dict, original_weights, important_features)
#                 #     # else:
#                 #     #     temp_clean[index] = torch.cat([diff.view(-1).contiguous() for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy().copy()[important_features]
                    
                    



#                 # step5_time = time.time() - start_time
#                 # start_time = time.time()
#                 # step5_time_avg += step5_time
                
                

#                 # start_time_2 = time.time()                
#                 # if not sample_from_test: 
#                 #     if len(extra_clean_indices_2) > 0:
#                 #         clean_batch = torch.cat([images[clean_indices_2 ], target_images[extra_clean_indices_2]])
#                 #         clean_labels = torch.cat([labels[clean_indices_2 ], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
#                 #         clean_indexes = np.concatenate([indices[clean_indices_2].cpu().numpy(), np.array(target_indices)[extra_clean_indices_2]])
#                 #     else:
#                 #         clean_batch = images[clean_indices_2]
#                 #         clean_labels = labels[clean_indices_2]
#                 #         clean_indexes = indices[clean_indices_2].cpu().numpy()
#                 # else:
#                 #     clean_batch = target_images[clean_indices_2]
#                 #     clean_labels = torch.tensor([target_class] * len(clean_indices_2)).to(device)
#                 #     clean_indexes = np.array(clean_indices_2) + 50000

#                 # step5_1_time = time.time() - start_time_2
#                 # step5_1_time_avg += step5_1_time
                
#                 # clean_2_loader = DataLoader(list(zip(clean_batch, clean_labels)), batch_size=1, shuffle=False)
                
#                 # temp_clean_2 = np.zeros(len(important_features))
#                 # batch_count = 0

#                 # for image, label in clean_2_loader:
                    
#                 #     sur_model.load_state_dict(original_weights)
#                 #     sur_model.train(mode=training_mode)
#                 #     # true_original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
#                 #     # assert compare_weights(true_original_weights, sur_model.state_dict())
#                 #     sur_optimizer.zero_grad()
                    
#                 #     try:
#                 #         output = sur_model(image)
#                 #     except:
#                 #         sur_model.eval()
#                 #         output = sur_model(image)
#                 #         sur_model.train(mode = training_mode)
                        
#                 #     clean_label = label.long()
#                 #     loss = criterion(output, clean_label)
                    
#                 #     loss.backward()
#                 #     sur_optimizer.step()
                    
#                 #     sur_model_state_dict = sur_model.state_dict()
                    
#                 #     # Compute differences or feature-specific values based on the attack type
#                 #     # if attack != 'ht':
#                 #     increment = optimize_weight_differences(sur_model_state_dict, original_weights, important_features)
#                 #     # else:
#                 #     #     increment = torch.cat([
#                 #     #         diff.view(-1).contiguous() for name, diff in sur_model_state_dict.items() if '20' in name
#                 #     #     ]).cpu().numpy().copy()[important_features]
                    
#                 #     # Update the running average
#                 #     batch_count += 1
#                 #     temp_clean_2 = temp_clean_2 + (increment - temp_clean_2) / batch_count
                                                
#                 # step6_time = time.time() - start_time
#                 # step6_time_avg += step6_time
#                 # start_time = time.time()
                
#                 # for index in temp_sus:
#                 #     sus_diff[(epoch, index)] = temp_sus[index] - temp_clean_2 

#                 # for index in temp_clean:
#                 #     clean_diff[(epoch, index)] = temp_clean[index] - temp_clean_2
                
#                 # Combine all groups into a single loader
#                 combined_batch = []
#                 combined_labels = []
#                 combined_indexes = []
#                 pos_indexes = indices[pos_indices].cpu().numpy()
#                 # Add suspected samples
#                 combined_batch.extend(images[pos_indices])
#                 combined_labels.extend(labels[pos_indices])
                
                
#                 if not sample_from_test: 
#                     if len(extra_clean_indices) > 0:
#                         clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
#                         clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
#                         clean_indexes = np.concatenate([indices[clean_indices].cpu().numpy(), np.array(target_indices)[extra_clean_indices]])
#                     else:
#                         clean_batch = images[clean_indices]
#                         clean_labels = labels[clean_indices]
#                         clean_indexes = indices[clean_indices].cpu().numpy()
#                 else:
#                     clean_batch = target_images[clean_indices]
#                     clean_labels = torch.tensor([target_class] * len(clean_indices)).to(device)
#                     clean_indexes = np.array(clean_indices) + 50000
                    

#                 # Add clean_1 samples
#                 combined_batch.extend(clean_batch)
#                 combined_labels.extend(clean_labels)
                
                
#                 if not sample_from_test: 
#                     if len(extra_clean_indices_2) > 0:
#                         clean_batch_2 = torch.cat([images[clean_indices_2 ], target_images[extra_clean_indices_2]])
#                         clean_labels_2 = torch.cat([labels[clean_indices_2 ], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
#                         clean_indexes_2 = np.concatenate([indices[clean_indices_2].cpu().numpy(), np.array(target_indices)[extra_clean_indices_2]])
#                     else:
#                         clean_batch_2 = images[clean_indices_2]
#                         clean_labels_2 = labels[clean_indices_2]
#                         clean_indexes_2 = indices[clean_indices_2].cpu().numpy()
#                 else:
#                     clean_batch_2 = target_images[clean_indices_2]
#                     clean_labels_2 = torch.tensor([target_class] * len(clean_indices_2)).to(device)
#                     clean_indexes_2 = np.array(clean_indices_2) + 50000
                
                
#                 tagged_clean_indexes = [(index, 1) for index in clean_indexes]  # Tag `1` for clean_indexes
#                 tagged_clean_indexes_2 = [(index, 2) for index in clean_indexes_2]  # Tag `2` for clean_indexes_2
#                 tagged_pos_indexes = [(index, 0) for index in pos_indexes]  # Tag `0` for suspected samples

#                 combined_indexes = tagged_pos_indexes + tagged_clean_indexes + tagged_clean_indexes_2

#                 # Add clean_2 samples
#                 combined_batch.extend(clean_batch_2)
#                 combined_labels.extend(clean_labels_2)

#                 # Convert to tensors
#                 combined_batch = torch.stack(combined_batch).to(device)
#                 combined_labels = torch.tensor(combined_labels).to(device)

#                 # Create a single DataLoader for all samples
#                 combined_loader = DataLoader(list(zip(combined_batch, combined_labels, combined_indexes)), batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(seed))

#                 # Dictionaries to store differences
#                 temp_sus = {}
#                 temp_clean = {}
#                 temp_clean_2 = np.zeros(len(important_features))  # Running average for clean_2
#                 batch_count_clean_2 = 0

#                 # Iterate through the combined loader
#                 for image, label, (index, tag) in combined_loader:
#                     # Reset surrogate model
#                     sur_model.load_state_dict(original_weights)
#                     sur_model.train(mode=training_mode)
#                     sur_optimizer.zero_grad()

#                     # try:
#                     #     output = sur_model(image)
#                     # except Exception:
#                     #     sur_model.eval()
#                     #     output = sur_model(image)
#                     #     sur_model.train(mode=training_mode)

#                     output = sur_model(image)

#                     clean_label = label.long()
#                     loss = criterion(output, clean_label)
#                     loss.backward()
#                     sur_optimizer.step()

#                     # Calculate weight differences
#                     sur_model_state_dict = sur_model.state_dict()
#                     # increment = optimize_weight_differences(sur_model_state_dict, original_weights, important_features) 
#                     flat_sur_weights = torch.cat([param.flatten() for  param in sur_model_state_dict.values()])
#                     flat_orig_weights = torch.cat([param.flatten() for  param in original_weights.values()])
#                     important_flat_indices = torch.tensor(important_features).to(device)
#                     important_diff = flat_sur_weights[important_flat_indices] - flat_orig_weights[important_flat_indices]

#                     increment = important_diff.cpu().numpy()


#                     # increment = torch.cat([diff.view(-1).contiguous() for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy().copy()[important_features]  - torch.cat([diff.view(-1).contiguous() for name, diff in original_weights.items() if '20' in name]).cpu().numpy().copy()[important_features]

#                     if tag == 0:  # Suspected samples
#                         temp_sus[index] = increment
#                     elif tag == 1:  # Clean_1 samples
#                         temp_clean[index] = increment
#                     elif tag == 2:  # Clean_2 samples
#                         batch_count_clean_2 += 1
#                         temp_clean_2 = temp_clean_2 + (increment - temp_clean_2) / batch_count_clean_2

#                 # Calculate sus_diff and clean_diff
#                 for index in temp_sus:
#                     sus_diff[(epoch, index)] = temp_sus[index] - temp_clean_2

#                 for index in temp_clean:
#                     clean_diff[(epoch, index)] = temp_clean[index] - temp_clean_2


                    
                
#                 step7_time = time.time() - start_time
#                 step7_time_avg += step7_time
#                 step7_time_avg += step7_time
                
#                 del temp_sus, temp_clean, temp_clean_2
                
                

#         torch.cuda.empty_cache()       
        
        
#         train_ACC.append(acc_meter.avg)
#         print('Train_loss:',loss)
#         # if opt == 'sgd':
#         #     scheduler.step()

        
#         start_time = time.time()
        
        
        
#         step7_time = time.time() - start_time
#         step7_time_avg += step7_time
        
#         print("Step 1 time:", step1_time_avg)
#         print("Step 2 time:", step2_time_avg)
#         print("Step 3 time:", step3_time_avg)
#         print("Step 4_1 time:", step_4_1_time_avg)
#         print("Step 4_2 time:", step_4_2_time_avg)
#         print("Step 4_3 time:", step_4_3_time_avg)
#         print("Step 4_4 time:", step_4_4_time_avg)
#         print("Step 4_5 time:", step_4_5_time_avg)
#         print("Step 5 time:", step5_time_avg)
#         print("Step 5_1 time:", step5_1_time_avg)
#         print("Step 6 time:", step6_time_avg)
#         print("Step 7 time:", step7_time_avg)
        
        
#         # Testing attack effect
#         if type(poisoned_test_loader) == dict:
#             for attack_name in poisoned_test_loader:
#                 print(f"Testing attack effect for {attack_name}")
#                 model.eval()
#                 correct, total = 0, 0
#                 for i, (images, labels) in enumerate(poisoned_test_loader[attack_name]):
#                     images, labels = images.to(device), labels.to(device)
#                     with torch.no_grad():
#                         logits = model(images)
#                         out_loss = criterion(logits,labels)
#                         _, predicted = torch.max(logits.data, 1)
#                         total += labels.size(0)
#                         correct += (predicted == labels).sum().item()
#                 acc = correct / total
#                 test_ACC.append(acc)
#                 print('\nAttack success rate %.2f' % (acc*100))
#                 print('Test_loss:',out_loss)
#         else:
#             # Testing attack effect
#             model.eval()
#             correct, total = 0, 0
#             for i, (images, labels) in enumerate(poisoned_test_loader):
#                 images, labels = images.to(device), labels.to(device)
#                 with torch.no_grad():
#                     logits = model(images)
#                     out_loss = criterion(logits,labels)
#                     _, predicted = torch.max(logits.data, 1)
#                     total += labels.size(0)
#                     correct += (predicted == labels).sum().item()
#             acc = correct / total
#             test_ACC.append(acc)
#             print('\nAttack success rate %.2f' % (acc*100))
#             print('Test_loss:',out_loss)
        
#         correct_clean, total_clean = 0, 0
#         for i, (images, labels) in enumerate(test_loader):
#             images, labels = images.to(device).float(), labels.to(device)
#             with torch.no_grad():
#                 logits = model(images)
#                 out_loss = criterion(logits,labels)
#                 _, predicted = torch.max(logits.data, 1)
#                 total_clean += labels.size(0)
#                 correct_clean += (predicted == labels).sum().item()
#         acc_clean = correct_clean / total_clean
#         clean_ACC.append(acc_clean)
#         print('\nTest clean Accuracy %.2f' % (acc_clean*100))
#         print('Test_loss:', out_loss)
    
#     sus_inds = np.array([ind.item() for epoch, ind in sus_diff])
#     clean_inds = np.array([ind.item() for epoch, ind in clean_diff])
#     sus_diff = np.array([sus_diff[key] for key in sus_diff])
#     clean_diff = np.array([clean_diff[key] for key in clean_diff])
    
#     return sus_diff, clean_diff, sus_inds, clean_inds





def capture_sample_level_weight_updates_idv(random_sus_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, important_features, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    sus_diff = {}
    clean_diff = {}
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    random_sus_idx = set(random_sus_idx)
    
    dataset = poisoned_train_loader.dataset

    # target_indices_class = [dataset[i][2].item() for i in range(len(dataset)) if dataset[i][1] == target_class]
    # print("Number of target indices in the dataset: ", len(target_indices_class))
    # print(len(set(target_indices_class) & random_sus_idx))
    # assert set(target_indices_class) & random_sus_idx == random_sus_idx 
    
    
    print("shape of dataset: ", dataset[0][0].shape)
    if not sample_from_test:
        target_images = [dataset[i][0] for i in range(len(dataset)) if dataset[i][1] == target_class]
        target_images = torch.stack(target_images).to(device)
        target_indices = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] == target_class and dataset[i][2] not in random_sus_idx]
    else:
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ])
        
        transform_sa = Compose([
            RandomHorizontalFlip(),
        ])
        
        test_dataset = test_loader.dataset
        target_images = [test_dataset[i][0].clone() for i in range(len(test_dataset)) if test_dataset[i][1] == target_class]
        if attack == 'narcissus' or attack == 'lc':
            target_images = [transform_train(img) for img in target_images]
        elif attack == 'sa':
            target_images = [transform_sa(img) for img in target_images]
               
            
        print(len(target_images))
        target_images = torch.stack(target_images).to(device)
        print("target images shape:", target_images.shape)
    
    
    def compare_weights(current_weights, reference_weights):
        for key in current_weights:
            if not torch.equal(current_weights[key], reference_weights[key]):
                print(f"Mismatch found in layer: {key}")
                return False
        return True
    
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    
    sur_model = copy.deepcopy(model)
    sur_model.to(device)
    if opt == 'sgd':
        sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt == 'adam':
        sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
    
    torch.save(sur_optimizer.state_dict(), f'./temp_folder/temp_optimizer_state_dict_{random_num}.pth')
    optimizer_state = torch.load(f'./temp_folder/temp_optimizer_state_dict_{random_num}.pth')
    sur_optimizer.load_state_dict(optimizer_state)
    
    os.remove(f"temp_folder/temp_optimizer_state_dict_{random_num}.pth")
    
    
    flattened_weights = torch.cat([param.flatten() for name, param in sur_model.state_dict().items()])
    
    model.to(device) 
    for epoch in tqdm(range(training_epochs)):
        
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 

        step1_time_avg = 0
        step2_time_avg = 0
        step3_time_avg = 0
        step_4_1_time_avg = 0
        step_4_2_time_avg = 0
        step_4_3_time_avg = 0
        step_4_4_time_avg = 0
        step_4_5_time_avg = 0
        step5_time_avg = 0
        step5_1_time_avg = 0
        step6_time_avg = 0
        step7_time_avg = 0
    
        
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device), labels.to(device), indices                    
            
            start_time = time.time()
            
            original_weights = copy.deepcopy(model.state_dict())

            step1_time = time.time() - start_time
            start_time = time.time()
            step1_time_avg += step1_time
            

            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
          
            
            temp_sus = {}
            temp_clean = {}
            
            
            # torch_rng_state = torch.get_rng_state()
            # np_rng_state = np.random.get_state()
            # python_rng_state = random.getstate()
            
            
            indices = indices.clone()
            labels  = labels.clone()
            indices = indices.clone()
            

            pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
            assert np.all(labels[pos_indices].cpu().numpy() == target_class)
            if len(pos_indices) > 0:
                step2_time = time.time() - start_time
                start_time = time.time()
                step2_time_avg += step2_time
                # org_w = np.concatenate([original_weights[name].cpu().numpy().flatten() for name in original_weights])[important_features]
                
                if not sample_from_test:
                    target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
                    available_indices = list(set(target_indices_batch) - set(pos_indices))
                    # print("Length of available indices: ", len(available_indices), "Length of pos indices: ", len(pos_indices), "Length of target indices: ", len(target_indices))
                    if len(available_indices) < len(pos_indices):
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices = available_indices 
                    else:
                        extra_clean_indices = []
                        clean_indices = random.sample(available_indices, len(pos_indices)) 
                    available_indices = list(set(target_indices_batch) - set(pos_indices) - set(clean_indices))
                    if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices_2 = available_indices
                    elif len(available_indices) == 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
                        clean_indices_2 = []
                    else:
                        extra_clean_indices_2 = []
                        clean_indices_2 = random.sample(available_indices, len(pos_indices))
                    

                    assert set(indices[pos_indices].cpu().numpy()) & set(random_sus_idx) == set(indices[pos_indices].cpu().numpy())
                    assert np.all(labels[clean_indices].cpu().numpy() == target_class)
                    assert np.all(labels[clean_indices_2].cpu().numpy() == target_class)
                    
                else:
                    clean_indices = list(random.sample(range(len(target_images)), len(pos_indices)))
                    clean_indices_2 = list(random.sample(set(range(len(target_images))) - set(clean_indices), len(pos_indices)))
                    
                    assert set(clean_indices) & set(clean_indices_2) == set()
                    assert len(clean_indices) == len(clean_indices_2) == len(pos_indices)
              
                
                step3_time = time.time() - start_time
                start_time = time.time()
                step3_time_avg += step3_time
                
                
                
                combined_batch = []
                combined_labels = []
                combined_indexes = []
                pos_indexes = indices[pos_indices].cpu().numpy()
                
                
                # Add suspected samples
                combined_batch.extend(images[pos_indices])
                combined_labels.extend(labels[pos_indices])
                
                
                if not sample_from_test: 
                    if len(extra_clean_indices) > 0:
                        clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
                        clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                        clean_indexes = np.concatenate([indices[clean_indices].cpu().numpy(), np.array(target_indices)[extra_clean_indices]])
                    else:
                        clean_batch = images[clean_indices]
                        clean_labels = labels[clean_indices]
                        clean_indexes = indices[clean_indices].cpu().numpy()
                else:
                    clean_batch = target_images[clean_indices]
                    clean_labels = torch.tensor([target_class] * len(clean_indices)).to(device)
                    clean_indexes = np.array(clean_indices) + 50000
                
                step_4_1_time = time.time() - start_time
                step_4_1_time_avg += step_4_1_time
                start_time = time.time()

                # Add clean_1 samples
                combined_batch.extend(clean_batch)
                combined_labels.extend(clean_labels)
                
                
                if not sample_from_test: 
                    if len(extra_clean_indices_2) > 0:
                        clean_batch_2 = torch.cat([images[clean_indices_2], target_images[extra_clean_indices_2]])
                        clean_labels_2 = torch.cat([labels[clean_indices_2 ], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                        clean_indexes_2 = np.concatenate([indices[clean_indices_2].cpu().numpy(), np.array(target_indices)[extra_clean_indices_2]])
                    else:
                        clean_batch_2 = images[clean_indices_2]
                        clean_labels_2 = labels[clean_indices_2]
                        clean_indexes_2 = indices[clean_indices_2].cpu().numpy()
                else:
                    clean_batch_2 = target_images[clean_indices_2]
                    clean_labels_2 = torch.tensor([target_class] * len(clean_indices_2)).to(device)
                    clean_indexes_2 = np.array(clean_indices_2) + 50000
                
                
                tagged_clean_indexes = [(index, 1) for index in clean_indexes] 
                tagged_clean_indexes_2 = [(index, 2) for index in clean_indexes_2] 
                tagged_pos_indexes = [(index, 0) for index in pos_indexes] 

                combined_indexes = tagged_pos_indexes + tagged_clean_indexes + tagged_clean_indexes_2
                
                
                step_4_2_time = time.time() - start_time
                step_4_2_time_avg += step_4_2_time
                start_time = time.time()


                # Add clean_2 samples
                combined_batch.extend(clean_batch_2)
                combined_labels.extend(clean_labels_2)

                # Convert to tensors
                combined_batch = torch.stack(combined_batch).to(device)
                combined_labels = torch.tensor(combined_labels).to(device)
               
                # def seed_worker(worker_id):
                #     worker_seed = torch.initial_seed() % (2**32)
                #     np.random.seed(worker_seed)
                #     random.seed(worker_seed)
     
                gen = torch.Generator().manual_seed(seed)

                # Create a single DataLoader for all samples
                combined_loader = DataLoader(
                    list(zip(combined_batch, combined_labels, combined_indexes)),
                    batch_size=1,
                    shuffle=False,  # Can still shuffle deterministically
                    # num_workers=4,  # Number of workers
                    # worker_init_fn=seed_worker,  # Seed each worker
                    generator=gen  # Control shuffling
                )

                # Dictionaries to store differences
                temp_sus = {}
                temp_clean = {}
                temp_clean_2 = np.zeros(len(important_features))  # Running average for clean_2
                batch_count_clean_2 = 0

                
                # # Iterate through the combined loader
                for image, label, (index, tag) in combined_loader:
                    torch_rng_state = torch.get_rng_state()
                    cuda_rng_state = torch.cuda.get_rng_state()
                    np_rng_state = np.random.get_state()
                    python_rng_state = random.getstate()
    
                    # Reset surrogate model
                    sur_model.load_state_dict(original_weights)
                    sur_optimizer.load_state_dict(optimizer_state)
                    sur_model.train(mode=training_mode)
                    sur_optimizer.zero_grad()

                    output = sur_model(image)
                    
                    step_4_3_time = time.time() - start_time
                    step_4_3_time_avg += step_4_3_time
                    start_time = time.time()

                    clean_label = label.long()
                    loss = criterion(output, clean_label)
                    loss.backward()
                    sur_optimizer.step()

                    # Calculate weight differences
                    sur_model_state_dict = sur_model.state_dict()
                    
                    step_4_4_time = time.time() - start_time
                    step_4_4_time_avg += step_4_4_time
                    start_time = time.time()
            
                    
                    flat_sur_weights = torch.cat([param.flatten() for  param in sur_model_state_dict.values()])
                    flat_orig_weights = torch.cat([param.flatten() for  param in original_weights.values()])
                    important_flat_indices = torch.tensor(important_features).to(device)
                    important_diff = flat_sur_weights[important_flat_indices] - flat_orig_weights[important_flat_indices]

                    increment = important_diff.cpu().numpy()

                    
                    step_4_5_time = time.time() - start_time
                    step_4_5_time_avg += step_4_5_time
                    start_time = time.time()

                    if tag == 0:  # Suspected samples
                        temp_sus[index] = increment
                    elif tag == 1:  # Clean_1 samples
                        temp_clean[index] = increment
                    elif tag == 2:  # Clean_2 samples
                        batch_count_clean_2 += 1
                        temp_clean_2 = temp_clean_2 + (increment - temp_clean_2) / batch_count_clean_2
                        
                    step5_time = time.time() - start_time
                    step5_time_avg += step5_time
                    start_time = time.time()
                    
                    torch.set_rng_state(torch_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state(cuda_rng_state)
                    np.random.set_state(np_rng_state)
                    random.setstate(python_rng_state)
    

                # Calculate sus_diff and clean_diff
                for index in temp_sus:
                    sus_diff[(epoch, index)] = temp_sus[index] - temp_clean_2
                    
                    
                step6_time = time.time() - start_time
                step6_time_avg += step6_time
                start_time = time.time()

                for index in temp_clean:
                    clean_diff[(epoch, index)] = temp_clean[index] - temp_clean_2

                
                    
                
                step7_time = time.time() - start_time
                step7_time_avg += step7_time
                step7_time_avg += step7_time
                
                del temp_sus, temp_clean, temp_clean_2
                
                # torch.set_rng_state(torch_rng_state)
                # np.random.set_state(np_rng_state)
                # random.setstate(python_rng_state)
        
                

        torch.cuda.empty_cache()       
        
        
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        # if opt == 'sgd':
        #     scheduler.step()

        
        
        start_time = time.time()
        
        
        
        step7_time = time.time() - start_time
        step7_time_avg += step7_time
        
        print("Step 1 time:", step1_time_avg)
        print("Step 2 time:", step2_time_avg)
        print("Step 3 time:", step3_time_avg)
        print("Step 4_1 time:", step_4_1_time_avg)
        print("Step 4_2 time:", step_4_2_time_avg)
        print("Step 4_3 time:", step_4_3_time_avg)
        print("Step 4_4 time:", step_4_4_time_avg)
        print("Step 4_5 time:", step_4_5_time_avg)
        print("Step 5 time:", step5_time_avg)
        print("Step 5_1 time:", step5_1_time_avg)
        print("Step 6 time:", step6_time_avg)
        print("Step 7 time:", step7_time_avg)
        
        
        # Testing attack effect
        if type(poisoned_test_loader) == dict:
            for attack_name in poisoned_test_loader:
                print(f"Testing attack effect for {attack_name}")
                model.eval()
                correct, total = 0, 0
                for i, (images, labels) in enumerate(poisoned_test_loader[attack_name]):
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        logits = model(images)
                        out_loss = criterion(logits,labels)
                        _, predicted = torch.max(logits.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                acc = correct / total
                test_ACC.append(acc)
                print('\nAttack success rate %.2f' % (acc*100))
                print('Test_loss:',out_loss)
        else:
            # Testing attack effect
            model.eval()
            correct, total = 0, 0
            for i, (images, labels) in enumerate(poisoned_test_loader):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    logits = model(images)
                    out_loss = criterion(logits,labels)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            test_ACC.append(acc)
            print('\nAttack success rate %.2f' % (acc*100))
            print('Test_loss:',out_loss)
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device).float(), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:', out_loss)
    
    
    sus_inds = np.array([ind.item() for epoch, ind in sus_diff])
    clean_inds = np.array([ind.item() for epoch, ind in clean_diff])
    sus_diff = np.array([sus_diff[key] for key in sus_diff])
    clean_diff = np.array([clean_diff[key] for key in clean_diff])
    
    return sus_diff, clean_diff, sus_inds, clean_inds




# def capture_sample_level_weight_updates_idv(random_sus_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, important_features, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1): 
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
    
#     train_ACC = []
#     test_ACC = []
#     clean_ACC = []
#     target_ACC = []

#     sus_diff = {}
#     clean_diff = {}
#     separate_rng = np.random.default_rng()
#     random_num = separate_rng.integers(1, 10000)
#     random_sus_idx = set(random_sus_idx)
    
#     target_indices_class = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl == target_class]
#     print(len(set(target_indices_class) & random_sus_idx))
#     assert set(target_indices_class) & random_sus_idx == random_sus_idx
#     if not sample_from_test:
#         target_images = [img for imgs, lbls, idxs in poisoned_train_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
#         target_images = torch.stack(target_images).to(device)
#         target_indices = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl.item() == target_class and idx.item() not in random_sus_idx]
#     else:
#         transform_train = Compose([
#             RandomCrop(32, padding=4),
#             RandomHorizontalFlip(),
#         ])
        
#         transform_sa = Compose([
#             RandomHorizontalFlip(),
#         ])
#         if attack == 'narcissus' or attack == 'lc':
#             target_images = [transform_train(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]   
#         else:
#             target_images = [transform_sa(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]   
            
#         print(len(target_images))
#         target_images = torch.stack(target_images).to(device)
#         print("target images shape:", target_images.shape)
    
    
    
    
#     def optimize_weight_differences(pos_model_state_dict, original_weights, important_features):
#         # Extract the keys once to avoid repeated dict key access
#         state_dict_keys = list(pos_model_state_dict.keys())
        
#         # Preallocate a list for the differences
#         differences = []
        
#         # Compute the differences and flatten them in a single loop
#         for name in state_dict_keys:
#             diff = (pos_model_state_dict[name] - original_weights[name]).cpu().numpy().flatten()
#             differences.append(diff)
        
#         # Concatenate all differences into a single array
#         differences_array = np.concatenate(differences)
        
#         # Select important features
#         optimized_result = differences_array[important_features]
        
#         return optimized_result
    
#     def clone_optimizer_state(optimizer_state):
#         """Helper function to deep clone the optimizer state dictionary."""
#         cloned_state = {}
#         for k, v in optimizer_state.items():
#             if isinstance(v, torch.Tensor):
#                 cloned_state[k] = v.clone()
#             elif isinstance(v, dict):
#                 cloned_state[k] = clone_optimizer_state(v)
#             elif isinstance(v, list):
#                 cloned_state[k] = [item.clone() if isinstance(item, torch.Tensor) else item for item in v]
#             else:
#                 raise ValueError(f"Unsupported type: {type(v)}")
#         return cloned_state
    
#     def compare_optimizer_states(state1, state2):
#         """Helper function to compare optimizer states."""
#         for k in state1:
#             if isinstance(state1[k], torch.Tensor):
#                 assert torch.equal(state1[k], state2[k]), f"Mismatch in key {k}"
#             elif isinstance(state1[k], dict):
#                 compare_optimizer_states(state1[k], state2[k])
#             elif isinstance(state1[k], list):
#                 for item1, item2 in zip(state1[k], state2[k]):
#                     if isinstance(item1, torch.Tensor):
#                         assert torch.equal(item1, item2), f"Mismatch in list item for key {k}"
#                     else:
#                         assert item1 == item2, f"Mismatch in list item for key {k}"
#             else:
#                 assert state1[k] == state2[k], f"Mismatch in key {k}"
    
#     separate_rng = np.random.default_rng()
#     random_num = separate_rng.integers(1, 10000)
    
#     sur_model = copy.deepcopy(model)
#     sur_model.to(device)
#     sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    
#     for epoch in tqdm(range(training_epochs)):
#         # Train
#         model.to(device) 
#         model.train(mode = training_mode)
#         acc_meter = AverageMeter()
#         loss_meter = AverageMeter()
#         pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        
        
        
#         step1_time_avg = 0
#         step2_time_avg = 0
#         step3_time_avg = 0
#         step_4_1_time_avg = 0
#         step_4_2_time_avg = 0
#         step_4_3_time_avg = 0
#         step_4_4_time_avg = 0
#         step_4_5_time_avg = 0
#         step5_time_avg = 0
#         step5_1_time_avg = 0
#         step6_time_avg = 0
#         step7_time_avg = 0
    
        
#         for images, labels, indices in pbar:
#             images, labels, indices = images.to(device), labels.to(device), indices                       
            
#             start_time = time.time()
            
#             original_weights = torch.save(model.state_dict(), f'./temp_folder/temp_weights_{random_num}.pth')
#             # original_weights = copy.deepcopy(model.state_dict())
#             if opt == 'sgd':
#                 torch.save(optimizer.state_dict(), f'./temp_folder/temp_optimizer_{random_num}.pth')       
#             elif opt == 'adam':
#                 torch.save(model.state_dict(), f'./temp_folder/temp_weights_{random_num}.pth')

            
#             model.zero_grad()
#             optimizer.zero_grad()
#             logits = model(images)
#             loss = criterion(logits, labels)
#             loss.backward()
#             optimizer.step()
            
#             _, predicted = torch.max(logits.data, 1)
#             acc = (predicted == labels).sum().item()/labels.size(0)
#             acc_meter.update(acc)
#             loss_meter.update(loss.item())
#             pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
#             step1_time = time.time() - start_time
#             start_time = time.time()
#             step1_time_avg += step1_time
            
#             temp_sus = {}
#             temp_clean = {}
            

#             pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
#             assert np.all(labels[pos_indices].cpu().numpy() == target_class)
#             if len(pos_indices) > 0:
#                 step2_time = time.time() - start_time
#                 start_time = time.time()
#                 step2_time_avg += step2_time
#                 # org_w = np.concatenate([original_weights[name].cpu().numpy().flatten() for name in original_weights])[important_features]
                
#                 if not sample_from_test:
#                     target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
#                     available_indices = list(set(target_indices_batch) - set(pos_indices))
#                     # print("Length of available indices: ", len(available_indices), "Length of pos indices: ", len(pos_indices), "Length of target indices: ", len(target_indices))
#                     if len(available_indices) < len(pos_indices):
#                         indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
#                         remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
#                         extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
#                         clean_indices = available_indices 
#                     else:
#                         extra_clean_indices = []
#                         clean_indices = random.sample(available_indices, len(pos_indices)) 
#                     available_indices = list(set(target_indices_batch) - set(pos_indices) - set(clean_indices))
#                     if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
#                         indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
#                         remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
#                         extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
#                         clean_indices_2 = available_indices
#                     elif len(available_indices) == 0:
#                         indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
#                         remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
#                         extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
#                         clean_indices_2 = []
#                     else:
#                         extra_clean_indices_2 = []
#                         clean_indices_2 = random.sample(available_indices, len(pos_indices))
                    

#                     assert set(indices[pos_indices].cpu().numpy()) & set(random_sus_idx) == set(indices[pos_indices].cpu().numpy())
#                     assert np.all(labels[clean_indices].cpu().numpy() == target_class)
#                     assert np.all(labels[clean_indices_2].cpu().numpy() == target_class)
                    
#                 else:
#                     clean_indices = list(random.sample(range(len(target_images)), len(pos_indices)))
#                     clean_indices_2 = list(random.sample(set(range(len(target_images))) - set(clean_indices), len(pos_indices)))
                    
#                     assert set(clean_indices) & set(clean_indices_2) == set()
              
                
#                 step3_time = time.time() - start_time
#                 start_time = time.time()
#                 step3_time_avg += step3_time
                
                
#                 step_4_1_time = time.time() - start_time
#                 step_4_1_time_avg += step_4_1_time
#                 start_time = time.time()
                
                
                
                
#                 temp_pos_loader = DataLoader(list(zip(images[pos_indices], labels[pos_indices], pos_indices)), batch_size=1, shuffle=False) 
#                 for image, label, index in temp_pos_loader:
                    
#                     if opt == 'sgd':
#                         original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
#                         original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
#                         sur_optimizer.load_state_dict(original_optimizer)
#                     elif opt == 'adam':
#                         original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
#                         sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
#                     sur_model.train(mode = training_mode)
                    
#                     sur_model.load_state_dict(original_weights)
#                     sur_optimizer.zero_grad()
                    
#                     step_4_2_time = time.time() - start_time
#                     step_4_2_time_avg += step_4_2_time
#                     start_time = time.time()
                    
#                     output = sur_model(image)
                                        
#                     loss = criterion(output, label)
#                     loss.backward()
#                     sur_optimizer.step()
                
#                     step_4_3_time = time.time() - start_time 
#                     step_4_3_time_avg += step_4_3_time
#                     start_time = time.time()
                    
#                     sur_model_state_dict = sur_model.state_dict()                    
#                     if attack != 'ht':                        
#                         diffs = sur_model_state_dict.values()
#                         temp_sus[indices[index]] = torch.cat([diff.view(-1) for diff in diffs])[important_features].cpu().numpy()

#                     else:
#                         temp_sus[indices[index]] = torch.cat([diff.view(-1).contiguous() for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy().copy()[important_features]
                     
#                     step4_4_time = time.time() - start_time
#                     step_4_4_time_avg += step4_4_time
#                     start_time = time.time()
                
                
                

                
#                 if not sample_from_test: 
#                     if len(extra_clean_indices) > 0:
#                         clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
#                         clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
#                         clean_indexes = np.concatenate([indices[clean_indices].cpu().numpy(), np.array(target_indices)[extra_clean_indices]])
#                     else:
#                         clean_batch = images[clean_indices]
#                         clean_labels = labels[clean_indices]
#                         clean_indexes = indices[clean_indices].cpu().numpy()
#                 else:
#                     clean_batch = target_images[clean_indices]
#                     clean_labels = torch.tensor([target_class] * len(clean_indices)).to(device)
#                     clean_indexes = np.array(clean_indices) + 50000
                
#                 start_time = time.time()

#                 temp_clean_1_loader = DataLoader(list(zip(clean_batch, clean_labels, clean_indexes)), batch_size=1, shuffle=False)
#                 for image, label, index in temp_clean_1_loader:
                    
#                     if opt == 'sgd':
#                         original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
#                         original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
#                         sur_optimizer.load_state_dict(original_optimizer)
#                     elif opt == 'adam':
#                         original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
#                         sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
#                     sur_model.load_state_dict(original_weights)
#                     sur_model.train(mode = training_mode)
                    
#                     sur_optimizer.zero_grad()
                    
                    
#                     output = sur_model(image) 
#                     clean_label = label.long()
#                     loss = criterion(output, clean_label)
#                     loss.backward()
#                     sur_optimizer.step()
                    
#                     sur_model_state_dict = sur_model.state_dict()                    
#                     if attack != 'ht':
#                         diffs = sur_model_state_dict.values()
#                         temp_clean[index] = torch.cat([diff.view(-1) for diff in diffs])[important_features].cpu().numpy()

#                     else:
#                         temp_clean[index] = torch.cat([diff.view(-1).contiguous() for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy().copy()[important_features]
                    
                    



#                 step5_time = time.time() - start_time
#                 start_time = time.time()
#                 step5_time_avg += step5_time
                
                

#                 start_time_2 = time.time()                
#                 if not sample_from_test: 
#                     if len(extra_clean_indices_2) > 0:
#                         clean_batch = torch.cat([images[clean_indices_2 ], target_images[extra_clean_indices_2]])
#                         clean_labels = torch.cat([labels[clean_indices_2 ], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
#                         clean_indexes = np.concatenate([indices[clean_indices_2].cpu().numpy(), np.array(target_indices)[extra_clean_indices_2]])
#                     else:
#                         clean_batch = images[clean_indices_2]
#                         clean_labels = labels[clean_indices_2]
#                         clean_indexes = indices[clean_indices_2].cpu().numpy()
#                 else:
#                     clean_batch = target_images[clean_indices_2]
#                     clean_labels = torch.tensor([target_class] * len(clean_indices_2)).to(device)
#                     clean_indexes = np.array(clean_indices_2) + 50000

#                 step5_1_time = time.time() - start_time_2
#                 step5_1_time_avg += step5_1_time
                
#                 clean_2_loader = DataLoader(list(zip(clean_batch, clean_labels)), batch_size=1, shuffle=False)
                
#                 temp_clean_2 = np.zeros(len(important_features))
#                 for image, label in clean_2_loader:
                    
#                     if opt == 'sgd':
#                         original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
#                         original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
#                         sur_optimizer.load_state_dict(original_optimizer)
#                     elif opt == 'adam':
#                         original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
#                         sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
                    
#                     sur_model.load_state_dict(original_weights)
#                     sur_model.train(mode = training_mode)
                    
                    
#                     sur_optimizer.zero_grad()
                    
#                     output = sur_model(image)
#                     clean_label = label.long()
#                     loss = criterion(output, clean_label)
#                     loss.backward()
#                     sur_optimizer.step()
                    
#                     sur_model_state_dict = sur_model.state_dict()
#                     if attack != 'ht':
#                         diffs = sur_model_state_dict.values()
#                         temp_clean_2 += torch.cat([diff.view(-1) for diff in diffs])[important_features].cpu().numpy()
#                     else:
#                         temp_clean_2 += torch.cat([diff.view(-1).contiguous() for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy().copy()[important_features]
                    
#                 temp_clean_2 /= len(clean_indices_2)
                
#                 step6_time = time.time() - start_time
#                 step6_time_avg += step6_time
#                 start_time = time.time()
                
#                 for index in temp_sus:
#                     sus_diff[(epoch, index)] = temp_sus[index] - temp_clean_2 

#                 for index in temp_clean:
#                     clean_diff[(epoch, index)] = temp_clean[index] - temp_clean_2
                
#                 step7_time = time.time() - start_time
#                 step7_time_avg += step7_time
#                 step7_time_avg += step7_time
                
#                 del temp_sus, temp_clean, temp_clean_2
                
                

#         torch.cuda.empty_cache()       
        
        
#         train_ACC.append(acc_meter.avg)
#         print('Train_loss:',loss)
#         if opt == 'sgd':
#             scheduler.step()

        
#         start_time = time.time()
        
        
        
#         step7_time = time.time() - start_time
#         step7_time_avg += step7_time
        
#         print("Step 1 time:", step1_time_avg)
#         print("Step 2 time:", step2_time_avg)
#         print("Step 3 time:", step3_time_avg)
#         print("Step 4_1 time:", step_4_1_time_avg)
#         print("Step 4_2 time:", step_4_2_time_avg)
#         print("Step 4_3 time:", step_4_3_time_avg)
#         print("Step 4_4 time:", step_4_4_time_avg)
#         print("Step 4_5 time:", step_4_5_time_avg)
#         print("Step 5 time:", step5_time_avg)
#         print("Step 5_1 time:", step5_1_time_avg)
#         print("Step 6 time:", step6_time_avg)
#         print("Step 7 time:", step7_time_avg)
        
        
#         # Testing attack effect
#         model.eval()
#         correct, total = 0, 0
#         for i, (images, labels) in enumerate(poisoned_test_loader):
#             images, labels = images.to(device), labels.to(device)
#             with torch.no_grad():
#                 logits = model(images)
#                 out_loss = criterion(logits,labels)
#                 _, predicted = torch.max(logits.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         acc = correct / total
#         test_ACC.append(acc)
#         print('\nAttack success rate %.2f' % (acc*100))
#         print('Test_loss:',out_loss)
        
#         correct_clean, total_clean = 0, 0
#         for i, (images, labels) in enumerate(test_loader):
#             images, labels = images.to(device), labels.to(device)
#             with torch.no_grad():
#                 logits = model(images)
#                 out_loss = criterion(logits,labels)
#                 _, predicted = torch.max(logits.data, 1)
#                 total_clean += labels.size(0)
#                 correct_clean += (predicted == labels).sum().item()
#         acc_clean = correct_clean / total_clean
#         clean_ACC.append(acc_clean)
#         print('\nTest clean Accuracy %.2f' % (acc_clean*100))
#         print('Test_loss:',out_loss)
    
#     sus_inds = np.array([ind.item() for epoch, ind in sus_diff])
#     clean_inds = np.array([ind.item() for epoch, ind in clean_diff])
#     sus_diff = np.array([sus_diff[key] for key in sus_diff])
#     clean_diff = np.array([clean_diff[key] for key in clean_diff])
    
#     return sus_diff, clean_diff, sus_inds, clean_inds



def capture_simple_sample_level_weight_updates(random_sus_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, poisoned_train_dataset, test_dataset, important_features, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    sus_diff = {}
    clean_diff = {}
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    random_sus_idx = set(random_sus_idx)

    sur_model = copy.deepcopy(model) 
    sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    flattened_weights = torch.cat([param.flatten() for param in sur_model.state_dict().values()]).detach().cpu().numpy() 

    layer_mapping = {}
    current_index = 0

    for layer_name, param in sur_model.state_dict().items():
        param_size = param.numel()
        # Identify important features within the current parameter's range
        relevant_indices = [idx - current_index for idx in important_features if current_index <= idx < current_index + param_size]

        if relevant_indices:
            layer_mapping[layer_name] = relevant_indices

        current_index += param_size
    
    
    if not sample_from_test:
        target_indices = [i for i, (img, lbl, idx) in enumerate(poisoned_train_dataset) if lbl == target_class and idx not in random_sus_idx]
        
        # Determine the sample size based on availability
        sample_size = min(len(random_sus_idx), len(target_indices))
        
        # Sample for clean_1_indices
        clean_1_indices = set(np.random.choice(target_indices, sample_size, replace=False))
        clean_1_dataset = Subset(poisoned_train_dataset, list(clean_1_indices))
        clean_1_loader = DataLoader(clean_1_dataset, batch_size=1, shuffle=False, num_workers=2)
        
        # Sample for clean_2_indices with the remaining indices
        remaining_indices = list(set(target_indices) - clean_1_indices)
        clean_2_sample_size = min(sample_size, len(remaining_indices))  # Ensure we don't exceed available samples
        clean_2_indices = np.random.choice(remaining_indices, clean_2_sample_size, replace=False)
        clean_2_dataset = Subset(poisoned_train_dataset, list(clean_2_indices))
        clean_2_loader = DataLoader(clean_2_dataset, batch_size=1, shuffle=False, num_workers=2)
        
        assert len(clean_1_indices) > 0 and len(clean_2_indices) > 0
    else:
        
        class TransformedDataset(Dataset):
            def __init__(self, dataset, transform=None):
                self.dataset = dataset
                self.transform = transform

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                if self.transform:
                    img = self.transform(img)
                return img, label, idx
        
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ])
        
        transform_sa = Compose([
            RandomHorizontalFlip(),
        ])
        
        if attack == 'narcissus' or attack == 'lc':
            transform = transform_train
        elif attack == 'sa':
            transform = transform_sa
        else:
            transform = None
        
        test_dataset = TransformedDataset(test_dataset, transform)
        target_indices = [i for i, (img, lbl, _) in enumerate(test_dataset) if lbl == target_class]
        clean_1_dataset = Subset(test_dataset, target_indices[:len(target_indices)//2])
        clean_1_loader = DataLoader(clean_1_dataset, batch_size=1, shuffle=False, num_workers=2)   
        clean_2_dataset = Subset(test_dataset, target_indices[len(target_indices)//2:])
        clean_2_loader = DataLoader(clean_2_dataset, batch_size=1, shuffle=False, num_workers=2)
        
    def get_index_value(idx):
        return idx.item() if isinstance(idx, torch.Tensor) else idx

    pos_indices = [i for i, (img, lbl, idx) in enumerate(poisoned_train_dataset) if get_index_value(idx) in random_sus_idx]

    pos_dataset = Subset(poisoned_train_dataset, pos_indices)
    pos_loader = DataLoader(pos_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    
    print("Length of clean_1_loader: ", len(clean_1_loader), "Length of clean_2_loader: ", len(clean_2_loader), "Length of pos_loader: ", len(pos_loader))
    for epoch in tqdm(range(training_epochs)):
        # Train
        model.to(device) 
        model.train(mode = training_mode)
        
        step1_time_avg = 0
        step2_time_avg = 0
        step3_time_avg = 0
        step4_time_avg = 0
        step5_time_avg = 0
        
        
        start_time = time.time()
        
        
        original_weights = copy.deepcopy(model.state_dict())
        # current_lr = optimizer.param_groups[0]['lr']
        # momentum = optimizer.param_groups[0]['momentum']
        # weight_decay = optimizer.param_groups[0]['weight_decay']
        
        torch.save(optimizer.state_dict(), f'./temp_folder/temp_optimizer_{random_num}.pth')     
        step1_time = time.time() - start_time
        start_time = time.time()
        
        
        print()
        sur_model.train()
        sus_dict = {}
        for i, (images, labels, indices) in enumerate(pos_loader):
            print(f"\rProcessing batch {i + 1}/{len(pos_loader)}", end="")
            sur_model.train(mode = training_mode)
            sur_model.load_state_dict(original_weights)
            
            # sur_optimizer.param_groups[0]['lr'] = current_lr
            # sur_optimizer.param_groups[0]['momentum'] = momentum
            # sur_optimizer.param_groups[0]['weight_decay'] = weight_decay
            original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
            sur_optimizer.load_state_dict(original_optimizer)
            
            images, labels = images.to(device), labels.to(device)
            sur_optimizer.zero_grad()   
            logits = sur_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            sur_optimizer.step()
            
            sur_model_state_dict = sur_model.state_dict()
            
            temp_sus = []
            for layer_name, indices_ in layer_mapping.items():
                sur_layer_weights = sur_model_state_dict[layer_name]
                orig_layer_weights = original_weights[layer_name]

                # Access the important indices and compute the difference
                important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                temp_sus.append(important_diff)
                
            temp_sus = torch.cat(temp_sus).cpu().numpy()
            sus_dict[indices.item()] = temp_sus
        
        start_time = time.time()

        print()
        clean_1_dict = {}
        for i, (images, labels, indices) in enumerate(clean_1_loader):
            print(f"\rProcessing batch {i + 1}/{len(clean_1_loader)}", end="")
            sur_model.train(mode = training_mode)
            sur_model.load_state_dict(original_weights)
            images, labels = images.to(device), labels.to(device)
            
            original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
            sur_optimizer.load_state_dict(original_optimizer)
            
            sur_optimizer.zero_grad()
            logits = sur_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            sur_optimizer.step()
            
            sur_model_state_dict = sur_model.state_dict()
            
            temp_clean_1 = []
            for layer_name, indices_ in layer_mapping.items():
                sur_layer_weights = sur_model_state_dict[layer_name]
                orig_layer_weights = original_weights[layer_name]

                # Access the important indices and compute the difference
                important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                temp_clean_1.append(important_diff)

            temp_clean_1 = torch.cat(temp_clean_1).cpu().numpy()
            clean_1_dict[indices.item()] = temp_clean_1
        


        step2_time = time.time() - start_time
        step2_time_avg += step2_time
        start_time = time.time()
     
        print()
        avg_clean_2 = np.array([0.0] * len(important_features))
        for i, (images, labels, indices) in enumerate(clean_2_loader):
            print(f"\rProcessing batch {i + 1}/{len(clean_2_loader)}", end="")
            sur_model.train(mode = training_mode)
            sur_model.load_state_dict(original_weights)
            images, labels = images.to(device), labels.to(device)
            
            original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
            sur_optimizer.load_state_dict(original_optimizer)
            
            sur_optimizer.zero_grad()
            logits = sur_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            sur_optimizer.step()
            
            sur_model_state_dict = sur_model.state_dict()
            
            temp_clean_2 = []
            for layer_name, indices_ in layer_mapping.items():
                sur_layer_weights = sur_model_state_dict[layer_name]
                orig_layer_weights = original_weights[layer_name]

                # Access the important indices and compute the difference
                important_diff = sur_layer_weights.view(-1)[indices_] - orig_layer_weights.view(-1)[indices_]
                temp_clean_2.append(important_diff)
            
            temp_clean_2 = torch.cat(temp_clean_2).cpu().numpy()
            avg_clean_2 += temp_clean_2
        
        avg_clean_2 /= len(clean_2_loader)

        
        step3_time = time.time() - start_time
        step3_time_avg += step3_time
        start_time = time.time()
        
        for idx, diff in sus_dict.items():
            sus_diff[(epoch, idx)] = diff - avg_clean_2
        for idx, diff in clean_1_dict.items():
            clean_diff[(epoch, idx)] = diff - avg_clean_2
        
        step4_time = time.time() - start_time
        step4_time_avg += step4_time
        step4_time_avg += step4_time
        
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        
        
        if epoch < training_epochs - 1:
            for images, labels, indices in pbar:
                images, labels, indices = images.to(device), labels.to(device), indices               

                model.zero_grad()
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(logits.data, 1)
                acc = (predicted == labels).sum().item()/labels.size(0)
                acc_meter.update(acc)
                loss_meter.update(loss.item())
                pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
                
                
                
                    
                    

            torch.cuda.empty_cache()       
                    
            train_ACC.append(acc_meter.avg)
            print('Train_loss:',loss)
            if opt == 'sgd':
                scheduler.step()
        
            
        step5_time = time.time() - start_time
        step5_time_avg += step5_time
        start_time = time.time()
        
        print("Step 1 time:", step1_time_avg)
        print("Step 2 time:", step2_time_avg)
        print("Step 3 time:", step3_time_avg)
        print("Step 4 time:", step4_time_avg)
        print("Step 5 time:", step5_time_avg)
        
        
        # Testing attack effect
        model.eval()
        correct, total = 0, 0
        for i, (images, labels) in enumerate(poisoned_test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        test_ACC.append(acc)
        print('\nAttack success rate %.2f' % (acc*100))
        print('Test_loss:',out_loss)
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:',out_loss)
    
    sus_inds = np.array([ind for epoch, ind in sus_diff])
    clean_inds = np.array([ind for epoch, ind in clean_diff])
    sus_diff = np.array([sus_diff[key] for key in sus_diff])
    clean_diff = np.array([clean_diff[key] for key in clean_diff])
    
    return sus_diff, clean_diff, sus_inds, clean_inds


def capture_sample_level_weight_updates(random_sus_idx, model, orig_model, optimizer, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, important_features, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    sus_diff = {}
    clean_diff = {}
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    random_sus_idx = set(random_sus_idx)
    
    target_indices_class = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl == target_class]
    print(len(set(target_indices_class) & random_sus_idx))
    assert set(target_indices_class) & random_sus_idx == random_sus_idx
    if not sample_from_test:
        target_images = [img for imgs, lbls, idxs in poisoned_train_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]
        target_images = torch.stack(target_images).to(device)
        target_indices = [idx.item() for imgs, lbls, idxs in poisoned_train_loader for idx, lbl in zip(idxs, lbls) if lbl.item() == target_class and idx.item() not in random_sus_idx]
    else:
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ])
        
        
        if attack == 'narcissus' or attack == 'lc':
            target_images = [transform_train(img) for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]   
        else:
            target_images = [img for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl.item() == target_class]   
            
        print(len(target_images))
        target_images = torch.stack(target_images).to(device)
    
    
    sur_model = copy.deepcopy(model)
    sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    
    def optimize_weight_differences(pos_model_state_dict, original_weights, important_features):
        # Extract the keys once to avoid repeated dict key access
        state_dict_keys = list(pos_model_state_dict.keys())
        
        # Preallocate a list for the differences
        differences = []
        
        # Compute the differences and flatten them in a single loop
        for name in state_dict_keys:
            diff = (pos_model_state_dict[name] - original_weights[name]).cpu().numpy().flatten()
            differences.append(diff)
        
        # Concatenate all differences into a single array
        differences_array = np.concatenate(differences)
        
        # Select important features
        optimized_result = differences_array[important_features]
        
        return optimized_result
    
    
    for epoch in tqdm(range(training_epochs)):
        # Train
        model.to(device) 
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        
        step1_time_avg = 0
        step2_time_avg = 0
        step3_time_avg = 0
        step_4_1_time_avg = 0
        step_4_2_time_avg = 0
        step_4_3_time_avg = 0
        step_4_4_time_avg = 0
        step_4_5_time_avg = 0
        step5_time_avg = 0
        step5_1_time_avg = 0
        step6_time_avg = 0
        step7_time_avg = 0
    
        
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device), labels.to(device), indices                       
            
            start_time = time.time()
            
            original_weights = copy.deepcopy(model.state_dict())
            torch.save(optimizer.state_dict(), f'./temp_folder/temp_optimizer_{random_num}.pth')
                        
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
            step1_time = time.time() - start_time
            start_time = time.time()
            step1_time_avg += step1_time
            
            temp_sus = {}
            temp_clean = {}
            

            pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
            if len(pos_indices) > 0:
                step2_time = time.time() - start_time
                start_time = time.time()
                step2_time_avg += step2_time
                
                if not sample_from_test:
                    target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
                    available_indices = list(set(target_indices_batch) - set(pos_indices))
                    # print("Length of available indices: ", len(available_indices), "Length of pos indices: ", len(pos_indices), "Length of target indices: ", len(target_indices))
                    if len(available_indices) < len(pos_indices):
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices = available_indices 
                    else:
                        extra_clean_indices = []
                        clean_indices = random.sample(available_indices, len(pos_indices)) 
                    available_indices = list(set(target_indices_batch) - set(pos_indices) - set(clean_indices))
                    if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices_2 = available_indices
                    elif len(available_indices) == 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
                        clean_indices_2 = []
                    else:
                        extra_clean_indices_2 = []
                        clean_indices_2 = random.sample(available_indices, len(pos_indices))
                    
                    remaining_indices = list(set(range(len(indices))) - set(pos_indices) - set(clean_indices) - set(clean_indices_2))
                    assert set(indices[remaining_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices_2]) & set(random_sus_idx) == set()

                    assert set(indices[pos_indices].cpu().numpy()) & set(random_sus_idx) == set(indices[pos_indices].cpu().numpy())
                    assert np.all(labels[clean_indices].cpu().numpy() == target_class)
                    assert np.all(labels[clean_indices_2].cpu().numpy() == target_class)
                    
                else:
                    clean_indices = list(random.sample(range(len(target_images)), len(pos_indices)))
                    clean_indices_2 = list(random.sample(set(range(len(target_images))) - set(clean_indices), len(pos_indices)))
                    remaining_indices = [i for i, ind in enumerate(indices) if ind.item() not in random_sus_idx]
                    assert set(indices[remaining_indices].cpu().numpy()) | set(indices[pos_indices].cpu().numpy()) == set(indices.cpu().numpy()) and set(indices[remaining_indices].cpu().numpy()) & set(indices[pos_indices].cpu().numpy()) == set()
                
              
                
                step3_time = time.time() - start_time
                start_time = time.time()
                step3_time_avg += step3_time
                
                
                step_4_1_time = time.time() - start_time
                step_4_1_time_avg += step_4_1_time
                start_time = time.time()
                
                
                
                
                temp_pos_loader = DataLoader(list(zip(images[pos_indices], labels[pos_indices], pos_indices)), batch_size=1, shuffle=False)
                for image, label, index in temp_pos_loader:
                    
                    original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                    
                    sur_model.load_state_dict(original_weights)
                    sur_model.train(mode = training_mode)
                    
                    sur_optimizer.load_state_dict(original_optimizer)
                    sur_optimizer.zero_grad()
                    
                    step_4_2_time = time.time() - start_time
                    step_4_2_time_avg += step_4_2_time
                    start_time = time.time()
                    
                    output = sur_model(torch.cat([image, images[remaining_indices]]))
                                        
                    loss = criterion(output, torch.cat([label, labels[remaining_indices]]))
                    loss.backward()
                    sur_optimizer.step()
                
                    step_4_3_time = time.time() - start_time 
                    step_4_3_time_avg += step_4_3_time
                    start_time = time.time()
                    
                    sur_model_state_dict = sur_model.state_dict()
                    if attack != 'ht':                        
                        temp_sus[indices[index]] = torch.cat([diff.view(-1) for diff in sur_model_state_dict.values()])[important_features].cpu().numpy()
                    else:
                        temp_sus[indices[index]] = torch.cat([diff.view(-1).contiguous() for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy().copy()[important_features]
                
                step4_4_time = time.time() - start_time
                step_4_4_time_avg += step4_4_time
                start_time = time.time()
                
                
                if not sample_from_test: 
                    if len(extra_clean_indices) > 0:
                        clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
                        clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                        clean_indexes = np.concatenate([indices[clean_indices].cpu().numpy(), np.array(target_indices)[extra_clean_indices]])
                    else:
                        clean_batch = images[clean_indices]
                        clean_labels = labels[clean_indices]
                        clean_indexes = indices[clean_indices].cpu().numpy()
                else:
                    clean_batch = target_images[clean_indices]
                    clean_labels = torch.tensor([target_class] * len(clean_indices)).to(device)
                    clean_indexes = np.array(range(len(target_images)))[clean_indices]
               

                
                temp_clean_1_loader = DataLoader(list(zip(clean_batch, clean_labels, clean_indexes)), batch_size=1, shuffle=False)
                for image, label, index in temp_clean_1_loader:
                    
                    original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                    
                    
                    sur_model.load_state_dict(original_weights)
                    sur_model.train(mode = training_mode)
                    
                    sur_optimizer.load_state_dict(original_optimizer)
                    sur_optimizer.zero_grad()
                    
                    
                    output = sur_model(torch.cat([image, images[remaining_indices]]))
                    clean_label = label.long()
                    loss = criterion(output, torch.cat([clean_label, labels[remaining_indices]]))
                    loss.backward()
                    sur_optimizer.step()
                    
                    sur_model_state_dict = sur_model.state_dict()
                    if attack != 'ht':                        
                        temp_clean[index] = torch.cat([diff.view(-1) for diff in sur_model_state_dict.values()])[important_features].cpu().numpy()
                    else:
                        temp_clean[index]  = torch.cat([diff.view(-1).contiguous() for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy().copy()[important_features]
                    



                step5_time = time.time() - start_time
                start_time = time.time()
                step5_time_avg += step5_time
                
                

                start_time_2 = time.time()       
                if not sample_from_test: 
                    if len(extra_clean_indices_2) > 0:
                        clean_batch = torch.cat([images[clean_indices_2], target_images[extra_clean_indices_2]])
                        clean_labels = torch.cat([labels[clean_indices_2], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                        clean_indexes = np.concatenate([indices[clean_indices_2].cpu().numpy(), np.array(target_indices)[extra_clean_indices_2]])
                    else:
                        clean_batch = images[clean_indices_2]
                        clean_labels = labels[clean_indices_2]
                        clean_indexes = indices[clean_indices_2].cpu().numpy()
                else:
                    clean_batch = target_images[clean_indices_2]
                    clean_labels = torch.tensor([target_class] * len(clean_indices_2)).to(device)
                    clean_indexes = np.array(range(len(target_images)))[clean_indices_2]
                    
                    
                step5_1_time = time.time() - start_time_2
                step5_1_time_avg += step5_1_time
                
                clean_2_loader = DataLoader(list(zip(clean_batch, clean_labels)), batch_size=1, shuffle=False)
                
                temp_clean_2 = np.zeros(len(important_features))
                for image, label in clean_2_loader:
                    
                    
                    original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth')
                    
                    sur_model.load_state_dict(original_weights)
                    sur_model.train(mode = training_mode)
                    
                    sur_optimizer.load_state_dict(original_optimizer)
                    sur_optimizer.zero_grad()
                    
                    output = sur_model(torch.cat([image, images[remaining_indices]]))
                    clean_label = label.long()
                    loss = criterion(output, torch.cat([clean_label, labels[remaining_indices]]))
                    loss.backward()
                    sur_optimizer.step()
                    
                    sur_model_state_dict = sur_model.state_dict()
                    if attack != 'ht':                        
                        temp_clean_2 += torch.cat([diff.view(-1) for diff in sur_model_state_dict.values()])[important_features].cpu().numpy()
                    else:
                        temp_clean_2  += torch.cat([diff.view(-1).contiguous() for name, diff in sur_model_state_dict.items() if '20' in name]).cpu().numpy().copy()[important_features]                     
                temp_clean_2 /= len(clean_indices_2)
                
                step6_time = time.time() - start_time
                step6_time_avg += step6_time
                start_time = time.time()
                
                for index in temp_sus:
                    sus_diff[(epoch, index)] = temp_sus[index] - temp_clean_2 
                if not sample_from_test:
                    for index in temp_clean:
                        clean_diff[(epoch, index)] = temp_clean[index] - temp_clean_2
                else:
                    for index in temp_clean:
                        clean_diff[(epoch, index + 50000)] = temp_clean[index] - temp_clean_2
                
                step7_time = time.time() - start_time
                step7_time_avg += step7_time
                step7_time_avg += step7_time
                
                del temp_sus, temp_clean, temp_clean_2
                
                

        torch.cuda.empty_cache()       
        
        
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

        
        start_time = time.time()
        
        
        
        step7_time = time.time() - start_time
        step7_time_avg += step7_time
        
        # print("Step 1 time:", step1_time_avg)
        # print("Step 2 time:", step2_time_avg)
        # print("Step 3 time:", step3_time_avg)
        # print("Step 4_1 time:", step_4_1_time_avg)
        # print("Step 4_2 time:", step_4_2_time_avg)
        # print("Step 4_3 time:", step_4_3_time_avg)
        # print("Step 4_4 time:", step_4_4_time_avg)
        # print("Step 4_5 time:", step_4_5_time_avg)
        # print("Step 5 time:", step5_time_avg)
        # print("Step 5_1 time:", step5_1_time_avg)
        # print("Step 6 time:", step6_time_avg)
        # print("Step 7 time:", step7_time_avg)
        
        
        # Testing attack effect
        model.eval()
        correct, total = 0, 0
        for i, (images, labels) in enumerate(poisoned_test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        test_ACC.append(acc)
        print('\nAttack success rate %.2f' % (acc*100))
        print('Test_loss:',out_loss)
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:',out_loss)
    
    sus_inds = np.array([ind.item() for epoch, ind in sus_diff])
    clean_inds = np.array([ind.item() for epoch, ind in clean_diff])
    sus_diff = np.array([sus_diff[key] for key in sus_diff])
    clean_diff = np.array([clean_diff[key] for key in clean_diff])
    
    return sus_diff, clean_diff, sus_inds, clean_inds



# Function to train different models and get feature importances
def train_prov_data_custom(X_sus, X_clean, clean_igs_inds, sus_igs_inds, random_poison_idx, random_clean_sus_idx, n_groups, seed, device, model_name='RandomForest', verbose=True, training_mode=True, max_iters=1, confidence_threshold=0.7):
    """
    Train a model on the given provenance data.

    :param X_sus: Features for suspicious samples.
    :param X_clean: Features for clean samples.
    :param clean_igs_inds: Indices for clean IGs (Integrated Gradients).
    :param sus_igs_inds: Indices for suspicious IGs.
    :param n_groups: Number of groups to split the data into for cross-validation.
    :param model_name: Name of the model to be used ('RandomForest', 'LogisticRegression', 'LinearSVM', 'KernelSVM', 'MLP').
    :return: Various metrics and data from the training process.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # Label suspicious data as 1 and clean data as 0
    y_sus = np.ones(len(X_sus))
    y_clean = np.zeros(len(X_clean))
    # Combine the datasets
    X = np.concatenate([X_clean, X_sus])
    del X_sus, X_clean
    # X = np.clip(X, -1e5, 1e5)
    y = np.concatenate([y_clean, y_sus])
    # print(np.isinf(X).sum())
    assert not np.isinf(X).any()
    # X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    del y_clean, y_sus

    # print(X.shape, y.shape)  # Ensure shapes are as expected

    # # Convert to pandas DataFrame/Series
    # X_df = pd.DataFrame(X)
    # y_series = pd.Series(y, name='label')
    # clean_igs_inds_series = pd.Series(clean_igs_inds, name='clean_igs_inds')
    # sus_igs_inds_series = pd.Series(sus_igs_inds, name='sus_igs_inds')

    # # Combine into a single DataFrame
    # train_df = pd.concat([X_df, y_series, clean_igs_inds_series, sus_igs_inds_series], axis=1)

    # # Drop rows with NaN values
    # train_df.dropna(axis=0, inplace=True)

    # # Split back into separate arrays
    # X = train_df.iloc[:, :-3].values  # Features
    # y = train_df.iloc[:, -3].values  # Labels
    # clean_igs_inds = train_df.iloc[:, -2].values  # Clean IG indices
    # sus_igs_inds = train_df.iloc[:, -1].values  # Sus IG indices

    # # Check the shapes to ensure everything is correct
    # print(X.shape, y.shape, clean_igs_inds.shape, sus_igs_inds.shape)
    # Helper function to split images into groups
    def split_images_into_groups(image_indices, n_splits, seed=seed):
        """
        Randomly splits image indices into groups.

        :param image_indices: Array of image indices to split.
        :param n_splits: Number of groups to split into.
        :param seed: Seed for the random number generator.
        :return: Array of image index groups.
        """
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(image_indices)
        return np.array_split(shuffled_indices, n_splits)
    
    
    

    # Get unique indices for suspicious and clean images
    unique_sus_images = np.unique(sus_igs_inds)
    unique_clean_images = np.unique(clean_igs_inds)

    # Split the unique image indices into groups
    sus_image_groups = split_images_into_groups(unique_sus_images, n_groups)
    clean_image_groups = split_images_into_groups(unique_clean_images, n_groups)

    # Initialize lists to store results
    predictions, true_labels, predictions_proba = [], [], []
    group_feature_importances, index_tracker = [], []

    # Dictionary to track predictions with their corresponding indices
    predictions_with_indices = {}

    # Combine clean and suspicious indices for easier access
    concated_igs = np.concatenate([clean_igs_inds, sus_igs_inds])
    del clean_igs_inds, sus_igs_inds

    # Define models
    models = {
        'prf': prf(n_estimators=100, bootstrap=True, n_jobs=-1),
        'RandomForest': RandomForestClassifier(random_state=seed, n_jobs=-1, n_estimators=100, class_weight='balanced'), 
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=seed),
        'LinearSVM': SVC(kernel='linear', probability=True, random_state=seed),
        'KernelSVM': SVC(kernel='rbf', probability=True, random_state=seed)
    }
    
    
    param_grid = {
        'n_estimators': [100,  300, 500],
        'max_depth': [None, 10, 5],  
    }

    # Iterate through each group to perform cross-validation
    for i in range(n_groups):
        # Select test and train indices for this fold
        test_sus_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for img_idx in sus_image_groups[i]])
        train_sus_indices = np.concatenate([np.where(concated_igs  == img_idx)[0] for j, group in enumerate(sus_image_groups) if j != i for img_idx in group])
        
        test_clean_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for img_idx in clean_image_groups[i]])
        train_clean_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for j, group in enumerate(clean_image_groups) if j != i for img_idx in group]) 
        
        # Create training and testing sets
        train_indices = np.concatenate([train_clean_indices, train_sus_indices])
        test_indices = np.concatenate([test_clean_indices, test_sus_indices])
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train and evaluate the selected model
        X_labeled = np.empty((0, X_train.shape[1]))  # Empty array with correct feature dimensions
        y_labeled = np.empty((0,))
        

        pos_inds_train = [i for i, ind in enumerate(concated_igs[train_indices[y_train==1]]) if ind in random_poison_idx]
        pos_inds = np.array([ind for ind in concated_igs[train_indices[y_train==1]]])

        max_iters = 1
        confidence_threshold = 0.7
        m = 3

        if model_name in models:
            if model_name == "RandomForest":  
                clf = models[model_name]
                iteration = 0
                X_sus_temp = X_train[y_train == 1].copy()
                X_clean_temp = X_train[y_train == 0].copy()

                
                # if i == 0:
                #     grid_search = GridSearchCV(estimator=models[model_name], param_grid=param_grid, 
                #                     scoring='f1_weighted', cv=3)
                #     grid_search.fit(X_train, y_train)
                #     clf = grid_search.best_estimator_
                #     print(grid_search.best_params_)
                # else:
                best_params = {'max_depth':10, 'n_estimators': 200}
                clf = RandomForestClassifier(**best_params, n_jobs=-1, random_state=seed)
                # clf = RandomForestClassifier(**grid_search.best_params_, n_jobs=-1, random_state=seed)
                clf.fit(X_train, y_train) 
                
                while iteration < max_iters:
                    iteration += 1
                    
                    if iteration > 1:
                        confidence_threshold = 0.7
                    # Predict probabilities for all suspicious samples
                    y_proba = clf.predict_proba(X_sus_temp)[:, 1]
                    
                    # Identify high-confidence poisonous samples
                    high_conf_indices = np.where(y_proba > confidence_threshold)[0]
                    # confidence_threshold = 0.5
                    # high_conf_indices = np.array(pos_inds_train)

                    true_pos = set(pos_inds_train) & set(high_conf_indices)

                    if verbose:
                        print(f"Iteration {iteration}: {len(high_conf_indices)} high-conf , true pos {len(true_pos)}, total pos {len(pos_inds_train)} total sus {len(X_sus_temp)}")

                    # If no new high-confidence samples, stop
                    if len(high_conf_indices) == 0:
                        break

                    # Update labeled dataset with new high-confidence poisonous samples
                    X_labeled = np.concatenate([X_labeled, X_sus_temp[high_conf_indices]])                                                                                                      
                    y_labeled = np.concatenate([y_labeled, np.ones(len(high_conf_indices))])
                    # X_train_temp = np.concatenate([X_labeled, X_clean_temp[:len(X_labeled)]])

                    X_train_temp = np.concatenate([X_labeled, X_clean_temp[:len(X_labeled)*m]])
                    y_train_temp = np.concatenate([y_labeled, (len(X_train_temp) - len(X_labeled)) * [0]])                 

                    X_sus_temp = np.delete(X_sus_temp, high_conf_indices, axis=0)  
                    if len(X_sus_temp) == 0:
                        break
                    # print("X_sus_temp shape", X_sus_temp.shape)
                    # print("high_conf_indices shape", len(high_conf_indices))
                    # print("pos_inds shape", len(pos_inds_train))

                    remaining_indices = set(range(len(X_sus_temp))) - set(high_conf_indices)
                    pos_inds_train = [i for i, ind in enumerate(remaining_indices) if pos_inds[ind] in random_poison_idx]
                    pos_inds  = np.delete(pos_inds, high_conf_indices , axis=0)
                    # Train a new model on the updated labeled dataset
                    # smote = SMOTE()
                    # X_resampled, y_resampled = smote.fit_resample(X_train_temp, y_train_temp)
                    # clf.fit(X_resampled, y_resampled)
                    clf.fit(X_train_temp, y_train_temp)     

 
            else:
                if i == 0:
                    grid_search = GridSearchCV(estimator=models[model_name], param_grid=param_grid, 
                                    scoring='f1_weighted', cv=2)
                    grid_search.fit(X_train, y_train)
                    clf = grid_search.best_estimator_
                    print(grid_search.best_params_)
                else:
                    clf = RandomForestClassifier(**grid_search.best_params_, n_jobs=-1, random_state=seed)
                    clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

            if model_name in ['RandomForest', 'LogisticRegression', 'LinearSVM', 'prf']:
                feature_importances = clf.coef_[0] if model_name in ['LogisticRegression', 'LinearSVM'] else clf.feature_importances_
                group_feature_importances.append(feature_importances)
            else:
                # Use permutation importance for kernel SVM
                perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=seed)
                group_feature_importances.append(perm_importance.importances_mean)
        elif model_name == 'MLP':
            # Train MLP
            input_shape = X_train.shape[1]
            model = DNN(input_shape, 2)
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)    

            X_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            X_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

            X_train = DataLoader(X_train, batch_size=5000, shuffle=True)
            X_test = DataLoader(X_test, batch_size=5000, shuffle=False)

            model.train(mode = training_mode)
            for epoch in range(10):
                for inputs, labels in X_train:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if np.isnan(inputs.cpu()).any():
                        inputs = np.nan_to_num(inputs.cpu(), nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
                        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            model.eval()
            y_pred, y_pred_proba = [], []
            with torch.no_grad():
                for inputs, labels in X_test:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if np.isnan(inputs.cpu()).any():
                        inputs = np.nan_to_num(inputs.cpu(), nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
                        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    y_pred.extend(preds.cpu().numpy())
                    y_pred_proba.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            
            # Calculate feature importance using integrated gradients
            ig = IntegratedGradients(model)
            input_tensor = torch.tensor(X[train_indices][:128], dtype=torch.float32, requires_grad=True).to(device)
            attr, delta = ig.attribute(input_tensor, target=1, return_convergence_delta=True)
            feature_importances = attr.mean(dim=0).detach().cpu().numpy()
            group_feature_importances.append(feature_importances)
        else:
            raise ValueError(f"Model {model_name} is not supported")

        # Store results for analysis
        predictions.extend(y_pred)
        true_labels.extend(y_test)
        predictions_proba.extend(y_pred_proba)
        index_tracker.extend(test_indices)

        # Calculate recall for different subsets of data
        TPR = recall_score(y_test, y_pred)
        ACC = accuracy_score(y_test, y_pred)
        pos_inds = [i for i, ind in enumerate(concated_igs[test_indices]) if ind in random_poison_idx]
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        pos_recall = recall_score(y_test[pos_inds], y_pred[pos_inds])
        
        if verbose:
            sus_clean_inds = [i for i, ind in enumerate(concated_igs[test_indices]) if ind in random_clean_sus_idx]
            if len(sus_clean_inds) > 0:
                sus_clean_acc = recall_score(y_test[sus_clean_inds], y_pred[sus_clean_inds])
                print(f"Model: {model_name} - Group {i+1} Test Acc: {ACC:.4f} TPR: {TPR:.4f} Poison TPR: {pos_recall:.4f} Suspected Clean TPR: {sus_clean_acc:.4f}")
            else:
                print(f"Model: {model_name} - Group {i+1} Test TPR: {TPR:.4f} Acc: {ACC:.4f}")
        
        del X_train, X_test, y_train, y_test
        # Update the dictionary with new predictions
        for idx, pred in zip(test_indices, y_pred_proba):
            if concated_igs[idx] in predictions_with_indices:
                predictions_with_indices[concated_igs[idx]].append(pred)
            else: 
                predictions_with_indices[concated_igs[idx]] = [pred]
    
    # Calculate overall recall
    final_acc = accuracy_score(true_labels, predictions)
    final_tpr = recall_score(true_labels, predictions)
    if verbose:
        print(f"Final TPR: {final_tpr} Final Acc: {final_acc}")

    # Average the feature importances across all groups
    average_feature_importances = np.mean(group_feature_importances, axis=0)
    
    true_labels = np.array(true_labels)
    predictions_proba = np.array(predictions_proba)
    index_tracker = np.array(index_tracker)
    
    return predictions_with_indices, average_feature_importances, true_labels, predictions_proba, final_acc, index_tracker



def train_prov_data_co_lrn(X_sus, X_clean, clean_igs_inds, sus_igs_inds, random_poison_idx, random_clean_sus_idx, n_groups, seed, device, model_name='RandomForest', verbose=True):
    """
    Train a model on the given provenance data.

    :param X_sus: Features for suspicious samples.
    :param X_clean: Features for clean samples.
    :param clean_igs_inds: Indices for clean IGs (Integrated Gradients).
    :param sus_igs_inds: Indices for suspicious IGs.
    :param n_groups: Number of groups to split the data into for cross-validation.
    :param model_name: Name of the model to be used ('RandomForest', 'LogisticRegression', 'LinearSVM', 'KernelSVM', 'MLP').
    :return: Various metrics and data from the training process.
    """
    
    # Label suspicious data as 1 and clean data as 0
    y_sus = np.ones(len(X_sus))
    y_clean = np.zeros(len(X_clean))
    # Combine the datasets
    X = np.concatenate([X_clean, X_sus])
    del X_sus, X_clean
    # X = np.clip(X, -1e5, 1e5)
    y = np.concatenate([y_clean, y_sus])
    
    del y_clean, y_sus

    # print(X.shape, y.shape)  # Ensure shapes are as expected

    # # Convert to pandas DataFrame/Series
    # X_df = pd.DataFrame(X)
    # y_series = pd.Series(y, name='label')
    # clean_igs_inds_series = pd.Series(clean_igs_inds, name='clean_igs_inds')
    # sus_igs_inds_series = pd.Series(sus_igs_inds, name='sus_igs_inds')

    # # Combine into a single DataFrame
    # train_df = pd.concat([X_df, y_series, clean_igs_inds_series, sus_igs_inds_series], axis=1)

    # # Drop rows with NaN values
    # train_df.dropna(axis=0, inplace=True)

    # # Split back into separate arrays
    # X = train_df.iloc[:, :-3].values  # Features
    # y = train_df.iloc[:, -3].values  # Labels
    # clean_igs_inds = train_df.iloc[:, -2].values  # Clean IG indices
    # sus_igs_inds = train_df.iloc[:, -1].values  # Sus IG indices

    # # Check the shapes to ensure everything is correct
    # print(X.shape, y.shape, clean_igs_inds.shape, sus_igs_inds.shape)
    # Helper function to split images into groups
    def split_images_into_groups(image_indices, n_splits, seed=seed):
        """
        Randomly splits image indices into groups.

        :param image_indices: Array of image indices to split.
        :param n_splits: Number of groups to split into.
        :param seed: Seed for the random number generator.
        :return: Array of image index groups.
        """
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(image_indices)
        return np.array_split(shuffled_indices, n_splits)


    # Get unique indices for suspicious and clean images
    unique_sus_images = np.unique(sus_igs_inds)
    unique_clean_images = np.unique(clean_igs_inds)

    # Split the unique image indices into groups
    sus_image_groups = split_images_into_groups(unique_sus_images, n_groups)
    clean_image_groups = split_images_into_groups(unique_clean_images, n_groups)

    # Initialize lists to store results
    predictions, true_labels, predictions_proba = [], [], []
    group_feature_importances, index_tracker = [], []

    # Dictionary to track predictions with their corresponding indices
    predictions_with_indices = {}

    # Combine clean and suspicious indices for easier access
    concated_igs = np.concatenate([clean_igs_inds, sus_igs_inds])
    del clean_igs_inds, sus_igs_inds

    # Define models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=seed),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=seed),
        'LinearSVM': SVC(kernel='linear', probability=True, random_state=seed),
        'KernelSVM': SVC(kernel='rbf', probability=True, random_state=seed)
    }

    # Iterate through each group to perform cross-validation
    for i in range(n_groups):
        # Select test and train indices for this fold
        test_sus_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for img_idx in sus_image_groups[i]])
        train_sus_indices = np.concatenate([np.where(concated_igs  == img_idx)[0] for j, group in enumerate(sus_image_groups) if j != i for img_idx in group])
        
        test_clean_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for img_idx in clean_image_groups[i]])
        train_clean_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for j, group in enumerate(clean_image_groups) if j != i for img_idx in group]) 
        
        # Create training and testing sets
        train_indices = np.concatenate([train_clean_indices, train_sus_indices])
        test_indices = np.concatenate([test_clean_indices, test_sus_indices])
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train and evaluate the selected model
        if model_name in models:
            clf = models[model_name]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

            if model_name in ['RandomForest', 'LogisticRegression', 'LinearSVM']:
                feature_importances = clf.coef_[0] if model_name in ['LogisticRegression', 'LinearSVM'] else clf.feature_importances_
                group_feature_importances.append(feature_importances)
            else:
                # Use permutation importance for kernel SVM
                perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=seed)
                group_feature_importances.append(perm_importance.importances_mean)
        elif model_name == 'MLP':
            
            # Evaluate the Model
            def evaluate(test_loader, model1, model2):
                # Model 1 evaluation
                model1.eval()
                y_pred_proba_1 = []
                correct1 = 0
                total1 = 0
                TP1, FP1, TN1, FN1 = 0, 0, 0, 0
                
                for images, labels in test_loader:
                    images = Variable(images).to(device)
                    labels = labels.to(device)
                    logits1 = model1(images)
                    outputs1 = F.softmax(logits1, dim=1)
                    _, pred1 = torch.max(outputs1.data, 1)
                    total1 += labels.size(0)
                    correct1 += (pred1.cpu() == labels.cpu()).sum()
                    y_pred_proba_1.extend(outputs1[:, 1].detach().cpu().numpy())
                    
                    # Calculate confusion matrix components
                    TP1 += ((pred1 == 1) & (labels == 1)).sum().item()
                    FP1 += ((pred1 == 1) & (labels == 0)).sum().item()
                    TN1 += ((pred1 == 0) & (labels == 0)).sum().item()
                    FN1 += ((pred1 == 0) & (labels == 1)).sum().item()

                # Model 2 evaluation
                model2.eval()
                y_pred_proba_2 = []
                correct2 = 0
                total2 = 0
                TP2, FP2, TN2, FN2 = 0, 0, 0, 0
                
                for images, labels in test_loader:
                    images = Variable(images).to(device)
                    labels = labels.to(device)
                    logits2 = model2(images)
                    outputs2 = F.softmax(logits2, dim=1)
                    _, pred2 = torch.max(outputs2.data, 1)
                    total2 += labels.size(0)
                    correct2 += (pred2.cpu() == labels.cpu()).sum()
                    y_pred_proba_2.extend(outputs2[:, 1].detach().cpu().numpy())
                    
                    # Calculate confusion matrix components
                    TP2 += ((pred2 == 1) & (labels == 1)).sum().item()
                    FP2 += ((pred2 == 1) & (labels == 0)).sum().item()
                    TN2 += ((pred2 == 0) & (labels == 0)).sum().item()
                    FN2 += ((pred2 == 0) & (labels == 1)).sum().item()

                # Calculate accuracies
                acc1 = 100 * float(correct1) / float(total1)
                acc2 = 100 * float(correct2) / float(total2)
                
                # Calculate TPR and FPR for model 1
                TPR1 = 100 * TP1 / (TP1 + FN1) if (TP1 + FN1) > 0 else 0
                FPR1 = 100 * FP1 / (FP1 + TN1) if (FP1 + TN1) > 0 else 0

                # Calculate TPR and FPR for model 2
                TPR2 = 100 * TP2 / (TP2 + FN2) if (TP2 + FN2) > 0 else 0
                FPR2 = 100 * FP2 / (FP2 + TN2) if (FP2 + TN2) > 0 else 0
                
                if FPR1 < FPR2:
                    y_pred_proba = y_pred_proba_1
                    y_pred = [1 if proba > 0.5 else 0 for proba in y_pred_proba]
                else:
                    y_pred_proba = y_pred_proba_2
                    y_pred = [1 if proba > 0.5 else 0 for proba in y_pred_proba]
                
                return acc1, acc2, y_pred, y_pred_proba
            
           
            def accuracy(logit, target, topk=(1,)):
                """Computes the precision@k for the specified values of k"""
                output = F.softmax(logit, dim=1)
                maxk = max(topk)
                batch_size = target.size(0)

                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))

                res = []
                for k in topk:
                    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                    res.append(correct_k.mul_(100.0 / batch_size))
                return res
            
            def loss_coteaching(y_1, y_2, t, forget_rate):
                loss_1 = F.cross_entropy(y_1, t, reduce = False)
                ind_1_sorted = np.argsort(loss_1.data.cpu()).to(device)
                loss_1_sorted = loss_1[ind_1_sorted]

                loss_2 = F.cross_entropy(y_2, t, reduce = False)
                ind_2_sorted = np.argsort(loss_2.data.cpu()).to(device)
                loss_2_sorted = loss_2[ind_2_sorted]

                remember_rate = 1 - forget_rate
                num_remember = int(remember_rate * len(loss_1_sorted))

                ind_1_update=ind_1_sorted[:num_remember]
                ind_2_update=ind_2_sorted[:num_remember]
                # exchange
                loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
                loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

                return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember
            
            # Train the Model
            def train_cl(train_loader,epoch, model1, optimizer1, model2, optimizer2):
                # print 'Training %s...' % model_str
                print('Training...')
                pure_ratio_list=[]
                pure_ratio_1_list=[]
                pure_ratio_2_list=[]
                
                train_total=0
                train_correct=0 
                train_total2=0
                train_correct2=0 

                for i, (images, labels) in enumerate(train_loader):
                    if i>num_iter_per_epoch:
                        break
                
                    images = Variable(images).cuda()
                    labels = Variable(labels).cuda()
                    
                    # Forward + Backward + Optimize
                    logits1=model1(images)
                    prec1 = accuracy(logits1, labels)[0]
                    train_total+=1
                    train_correct+=prec1

                    logits2 = model2(images)
                    prec2 = accuracy(logits2, labels)[0]
                    train_total2+=1
                    train_correct2+=prec2
                    loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch])
                    # pure_ratio_1_list.append(100*pure_ratio_1)
                    # pure_ratio_2_list.append(100*pure_ratio_2)

                    optimizer1.zero_grad()
                    loss_1.backward()
                    optimizer1.step()
                    optimizer2.zero_grad()
                    loss_2.backward()
                    optimizer2.step()
                    if (i+1) % print_freq == 0:
                        print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f'
                            %(epoch+1, n_epoch, i+1, len(X_train)//batch_size, prec1, prec2, loss_1.item(), loss_2.item()))

                train_acc1=float(train_correct)/float(train_total)
                train_acc2=float(train_correct2)/float(train_total2)
                return train_acc1, train_acc2

            def adjust_learning_rate(optimizer, epoch):
                for param_group in optimizer.param_groups:
                    param_group['lr']=alpha_plan[epoch]
                    param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
            
            input_shape = X_train.shape[1]          
            # Train MLP
            X_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            X_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
            batch_size = 5000
            X_train = DataLoader(X_train, batch_size=batch_size, shuffle=True)
            X_test = DataLoader(X_test, batch_size=batch_size, shuffle=False)
            
            learning_rate = 0.0001
            dnn1 = DNN(input_shape, n_outputs=2)
            dnn1.to(device)
            optimizer1 = optim.Adam(dnn1.parameters(), lr=learning_rate)  
            # optimizer1 = optim.SGD(dnn1.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
            num_iter_per_epoch = len(X_train)
            dnn2 = DNN(input_shape, n_outputs=2)
            dnn2.to(device)
            optimizer2 = optim.Adam(dnn2.parameters(), lr=learning_rate)
            # optimizer2 = optim.SGD(dnn2.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
            mean_pure_ratio1=0
            mean_pure_ratio2=0
            forget_rate = 0.5
            print_freq = 200
            
            print('epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2')
            n_epoch = 30
            epoch=0
            train_acc1=0
            train_acc2=0
            mom1 = 0.9
            mom2 = 0.1
            alpha_plan = [learning_rate] * n_epoch
            beta1_plan = [mom1] * n_epoch
            rate_schedule = np.ones(n_epoch)*forget_rate
            test_acc1, test_acc2, _, _ = evaluate(X_test, dnn1, dnn2)

            print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1, n_epoch, len(X_test), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
            for epoch in range(1, n_epoch):
                # train models
                dnn1.train(mode = True)
                adjust_learning_rate(optimizer1, epoch)
                dnn2.train(mode = True)
                adjust_learning_rate(optimizer2, epoch)
                train_acc1, train_acc2 = train_cl(X_train, epoch, dnn1, optimizer1, dnn2, optimizer2)
                # evaluate models
                test_acc1, test_acc2, _, _= evaluate(X_test, dnn1, dnn2)
                # save results
                # mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
                # mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
                print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%,' % (epoch+1, n_epoch, len(X_test), test_acc1, test_acc2))
            
            
            test_acc1, test_acc2, y_pred, y_pred_proba = evaluate(X_test, dnn1, dnn2)
            
            # model.train(mode = training_mode)
            # for epoch in range(10):
            #     for inputs, labels in X_train:
            #         inputs, labels = inputs.to(device), labels.to(device)
            #         if np.isnan(inputs.cpu()).any():
            #             inputs = np.nan_to_num(inputs.cpu(), nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
            #             inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            #         optimizer.zero_grad()
            #         outputs = model(inputs)
            #         loss = criterion(outputs, labels)
            #         loss.backward()
            #         optimizer.step()

            # model.eval()
            # y_pred, y_pred_proba = [], []
            # with torch.no_grad():
            #     for inputs, labels in X_test:
            #         inputs, labels = inputs.to(device), labels.to(device)
            #         if np.isnan(inputs.cpu()).any():
            #             inputs = np.nan_to_num(inputs.cpu(), nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
            #             inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            #         outputs = model(inputs)
            #         _, preds = torch.max(outputs, 1)
            #         y_pred.extend(preds.cpu().numpy())
            #         y_pred_proba.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            
            # Calculate feature importance using integrated gradients
            ig = IntegratedGradients(dnn1)
            input_tensor = torch.tensor(X[train_indices][:128], dtype=torch.float32, requires_grad=True).to(device)
            attr, delta = ig.attribute(input_tensor, target=1, return_convergence_delta=True)
            feature_importances = attr.mean(dim=0).detach().cpu().numpy()
            
            ig = IntegratedGradients(dnn2)
            attr, delta = ig.attribute(input_tensor, target=1, return_convergence_delta=True)
            feature_importances = np.mean([feature_importances, attr.mean(dim=0).detach().cpu().numpy()], axis=0)
            group_feature_importances.append(feature_importances)
        else:
            raise ValueError(f"Model {model_name} is not supported")

        # Store results for analysis
        predictions.extend(y_pred)
        true_labels.extend(y_test)
        predictions_proba.extend(y_pred_proba)
        index_tracker.extend(test_indices)

        # Calculate recall for different subsets of data
        TPR = recall_score(y_test, y_pred)
        ACC = accuracy_score(y_test, y_pred)
        pos_inds = [i for i, ind in enumerate(concated_igs[test_indices]) if ind in random_poison_idx]
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        pos_recall = recall_score(y_test[pos_inds], y_pred[pos_inds])
        
        if verbose:
            sus_clean_inds = [i for i, ind in enumerate(concated_igs[test_indices]) if ind in random_clean_sus_idx]
            if len(sus_clean_inds) > 0:
                sus_clean_acc = recall_score(y_test[sus_clean_inds], y_pred[sus_clean_inds])
                print(f"Model: {model_name} - Group {i+1} Test Acc: {ACC:.4f} TPR: {TPR:.4f} Poison TPR: {pos_recall:.4f} Suspected Clean TPR: {sus_clean_acc:.4f}")
            else:
                print(f"Model: {model_name} - Group {i+1} Test TPR: {TPR:.4f} Acc: {ACC:.4f}")
        
        del X_train, X_test, y_train, y_test
        # Update the dictionary with new predictions
        for idx, pred in zip(test_indices, y_pred_proba):
            if concated_igs[idx] in predictions_with_indices:
                predictions_with_indices[concated_igs[idx]].append(pred)
            else: 
                predictions_with_indices[concated_igs[idx]] = [pred]
    
    # Calculate overall recall
    final_acc = accuracy_score(true_labels, predictions)
    final_tpr = recall_score(true_labels, predictions)
    if verbose:
        print(f"Final TPR: {final_tpr} Final Acc: {final_acc}")

    # Average the feature importances across all groups
    average_feature_importances = np.mean(group_feature_importances, axis=0)
    
    true_labels = np.array(true_labels)
    predictions_proba = np.array(predictions_proba)
    index_tracker = np.array(index_tracker)
    
    return predictions_with_indices, average_feature_importances, true_labels, predictions_proba, final_acc, index_tracker



# Training function on the Provenance Data 
def train_prov_data_dnn(X_sus, X_clean, clean_igs_inds, sus_igs_inds, random_poison_idx, random_clean_sus_idx, n_groups, seed, verbose=True):
    y_sus = np.ones(len(X_sus))
    y_clean = np.zeros(len(X_clean))
    X = np.concatenate([X_clean, X_sus])
    y = np.concatenate([y_clean, y_sus])
    
    def split_images_into_groups(image_indices, n_splits, seed=seed):
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(image_indices)
        return np.array_split(shuffled_indices, n_splits)

    unique_sus_images = np.unique(sus_igs_inds)
    unique_clean_images = np.unique(clean_igs_inds)
    sus_image_groups = split_images_into_groups(unique_sus_images, n_groups)
    clean_image_groups = split_images_into_groups(unique_clean_images, n_groups)

    predictions, true_labels, predictions_proba = [], [], []
    group_feature_importances, index_tracker = [], []
    predictions_with_indices = {}
    concated_igs = np.concatenate([clean_igs_inds, sus_igs_inds])

    for i in range(n_groups):
        test_sus_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for img_idx in sus_image_groups[i]])
        train_sus_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for j, group in enumerate(sus_image_groups) if j != i for img_idx in group])
        
        test_clean_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for img_idx in clean_image_groups[i]])
        train_clean_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for j, group in enumerate(clean_image_groups) if j != i for img_idx in group]) 
        
        train_indices = np.concatenate([train_clean_indices, train_sus_indices])
        test_indices = np.concatenate([test_clean_indices, test_sus_indices])
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        model = DNN(input_size=X_train.shape[1], num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train(mode = training_mode)
        for epoch in range(10):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        model.eval()
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        y_pred = np.array(all_preds)
        y_pred_proba = np.array(all_probs)

        predictions.extend(y_pred)
        true_labels.extend(y_test)
        predictions_proba.extend(y_pred_proba)
        index_tracker.extend(test_indices)
        
        ig = IntegratedGradients(model)
        X_train_tensor.requires_grad_()
        attr = ig.attribute(X_train_tensor, target=1)
        feature_importances = attr.mean(dim=0).detach().cpu().numpy()  
        group_feature_importances.append(feature_importances)
        

        TPR = recall_score(y_test, y_pred)
        ACC = accuracy_score(y_test, y_pred)
        pos_inds = [i for i, ind in enumerate(concated_igs[test_indices]) if ind in random_poison_idx]
        pos_recall = recall_score(y_test[pos_inds], y_pred[pos_inds])
        
        
        
        if verbose:
            sus_clean_inds = [i for i, ind in enumerate(concated_igs[test_indices]) if ind in random_clean_sus_idx]
            if len(sus_clean_inds) > 0:
                sus_clean_acc = recall_score(y_test[sus_clean_inds], y_pred[sus_clean_inds])
                print(f"Group {i+1} Test Acc: {ACC:.4f}:  TPR: {TPR:.4f} Poison TPR: {pos_recall:.4f} Suspected Clean TPR: {sus_clean_acc:.4f}")
            else:
                print(f"Group {i+1} Test TPR: {TPR:.4f} Acc: {ACC:.4f}")
        
        for idx, pred in zip(test_indices, y_pred_proba):
            if concated_igs[idx] in predictions_with_indices:
                predictions_with_indices[concated_igs[idx]].append(pred)
            else: 
                predictions_with_indices[concated_igs[idx]] = [pred]

    final_acc = accuracy_score(true_labels, predictions)
    final_tpr = recall_score(true_labels, predictions)
    if verbose:
        print(f"Final TPR: {final_tpr} Final Acc: {final_acc}")

    average_feature_importances = np.mean(group_feature_importances, axis=0)
    true_labels = np.array(true_labels)
    predictions_proba = np.array(predictions_proba)
    index_tracker = np.array(index_tracker)
    
    return predictions_with_indices, average_feature_importances, true_labels, predictions_proba, final_acc, index_tracker

# Training function on the Provenance Data 
def train_prov_data(X_sus, X_clean, clean_igs_inds, sus_igs_inds, random_poison_idx, random_clean_sus_idx, n_groups, seed, verbose=True):
    """
    Train a model on the given provenance data.

    :param X_sus: Features for suspicious samples.
    :param X_clean: Features for clean samples.
    :param clean_igs_inds: Indices for clean IGs (Integrated Gradients).
    :param sus_igs_inds: Indices for suspicious IGs.
    :param n_groups: Number of groups to split the data into for cross-validation.
    :return: Various metrics and data from the training process.
    """
    # Label suspicious data as 1 and clean data as 0
    y_sus = np.ones(len(X_sus))
    y_clean = np.zeros(len(X_clean))

    # Combine the datasets
    X = np.concatenate([X_clean, X_sus])
    y = np.concatenate([y_clean, y_sus])
    
    # Helper function to split images into groups
    def split_images_into_groups(image_indices, n_splits, seed=seed):
        """
        Randomly splits image indices into groups.

        :param image_indices: Array of image indices to split.
        :param n_splits: Number of groups to split into.
        :param seed: Seed for the random number generator.
        :return: Array of image index groups.
        """
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(image_indices)
        return np.array_split(shuffled_indices, n_splits)

    # Get unique indices for suspicious and clean images
    unique_sus_images = np.unique(sus_igs_inds)
    unique_clean_images = np.unique(clean_igs_inds)

    # Split the unique image indices into groups
    sus_image_groups = split_images_into_groups(unique_sus_images, n_groups)
    clean_image_groups = split_images_into_groups(unique_clean_images, n_groups)

    # Initialize lists to store results
    predictions, true_labels, predictions_proba = [], [], []
    group_feature_importances, index_tracker = [], []

    # Dictionary to track predictions with their corresponding indices
    predictions_with_indices = {}

    # Combine clean and suspicious indices for easier access
    concated_igs = np.concatenate([clean_igs_inds, sus_igs_inds])

    # Iterate through each group to perform cross-validation
    for i in range(n_groups):
        # Select test and train indices for this fold
        test_sus_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for img_idx in sus_image_groups[i]])
        train_sus_indices = np.concatenate([np.where(concated_igs  == img_idx)[0] for j, group in enumerate(sus_image_groups) if j != i for img_idx in group])
        
        test_clean_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for img_idx in clean_image_groups[i]])
        train_clean_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for j, group in enumerate(clean_image_groups) if j != i for img_idx in group]) 
        
        # Create training and testing sets
        train_indices = np.concatenate([train_clean_indices, train_sus_indices])
        test_indices = np.concatenate([test_clean_indices, test_sus_indices])
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train a RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        clf.fit(X_train, y_train)

        # Predict using the trained model
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Store results for analysis
        predictions.extend(y_pred)
        true_labels.extend(y_test)
        predictions_proba.extend(y_pred_proba)
        index_tracker.extend(test_indices)
        group_feature_importances.append(clf.feature_importances_)

        # Calculate recall for different subsets of data
        TPR = recall_score(y_test, y_pred)
        ACC = accuracy_score(y_test, y_pred)
        pos_inds = [i for i, ind in enumerate(concated_igs[test_indices]) if ind in random_poison_idx]
        pos_recall = recall_score(y_test[pos_inds], y_pred[pos_inds])
        
        if verbose==True:
            sus_clean_inds = [i for i, ind in enumerate(concated_igs[test_indices]) if ind in random_clean_sus_idx]
            if len(sus_clean_inds) > 0:
                sus_clean_acc = recall_score(y_test[sus_clean_inds], y_pred[sus_clean_inds])
                print(f"Group {i+1} Test Acc: {ACC:.4f}:  TPR: {TPR:.4f} Poison TPR: {pos_recall:.4f} Suspected Clean TPR: {sus_clean_acc:.4f}")
            else:
                print(f"Group {i+1} Test TPR: {TPR:.4f} Acc: {ACC:.4f}")
        
        # Update the dictionary with new predictions
        for idx, pred in zip(test_indices, y_pred_proba):
            if concated_igs[idx] in predictions_with_indices:
                predictions_with_indices[concated_igs[idx]].append(pred)
            else: 
                predictions_with_indices[concated_igs[idx]] = [pred]
    
    # Calculate overall recall
    final_acc = accuracy_score(true_labels, predictions)
    final_tpr = recall_score(true_labels, predictions)
    if verbose == True:
        print(f"Final TPR: {final_tpr} Final Acc: {final_acc}")

    # Average the feature importances across all groups
    average_feature_importances = np.mean(group_feature_importances, axis=0)
    
    true_labels = np.array(true_labels)
    predictions_proba = np.array(predictions_proba)
    index_tracker = np.array(index_tracker)
    
    return predictions_with_indices, average_feature_importances, true_labels, predictions_proba, final_acc, index_tracker

def get_relevant_weight_dimensions_sample_level(sus_diff, clean_diff, sus_inds, clean_inds, poison_indices, random_sus_idx, pr_sus, seed, device, figure_path = "./figures/", min_features=2, max_features=100):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    
    random_clean_sus_idx = list(set(np.unique(sus_inds)) - set(poison_indices))
    assert abs(len(poison_indices) - int(pr_sus/100 * len(random_sus_idx))) < 3
    print("len clean indices: ", len(clean_inds), "len sus indices: ", len(sus_inds))
    predictions_with_indices, average_feature_importances, true_labels, predictions_proba, _, index_tracker = train_prov_data_custom(
        sus_diff, clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx,5, seed, device, model_name='MLP') 

    average_original_feature_importances = average_feature_importances
    
    
    z_scores = np.abs(stats.zscore(average_original_feature_importances))
    plt.figure()
    plt.plot(range(len(z_scores)), z_scores, label='Feature Importances', alpha=0.5, color='blue')
    plt.savefig(figure_path + f"Feature_importances.png")
    outliers = np.where(z_scores > 1)[0]
    print("len outliers: ", len(outliers))  

    acc_values = []
    
    max_features = min(max_features, len(outliers))
    # relevant_feature_space = list(range(min_features,max_features+1, 1))
    # if max_features > 100:
    #     relevant_feature_space = list(range(min_features,100, 1)) + list(range(100, max_features, 10))
    # for relevant_feature_num in relevant_feature_space:
    #     relevant_features = np.argsort(z_scores)[::-1][:relevant_feature_num]
    #     _, _, _, _, final_acc, _ = train_prov_data(
    #             sus_diff[:,relevant_features], clean_diff[:,relevant_features], clean_inds, sus_inds, poison_indices,random_clean_sus_idx, 5, seed, verbose=False) 
    #     acc_values.append(final_acc)
    
    print("Training using the most relevant weight parameters based on feature importance")
    # best_rel_feature = relevant_feature_space[np.argmax(acc_values)] 
    best_rel_feature = len(outliers)
    relevant_features = np.argsort(average_original_feature_importances)[::-1][:best_rel_feature]
    print("Number of final relevant features: ", len(relevant_features))
    predictions_with_indices_2, _, true_labels_2, predictions_proba_2, _, index_tracker_2 = train_prov_data_custom(
            sus_diff[:,relevant_features], clean_diff[:,relevant_features], clean_inds, sus_inds, poison_indices,random_clean_sus_idx, 5, seed, device, model_name='MLP') 
    
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    # X = np.concatenate([clean_diff[:,relevant_features], sus_diff[:,relevant_features]])
    # y = np.concatenate([np.zeros(len(clean_diff)), np.ones(len(sus_diff))])
    # refined_X_sus = X[index_tracker_2][true_labels_2 == 1][predictions_proba_2[true_labels_2 == 1] > 0.8]
    # refined_X_clean = X[index_tracker_2][true_labels_2 == 0][:len(refined_X_sus)]
    # refined_X = np.concatenate([refined_X_clean, refined_X_sus])
    # refined_y = np.concatenate([np.zeros(len(refined_X_clean)), np.ones(len(refined_X_sus))])
    # print("len refined_X: ", len(refined_X))
    # rf.fit(refined_X, refined_y)
    
    # mlp = MLP(input_size=len(relevant_features), num_classes=2)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    # mlp.to(device)
    # X = np.concatenate([clean_diff[:,relevant_features], sus_diff[:,relevant_features]])
    # y = np.concatenate([np.zeros(len(clean_diff)), np.ones(len(sus_diff))])
    # refined_X_sus = X[index_tracker_2][true_labels_2 == 1][predictions_proba_2[true_labels_2 == 1] > 0.6]
    # refined_X_clean = X[index_tracker_2][true_labels_2 == 0][:len(refined_X_sus)]
    # refined_X = np.concatenate([refined_X_clean, refined_X_sus])
    # refined_y = np.concatenate([np.zeros(len(refined_X_clean)), np.ones(len(refined_X_sus))])
    # refined_X_tensor = torch.tensor(refined_X, dtype=torch.float32).to(device)
    # refined_y_tensor = torch.tensor(refined_y, dtype=torch.long).to(device)
    # train_dataset = TensorDataset(refined_X_tensor, refined_y_tensor)
    # train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    # mlp.train(mode = training_mode)
    
    # for epoch in range(10):
    #     for inputs, labels in train_loader:
    #         optimizer.zero_grad()
    #         outputs = mlp(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    
    plt.figure()
    plt.scatter(range(len(predictions_proba_2[true_labels_2==0])), predictions_proba_2[true_labels_2==0], alpha=0.5, color='blue')
    concat_indices = clean_inds + sus_inds
    is_sus_pos = np.array([idx in poison_indices for idx in concat_indices])[index_tracker_2]
    # num_poisons = np.array([len(set(idxs) & set(poison_indices)) for idxs in concat_indices])[index_tracker_2][true_labels_2==1]
    # sizes = num_poisons * 20 
    # cmap = plt.get_cmap('plasma')
    # normalize = plt.Normalize(vmin=min(num_poisons), vmax=max(num_poisons))
    # colors = cmap(normalize(num_poisons))

    pred_proba_sus_clean = predictions_proba_2[~is_sus_pos][true_labels_2[~is_sus_pos] == 1]
    pred_proba_sus_pos = predictions_proba_2[is_sus_pos][true_labels_2[is_sus_pos] == 1]
    plt.scatter(range(len(pred_proba_sus_clean)), pred_proba_sus_clean, alpha=0.5, color='green')
    plt.scatter(range(len(pred_proba_sus_pos)), pred_proba_sus_pos, alpha=0.5, color='red')
    # scatter = plt.scatter(range(len(predictions_proba_2)), predictions_proba_2, alpha=0.5, s=sizes, c='blue', label='Predictions Probabilities')
    # # Add color bar
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Number of Poisons')
    
    plt.title("Predictions probabilities for each class")
    plt.legend()
    plt.savefig(figure_path + f"Predictions_proba.png")
    return relevant_features, mlp



def get_relevant_weight_dimensions(sus_diff, clean_diff, sus_inds, clean_inds, poison_indices, random_sus_idx, pr_sus, pr_tgt, bs_bl, attack,dataset, cv_model, seed, device, eps, figure_path = "./figures/", min_features=2, max_features=100): 
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # print("sus indices: ", sus_inds)
    sus_inds_flat = np.concatenate(sus_inds)
    clean_inds_flat = np.concatenate(clean_inds) 
    print("len unique sus indices: ", len(np.unique(sus_inds_flat)), "len unique clean indices: ", len(np.unique(clean_inds_flat)))
    # assert set(np.unique(sus_inds_flat)) & set(np.unique(clean_inds_flat)) == set() 
    
    # clean_inds = [[ind + 50000 for ind in inds] for inds in clean_inds]  
    random_clean_sus_idx = list(set(np.unique(sus_inds_flat)) - set(poison_indices))
    assert abs(len(poison_indices) - int(pr_sus/100 * len(random_sus_idx))) < 3
    clean_inds_idv =  np.array([np.max(idxs) if len(idxs) > 0 else 50001 for idxs in clean_inds])
    concat_indices = clean_inds + sus_inds 
    del clean_inds_flat, clean_inds 
    temp_sus_inds = []
    count_clean = 0
    for idxs in sus_inds:
        if set(idxs) & set(poison_indices) == set():
            temp_sus_inds.append(np.max(idxs))
            count_clean += 1
        else:
            temp_sus_inds.append(np.max(list(set(idxs) & set(poison_indices))))
    assert np.isnan(sus_diff).sum() == 0
    
    
    print("count clean: ", count_clean)
    sus_inds_idv = np.array(temp_sus_inds)
    print("len clean indices: ", clean_inds_idv.shape, "len sus indices: ", sus_inds_idv.shape, "len clean sus indices: ", len(random_clean_sus_idx))
    print("pos indices in sus: ", len(set(sus_inds_idv) & set(poison_indices)), "len sus idv indices: ",len(np.unique(sus_inds_flat)))
    del sus_inds, sus_inds_flat 
    predictions_with_indices, average_feature_importances, true_labels, predictions_proba, _, index_tracker = train_prov_data_custom(
        sus_diff, clean_diff, clean_inds_idv, sus_inds_idv, poison_indices, random_clean_sus_idx,5, seed, device, cv_model) 
    # # attack  = 'ht'
    # pr_tgt = 0.5
    # pr_sus = 50
    prov_path = "./Training_Prov_Data/"
    # with open(prov_path + f'relevant_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{512}_check_2.pkl', 'rb') as f:
    #     relevant_features = pickle.load(f)
    # # prov_path = "./Training_Prov_Data/"
    # with open(prov_path + f'important_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{512}_check_2.pkl', 'rb') as f:
    #         important_features = pickle.load(f)
    # relevant_features = [np.where(important_features == f)[0][0] for f in relevant_features]
    
   
    with open(prov_path + f'avg_feature_importances_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_rf.pkl', 'wb') as f:
        pickle.dump(average_feature_importances, f)
    # with open(prov_path + f'avg_feature_importances_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}.pkl', 'rb') as f:
    #     average_feature_importances = pickle.load(f)
        
        
    
    average_original_feature_importances = average_feature_importances
    z_scores = np.abs(stats.zscore(average_original_feature_importances))
    plt.figure()
    plt.plot(range(len(z_scores)), z_scores, label='Feature Importances', alpha=0.5, color='blue')
    plt.savefig(figure_path + f"Feature_importances.png")
    z_num = 1
    outliers = np.where(z_scores > z_num)[0]
    print("len outliers: ", len(outliers))  
    best_rel_feature = len(outliers)
    relevant_features = np.argsort(average_original_feature_importances)[::-1][:best_rel_feature]
    print("Number of final relevant features: ", len(relevant_features))
        
    
    # acc_values = []
    # relevant_feature_space = list(range(min_features,max_features+1, 1))
    # if max_features > 100:
    #     relevant_feature_space = list(range(min_features,100, 1)) + list(range(100, max_features, 10))
    # for relevant_feature_num in relevant_feature_space:
    #     relevant_features = np.argsort(z_scores)[::-1][:relevant_feature_num]
    #     _, _, _, _, final_acc, _ = train_prov_data(
    #             sus_diff[:,relevant_features], clean_diff[:,relevant_features], clean_inds_idv, sus_inds_idv, poison_indices,random_clean_sus_idx, 5, seed, verbose=False) 
    #     acc_values.append(final_acc)
    # print("Training using the most relevant weight parameters based on feature importance")
    # best_rel_feature = relevant_feature_space[np.argmax(acc_values)]
    
    
    # with open(prov_path + f'relevant_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_z_score_{z_num}.pkl', 'wb') as f:
        # pickle.dump(relevant_features, f)
    # with open(prov_path + f'relevant_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_z_score_{z_num}.pkl', 'rb') as f:
    #     relevant_features = pickle.load(f)
    # print("Number of final relevant features: ", len(relevant_features))
    
    # prov_path = "../../../../../data/phil_data/Training_Prov_Data/"
    # with open(prov_path + f'rf_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{128}_st.pkl', 'rb') as f:
    #     rf = pickle.load(f)

    
    
    
        
    # X = np.concatenate([clean_diff[:,relevant_features], sus_diff[:,relevant_features]])
    # y = np.array([1 if set(idxs) & set(poison_indices) != set() else 0 for idxs in concat_indices])

    # preds = rf.predict_proba(X)[:, 1]
    # print("acc: ", accuracy_score(y, [1 if p > 0.5 else 0 for p in preds]))
    # print("tpr: ", recall_score(y, [1 if p > 0.5 else 0 for p in preds]))
    
    # predictions_proba_2 = []
    # index_tracker_2 = []
    # predictions_with_indices_2 = []
    # true_labels_2 = []
    
    predictions_with_indices_2, _, true_labels_2, predictions_proba_2, _, index_tracker_2 = train_prov_data_custom(
            sus_diff[:,relevant_features], clean_diff[:,relevant_features], clean_inds_idv, sus_inds_idv, poison_indices,random_clean_sus_idx, 5, seed, device, model_name=cv_model) 
    
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    # X = np.concatenate([clean_diff[:,relevant_features], sus_diff[:,relevant_features]])
    # y = np.concatenate([np.zeros(len(clean_diff)), np.ones(len(sus_diff))])
    # rf.fit(X, y)
    # print("acc: ",rf.score(X, y))

            
    with open(prov_path + f'relevant_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}.pkl', 'wb') as f:
        pickle.dump(relevant_features, f)

    
    # mlp = MLP(input_size=len(relevant_features), num_classes=2)
    # mlp.load_state_dict(torch.load(prov_path + f"mlp_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{128}.pt"))
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(mlp.parameters(), lr=0.01)
    # mlp.to(device)
    # X = np.concatenate([clean_diff[:,relevant_features], sus_diff[:,relevant_features]])
    # y = np.array([1 if set(idxs) & set(poison_indices) != set() else 0 for idxs in concat_indices])
    # train_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    # train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    # # mlp.train(mode = training_mode)
    
    # # for epoch in range(10):
    # #     for inputs, labels in train_loader:
    # #         inputs, labels = inputs.to(device), labels.to(device)
    # #         optimizer.zero_grad()
    # #         outputs = mlp(inputs)
    # #         loss = criterion(outputs, labels)
    # #         loss.backward()
    # #         optimizer.step()
            
    # #predict 
    # test_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
    # mlp.eval()
    # all_preds = []
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = mlp(inputs)
    #         _, preds = torch.max(outputs, 1)
    #         probs = torch.softmax(outputs, dim=1)[:, 1]
    #         all_preds.extend(preds.cpu().numpy())
    # y_pred = np.array(all_preds)
    # print("acc: ", accuracy_score(y, y_pred)) 
    # print("tpr: ", recall_score(y, y_pred))
    # print("f1: ", f1_score(y, y_pred))
    # print("precision: ", precision_score(y, y_pred))
    
    # predictions_proba_2 = []
    # index_tracker_2 = []
    # predictions_with_indices_2 = []
    # true_labels_2 = []
    
    # torch.save(mlp.state_dict(), prov_path + f"mlp_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}.pt")
    # plt.figure()
    # plt.scatter(range(len(predictions_proba_2[true_labels_2==0])), predictions_proba_2[true_labels_2==0], alpha=0.5, color='blue')
    
    # is_sus_pos = np.array([set(idxs) & set(poison_indices) != set() for idxs in concat_indices])[index_tracker_2]
    # # num_poisons = np.array([len(set(idxs) & set(poison_indices)) for idxs in concat_indices])[index_tracker_2][true_labels_2==1]
    # # sizes = num_poisons * 20 
    # # cmap = plt.get_cmap('plasma')
    # # normalize = plt.Normalize(vmin=min(num_poisons), vmax=max(num_poisons))
    # # colors = cmap(normalize(num_poisons))

    # print("acc: ", accuracy_score(true_labels_2, [1 if p > 0.5 else 0 for p in predictions_proba_2]))
    # pred_proba_sus_clean = predictions_proba_2[~is_sus_pos][true_labels_2[~is_sus_pos] == 1]
    # pred_proba_sus_pos = predictions_proba_2[is_sus_pos][true_labels_2[is_sus_pos] == 1]
    # plt.scatter(range(len(pred_proba_sus_clean)), pred_proba_sus_clean, alpha=0.5, color='green')
    # plt.scatter(range(len(pred_proba_sus_pos)), pred_proba_sus_pos, alpha=0.5, color='red')
    # # scatter = plt.scatter(range(len(predictions_proba_2)), predictions_proba_2, alpha=0.5, s=sizes, c='blue', label='Predictions Probabilities')
    # # # Add color bar
    # # cbar = plt.colorbar(scatter)
    # # cbar.set_label('Number of Poisons')
    
    # plt.title("Predictions probabilities for each class")
    # plt.legend()
    # plt.savefig(figure_path + f"Predictions_proba.png")
    return relevant_features, rf, predictions_with_indices_2, true_labels_2, predictions_proba_2, index_tracker_2
    # return relevant_features, rf, predictions_with_indices, true_labels, predictions_proba, index_tracker



def analyze_batch_sample_weights(clean_inds, sus_inds, poison_indices,predictions_with_indices, predictions_proba, index_tracker, dataset, seed, attack, eps, percentage, poison_ratio, threshold, penalty, figure_path = "./figures/"):
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    predictions_with_indices = {}
    concat_inds = clean_inds + sus_inds
    concat_inds_real = [concat_inds[i] for i in index_tracker]
    print("len sus inds: ", len(sus_inds), "len clean inds: ", len(clean_inds))
    
    unique_sus_inds = np.unique(np.concatenate(sus_inds))
    

    
    avg_sus_scores_per_epoch = {}
    avg_clean_scores_per_epoch = {}
    
    avg_batch_value_sus = len(sus_inds)//167 + 1
    avg_batch_value_clean = len(clean_inds)//167 + 1
    for i in range(len(concat_inds_real)):
        for idx in concat_inds_real[i]:
            if idx in unique_sus_inds:
                epoch_num = (index_tracker[i]-len(clean_inds))//avg_batch_value_sus
                avg_sus_scores_per_epoch[epoch_num] = avg_sus_scores_per_epoch.get(epoch_num, []) + [predictions_proba[i]]
            else:
                epoch_num = (index_tracker[i])//avg_batch_value_clean
                avg_clean_scores_per_epoch[epoch_num] = avg_clean_scores_per_epoch.get(epoch_num, []) + [predictions_proba[i]]
                    
           
    if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
        poison_indices_all = poison_indices
        poison_indices = np.concatenate(list(poison_indices_all.values()))

    print(len(list(avg_sus_scores_per_epoch.keys())), len(list(avg_clean_scores_per_epoch.keys())))
    
    for epoch in avg_sus_scores_per_epoch:
        avg_sus_scores_per_epoch[epoch] = np.mean(avg_sus_scores_per_epoch[epoch])
    for epoch in avg_clean_scores_per_epoch:
        avg_clean_scores_per_epoch[epoch] = np.mean(avg_clean_scores_per_epoch[epoch])
        
    min_len = min(len(avg_sus_scores_per_epoch), len(avg_clean_scores_per_epoch))
    epoch_weights = np.array([
        avg_sus_scores_per_epoch[i] - avg_clean_scores_per_epoch[i]
        for i in range(min_len)
    ])

    from scipy.ndimage import gaussian_filter1d

    def gaussian_smoothing(data, sigma):
        return gaussian_filter1d(data, sigma=sigma)
   
    norm_epoch_weights = normalize(epoch_weights.reshape(-1, 1), axis=0).reshape(-1)
    # norm_epoch_weights =  gaussian_smoothing(norm_epoch_weights, sigma=1)
    
    
    for i in range(len(concat_inds_real)):
        for idx in concat_inds_real[i]:
            if idx in unique_sus_inds:
                epoch_num = (index_tracker[i]-len(clean_inds))//avg_batch_value_sus
                if idx in predictions_with_indices:
                    predictions_with_indices[idx].append(predictions_proba[i] * norm_epoch_weights[epoch_num])
                else:
                    predictions_with_indices[idx] = [predictions_proba[i] * norm_epoch_weights[epoch_num]]
            else:
                epoch_num = (index_tracker[i])//avg_batch_value_clean
                if idx in predictions_with_indices:
                    predictions_with_indices[idx].append(predictions_proba[i] * norm_epoch_weights[epoch_num])
                else:
                    predictions_with_indices[idx] = [predictions_proba[i] * norm_epoch_weights[epoch_num]]
    print("len epoch weights: ", len(epoch_weights))
                
    plt.figure()
    plt.plot(range(len(avg_sus_scores_per_epoch)), [avg_sus_scores_per_epoch[i] for i in range(len(avg_sus_scores_per_epoch))], label='Poison Scores', alpha=0.5, color='red')
    plt.plot(range(len(avg_clean_scores_per_epoch)), [avg_clean_scores_per_epoch[i] for i in range(len(avg_clean_scores_per_epoch))], label='Clean Scores', alpha=0.5, color='blue')
    plt.legend()
    plt.title(f"Average Scores per Epoch for {attack} percentage {percentage} pr {poison_ratio}")
    plt.savefig(figure_path + f"per_epoch_scores_{attack}_{dataset}_{eps}_{percentage}_{poison_ratio}_{threshold}_{penalty}.png")
    
                    
                    
    clean_inds_flat = np.concatenate(clean_inds)
    real_clean_indices = np.unique(clean_inds_flat) 
    
    
    
    # Initialize arrays
    pos_predictions_real = []
    clean_predictions_real = []
    sus_clean_predictions = []

    pos_prediction_indices = []
    clean_predicion_indices = []
    sus_clean_prediction_indices = []

    epoch_num = 2000   # set to a large number 
    # Fill arrays
    for k, v in predictions_with_indices.items():
        if k in poison_indices:
            v = np.array(v)
            if len(v) < epoch_num: 
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            pos_predictions_real.append(v)
            pos_prediction_indices.append(k)
        elif k in real_clean_indices:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            clean_predictions_real.append(v)
            clean_predicion_indices.append(k)
        else:
            v = np.array(v) 
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            sus_clean_predictions.append(v)
            sus_clean_prediction_indices.append(k)
            

    # Convert lists to arrays
    pos_predictions_real = np.array(pos_predictions_real)
    clean_predictions_real = np.array(clean_predictions_real)
    sus_clean_predictions = np.array(sus_clean_predictions)

    pos_prediction_indices = np.array(pos_prediction_indices)
    clean_prediction_indices = np.array(clean_predicion_indices)
    sus_clean_prediction_indices = np.array(sus_clean_prediction_indices)

 

    def compute_thresholds(pos_scores, clean_scores, sus_scores):
        if sus_scores is not None:
            combined_data = np.concatenate([pos_scores, sus_scores]).reshape(-1, 1)
        else:
            combined_data = np.concatenate([pos_scores, clean_scores]).reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(combined_data)
        
        

        # Compute the GMM Threshold
        x = np.linspace(combined_data.min(), combined_data.max(), num=1000).reshape(-1, 1)
        pdf_individual = gmm.predict_proba(x) * np.exp(gmm.score_samples(x).reshape(-1, 1))
        diff_sign = np.diff(np.sign(pdf_individual[:, 0] - pdf_individual[:, 1]))

        gaussian_threshold = next(x[i, 0] for i in range(1, len(x)) if np.diff(np.sign(pdf_individual[:, 0] - pdf_individual[:, 1]))[i] != 0)
        # Compute the KMeans Threshold
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(combined_data)
        labels = kmeans.labels_
        cluster_0_points = combined_data[labels == 0]
        cluster_1_points = combined_data[labels == 1]
        boundary_points = [min(cluster_0_points), max(cluster_1_points)]
        kmeans_threshold = np.mean(boundary_points)

        # Compute the Outlier Threshold
        mean_score = np.mean(combined_data)
        std_score = np.std(combined_data)
        outlier_threshold = mean_score + 2 * std_score 
        
        print("TPR custom threshold: ", np.mean(pos_scores > threshold))
        print("TPR gaussian threshold: ", np.mean(pos_scores > gaussian_threshold))
        print("TPR kmeans: ", np.mean(pos_scores > kmeans_threshold))
        
        if sus_scores is not None:
            print("FPR custom threshold: ", np.mean(sus_scores > threshold))
            print("FPR gaussian: ", np.mean(sus_scores > gaussian_threshold))
            print("FPR kmeans: ", np.mean(sus_scores > kmeans_threshold))
            print("Real TPR custom threshold: ", np.mean(np.concatenate([pos_scores , sus_scores]) > threshold))
            print("Real TPR gaussian: ", np.mean(np.concatenate([pos_scores , sus_scores]) > gaussian_threshold))
            print("Real TPR kmeans: ", np.mean(np.concatenate([pos_scores , sus_scores]) > kmeans_threshold))
            pos_custom = np.mean(pos_scores > threshold)
            fpos_custom = np.mean(sus_scores > threshold)
            
            pos_gaussian = np.mean(np.concatenate([pos_scores , sus_scores]) > gaussian_threshold)
            fpos_gaussian = np.mean(clean_scores > gaussian_threshold)
            
            pos_kmeans = np.mean(np.concatenate([pos_scores , sus_scores]) > kmeans_threshold)
            fpos_kmeans = np.mean(clean_scores > kmeans_threshold)
        else:
            print("FPR custom threshold: ", np.mean(clean_scores > threshold))
            print("FPR gaussian: ", np.mean(clean_scores > gaussian_threshold))
            print("FPR kmeans: ", np.mean(clean_scores > kmeans_threshold))
            
            pos_custom = np.mean(pos_scores > threshold)
            fpos_custom = np.mean(clean_scores > threshold)
            
            pos_gaussian = np.mean(pos_scores > gaussian_threshold)
            fpos_gaussian = np.mean(clean_scores > gaussian_threshold)
            
            pos_kmeans = np.mean(pos_scores > kmeans_threshold)
            fpos_kmeans = np.mean(clean_scores > kmeans_threshold)
            
            

        return threshold, kmeans_threshold, gaussian_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans, pos_custom, fpos_custom

    def plot_scores(pos_scores, clean_scores, sus_scores, title):
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'orange', 'red']
        plt.rcParams.update({'font.size': 18})
        if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
            start_index = 0
            for key in poison_indices_all:
                attack_indices = [i for i, ind in enumerate(pos_prediction_indices) if ind in poison_indices_all[key]]
                plt.scatter(np.arange(len(pos_scores[attack_indices])), pos_scores[attack_indices], label=f'Poison {key}', alpha=0.5, color=colors[start_index])
                start_index += 1
        else:
            plt.scatter(np.arange(len(pos_scores)), pos_scores, label='Poison', alpha=0.5, color='red')
        # plt.scatter(np.arange(len(clean_scores)), clean_scores, label='Clean', alpha=0.5, color='blue')
        if sus_scores is not None:
            plt.scatter(np.arange(len(sus_scores)), sus_scores, label='Clean Suspected', alpha=0.5, color='green')

        threshold, kmeans_threshold, gaussian_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans, pos_custom, fpos_custom = compute_thresholds(pos_scores, clean_scores, sus_scores)
        
        thresholds = [threshold, kmeans_threshold, gaussian_threshold]
        print("thresholds: ", thresholds)
        labels = ['Threshold', 'Threshold 1 (KMeans)', 'Threshold 2 (Gaussian)']
        colors = ['black', 'green', 'orange']

        for thr, color, label in zip(thresholds[1:], colors[1:], labels[1:]):
            plt.axhline(y=thr, color=color, linestyle='--', label=label)
        
        # plt.title(title)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=12)
        plt.xlabel('Sample Index', fontweight = 'bold')
        plt.ylabel('Poisoning Score', fontweight = 'bold')
        # plt.axhline(y=threshold, color='black', linestyle='--')
        plt.savefig(figure_path + f"{title}.png")
        
        return threshold, kmeans_threshold, gaussian_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans, pos_custom, fpos_custom


    def average_k_minimum_values(arr, k):
        # Mask NaN values with a large number
        nan_mask = np.isnan(arr)
        large_number = np.nanmax(arr[np.isfinite(arr)]) + 1
        arr_masked = np.where(nan_mask, large_number, arr)
        
        # Get the indices of the k smallest values along axis 1
        k_min_indices = np.argsort(arr_masked, axis=1)[:, :k]
        
        # Extract the k smallest values using the indices
        k_min_values = np.take_along_axis(arr, k_min_indices, axis=1)
        
        # Calculate the average of the k minimum values for each row
        k_min_averages = np.mean(k_min_values, axis=1)
        
        return k_min_averages

    j = 50
    # Compute different types of scores
    pos_mean = np.nanmean(pos_predictions_real, axis=1)
    pos_max = average_k_minimum_values(pos_predictions_real, j)
    pos_mean_max = pos_mean * pos_max

    clean_mean = np.nanmean(clean_predictions_real, axis=1)
    clean_max = average_k_minimum_values(clean_predictions_real, j)
    clean_mean_max = clean_mean * clean_max

    if len(sus_clean_predictions) > 0:
        sus_mean = np.nanmean(sus_clean_predictions, axis=1)
        sus_max = average_k_minimum_values(sus_clean_predictions, j)
        sus_mean_max = sus_mean * sus_max
    else:
        sus_mean = sus_max = sus_mean_max = None

    # Plot each type of score
    print("Mean")
    custom_threshold_mean,kmeans_threshold_mean, gaussian_threhold_mean, pos_gaussian_mean, fpos_gaussian_mean, pos_kmeans_mean, fpos_kmeans_mean, pos_custom_mean, fpos_custom_mean = plot_scores(pos_mean, clean_mean, sus_mean, f"BSL Mean {attack} pr {poison_ratio} percentage {percentage}")
    # print("Min")
    # gaussian_threshold_max,kmeans_threshold_max,pos_gaussian_max, fpos_gaussian_max, pos_kmeans_max, fpos_kmeans_max = plot_scores(pos_max, clean_max, sus_max, f"rf Min {attack} pr {poison_ratio} percentage {percentage}")
    best_config = "mean_kmeans"
    if len(sus_clean_prediction_indices) > 0:
        values = {
        "mean_gaussian": np.concatenate([pos_mean, sus_mean]) > gaussian_threhold_mean,
        "mean_kmeans": np.concatenate([pos_mean, sus_mean]) > kmeans_threshold_mean,
        # "max_gaussian": np.concatenate([pos_max, sus_max]) > gaussian_threshold_max,
        # "max_kmeans": np.concatenate([pos_max, sus_max]) > kmeans_threshold_max,
        # "mean_max_gaussian":np.concatenate([pos_mean_max, sus_mean_max]) > gaussian_threshold_mean_max,
        # "mean_max_kmeans": np.concatenate([pos_mean_max, sus_mean_max]) > kmeans_threshold_mean_max
        }
        
        sus_prediction_indices = np.concatenate([pos_prediction_indices, sus_clean_prediction_indices])
        indexes_to_exculde = sus_prediction_indices[values[best_config]]
        return indexes_to_exculde
        
    else:
        values = {
        "mean_gaussian": pos_mean > gaussian_threshold_mean,
        "mean_kmeans": pos_mean > kmeans_threshold_mean,
        # "max_gaussian": pos_max > gaussian_threshold_max,
        # "max_kmeans": pos_max > kmeans_threshold_max,
        # "mean_max_gaussian":pos_mean_max > gaussian_threshold_mean_max,
        # "mean_max_kmeans": pos_mean_max > kmeans_threshold_mean_max
        }
        
        print("values: ", len(pos_mean[pos_mean > gaussian_threshold_mean]))
        indexes_to_exculde = pos_prediction_indices[values[best_config]]
            
    return indexes_to_exculde



def capture_sample_level_weight_updates_2(random_sus_idx, random_poison_idx, model, orig_model, optimizer, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class, relevant_features,attack, sample_from_test, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    sus_weights = {}
    clean_weights = {}
    clean_weights_2 = {}
    
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    
    if sample_from_test:
        target_images = [img for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl == target_class]
        
   
    
    for epoch in tqdm(range(training_epochs)):
        # Train
        model.to(device)
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        sus_weights[epoch] = {}
        clean_weights[epoch] = {}
        clean_weights_2[epoch] = {}

        batch_num = 0
        

        for images, labels, indices in pbar:    
            images, labels, indices = images.to(device), labels.to(device), indices
            
            batch_num += 1
            
            torch.save(model.state_dict(), f'./temp_folder/temp_weights_{random_num}.pth')
            torch.save(optimizer.state_dict(), f'./temp_folder/temp_optimizer_{random_num}.pth')
            
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
                
            pos_indices = [ind.item() for ind in indices if ind in random_sus_idx]
            available_indices = list(set(indices[labels.cpu().numpy() == target_class].numpy()) - set(pos_indices))
            if len(pos_indices) > 0 and len(available_indices) >= 2 and not sample_from_test:
                if len(available_indices) < len(pos_indices):
                    repeat_times = (len(pos_indices) - 1) // len(available_indices) + 1  # Calculate how many times to repeat available_indices
                    extended_indices = (available_indices * repeat_times)[:len(pos_indices)]  # Extend and then slice to the needed length
                    clean_indices = extended_indices
                else:
                    clean_indices = random.sample(available_indices, len(pos_indices)) 
                ignore_set = set(clean_indices) | set(pos_indices)
                available_indices = list(set(indices[labels.cpu().numpy() == target_class].numpy()) - ignore_set)
                if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                    repeat_times = (len(pos_indices) - 1) // len(available_indices) + 1  # Calculate how many times to repeat available_indices
                    extended_indices = (available_indices * repeat_times)[:len(pos_indices)]  # Extend and then slice to the needed length
                    clean_indices_2 = extended_indices
                elif len(available_indices) == 0:
                    clean_indices_2 = [clean_indices[-1]] * len(pos_indices)
                    clean_indices = np.array(clean_indices)
                    clean_indices[np.where(clean_indices == clean_indices[-1])[0]] = list(set(clean_indices) - {clean_indices[-1]})[0]
                    clean_indices = list(clean_indices)
                else:
                    clean_indices_2 = random.sample(available_indices, len(pos_indices))
            
                remaining_indices = [ind for ind in indices if ind not in pos_indices and ind not in clean_indices and ind not in clean_indices_2]
            
                
                if not (set(remaining_indices) & set(random_poison_idx) == set() and set(clean_indices) & set(random_poison_idx) == set() and set(clean_indices_2) & set(random_poison_idx) == set()):
                    print(set(remaining_indices) & set(random_poison_idx), set(clean_indices) & set(random_poison_idx), set(clean_indices_2) & set(random_poison_idx))

                assert set(remaining_indices) & set(random_poison_idx) == set() and set(clean_indices) & set(random_poison_idx) == set() and set(clean_indices_2) & set(random_poison_idx) == set()
                assert len(clean_indices) == len(clean_indices_2) and len(clean_indices) == len(pos_indices)
                
                index_choices = list(set(pos_indices) | set(clean_indices) | set(clean_indices_2))
                for index in index_choices:
                    original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth') 
                    
                    temp_model = copy.deepcopy(orig_model)
                    temp_model.to(device)
                    temp_model.load_state_dict(original_weights)
                    temp_model.train(mode = training_mode)
                    
                    optimizer_temp = torch.optim.SGD(params=temp_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    optimizer_temp.load_state_dict(original_optimizer)
                    
                    temp_loss = nn.CrossEntropyLoss()
                    optimizer_temp.zero_grad()
                
                    batch = [images[idx] for idx in range(len(images)) if indices[idx] in [index] ]
                    batch = torch.stack(batch)
                    
                    labels_temp = [labels[idx] for idx in range(len(labels)) if indices[idx] in [index] ]
                    labels_temp  = torch.stack(labels_temp)
                    
                    output = temp_model(batch)
                    pred_labels = output.argmax(dim=1)
                    
                    loss = temp_loss(output, labels_temp)
                    loss.backward()
                    optimizer_temp.step()

                    if index in pos_indices:
                        sus_weights[epoch][index] = np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys()])[relevant_features]
                    elif index in clean_indices:
                        for num in np.where(clean_indices == index)[0]:
                            clean_weights[epoch][tuple(torch.tensor([index]))] = np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys()])[relevant_features]
                    else:
                        for num in np.where(clean_indices_2 == index)[0]:
                            clean_weights_2[epoch][tuple(torch.tensor([batch_num]))] = np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys()])[relevant_features]
            elif sample_from_test and len(pos_indices) > 0:
                clean_indices = random.sample(range(len(target_images)), len(pos_indices))
                ignore_set = set(clean_indices)
                clean_indices_2 = random.sample(set(range(len(target_images))) - ignore_set, len(pos_indices)) 
                remaining_indices = [ind for ind in indices if ind not in pos_indices] 
                assert set(remaining_indices) & set(random_poison_idx) == set()
                
                
                ref_clean_values = np.zeros(len(relevant_features))
                for index in clean_indices_2:
                    original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth') 
                    
                    temp_model = copy.deepcopy(orig_model)
                    temp_model.to(device)
                    temp_model.load_state_dict(original_weights)
                    temp_model.train(mode = training_mode)
                    
                    optimizer_temp = torch.optim.SGD(params=temp_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    optimizer_temp.load_state_dict(original_optimizer)
                    
                    temp_loss = nn.CrossEntropyLoss()
                    optimizer_temp.zero_grad()
                
                    batch = [target_images[index].to(device)]
                    batch += [images[idx] for idx in range(len(images)) if indices[idx] in remaining_indices]
                    batch = torch.stack(batch)
                    
                    labels_temp = [torch.tensor(target_class).to(device)]
                    labels_temp += [labels[idx] for idx in range(len(labels)) if indices[idx] in remaining_indices]
                    labels_temp  = torch.stack(labels_temp)
                    
                    output = temp_model(batch)
                    pred_labels = output.argmax(dim=1)
                    
                    loss = temp_loss(output, labels_temp)
                    loss.backward()
                    optimizer_temp.step()
                    
                    if attack != 'ht':
                        ref_clean_values += np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys()])[relevant_features]
                    else:
                        ref_clean_values +np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys() if "20" in name])[relevant_features]
                avg_ref_clean_values = ref_clean_values / len(clean_indices_2)

                
                for index in pos_indices:
                    original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth') 
                    
                    temp_model = copy.deepcopy(orig_model)
                    temp_model.to(device)
                    temp_model.load_state_dict(original_weights)
                    temp_model.train(mode = training_mode)
                    
                    optimizer_temp = torch.optim.SGD(params=temp_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    optimizer_temp.load_state_dict(original_optimizer)
                    
                    temp_loss = nn.CrossEntropyLoss()
                    optimizer_temp.zero_grad()
                
                    batch = [images[idx] for idx in range(len(images)) if indices[idx] in [index] + remaining_indices]
                    batch = torch.stack(batch)
                    
                    labels_temp = [labels[idx] for idx in range(len(labels)) if indices[idx] in [index] + remaining_indices]
                    labels_temp  = torch.stack(labels_temp)
                    
                    output = temp_model(batch)
                    pred_labels = output.argmax(dim=1)
                    
                    loss = temp_loss(output, labels_temp)
                    loss.backward()
                    optimizer_temp.step()

                    if attack != 'ht':
                        sus_weights[epoch][index] = np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys()])[relevant_features] - avg_ref_clean_values
                    else:
                        sus_weights[epoch][index] = np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys() if "20" in name])[relevant_features] - avg_ref_clean_values
                
                for index in clean_indices:
                    original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth') 
                    
                    temp_model = copy.deepcopy(orig_model)
                    temp_model.to(device)
                    temp_model.load_state_dict(original_weights)
                    temp_model.train(mode = training_mode)
                    
                    optimizer_temp = torch.optim.SGD(params=temp_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    optimizer_temp.load_state_dict(original_optimizer)
                    
                    temp_loss = nn.CrossEntropyLoss()
                    optimizer_temp.zero_grad()
                
                    batch = [target_images[index].to(device)]
                    batch += [images[idx] for idx in range(len(images)) if indices[idx] in remaining_indices]
                    batch = torch.stack(batch)
                    
                    labels_temp = [torch.tensor(target_class).to(device)]
                    labels_temp += [labels[idx] for idx in range(len(labels)) if indices[idx] in remaining_indices]
                    labels_temp  = torch.stack(labels_temp)
                    
                    output = temp_model(batch)
                    pred_labels = output.argmax(dim=1)
                    
                    loss = temp_loss(output, labels_temp)
                    loss.backward()
                    optimizer_temp.step()
                    
                    if attack != 'ht':
                        clean_weights[epoch][tuple(torch.tensor([index]))] = np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys()])[relevant_features] - avg_ref_clean_values
                    else:
                        clean_weights[epoch][tuple(torch.tensor([index]))] = np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys() if "20" in name])[relevant_features] - avg_ref_clean_values
                            
                              
        torch.cuda.empty_cache()       
                    
                
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

        
        # Testing attack effect
        model.eval()
        correct, total = 0, 0
        for i, (images, labels) in enumerate(poisoned_test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        test_ACC.append(acc)
        print('\nAttack success rate %.2f' % (acc*100))
        print('Test_loss:',out_loss)
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:',out_loss)
        
    
    os.remove(f'./temp_folder/temp_weights_{random_num}.pth')
    os.remove(f'./temp_folder/temp_optimizer_{random_num}.pth')
    
    
    sus_indices = np.array([
        image_idx
        for epoch in sus_weights
        for image_idx in sus_weights[epoch]
    ])
    sus_array = np.array([
        sus_weights[epoch][image_idx]
        for epoch in sus_weights
        for image_idx in sus_weights[epoch]
    ])

    clean_indices = np.array([
        image_idx[0]
        for epoch in clean_weights
        for image_idx
        in clean_weights[epoch]
    ])
    clean_array = np.array([
        clean_weights[epoch][image_idx]
        for epoch in clean_weights
        for image_idx in clean_weights[epoch]
    ])

        
    return sus_array, clean_array, sus_indices, clean_indices

def capture_sample_level_weight_updates_3(random_sus_idx, random_poison_idx, model, orig_model, optimizer, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class, relevant_features,sample_from_test, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    sus_weights = {}
    clean_weights = {}
    clean_weights_2 = {}
    
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    
    if sample_from_test:
        target_images = [img for imgs, lbls in test_loader for img, lbl in zip(imgs, lbls) if lbl == target_class]
        

    # sur_model = copy.deepcopy(model)
    # sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    for epoch in tqdm(range(training_epochs)):
        # Train
        model.to(device)
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        sus_weights[epoch] = {}
        clean_weights[epoch] = {}
        clean_weights_2[epoch] = {}

        batch_num = 0
        

        for images, labels, indices in pbar:    
            images, labels, indices = images.to(device), labels.to(device), indices
            
            batch_num += 1
            
            # torch.save(model.state_dict(), f'./temp_folder/temp_weights_{random_num}.pth')
            # torch.save(optimizer.state_dict(), f'./temp_folder/temp_optimizer_{random_num}.pth')
            original_weights = copy.deepcopy(model.state_dict())
            original_optimizer = copy.deepcopy(optimizer.state_dict())
            
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
                
            pos_indices = [ind.item() for ind in indices if ind in random_sus_idx]
            available_indices = list(set(indices[labels.cpu().numpy() == target_class].numpy()) - set(pos_indices))
            if len(pos_indices) > 0 and len(available_indices) >= 2 and not sample_from_test:
                if len(available_indices) < len(pos_indices):
                    repeat_times = (len(pos_indices) - 1) // len(available_indices) + 1  # Calculate how many times to repeat available_indices
                    extended_indices = (available_indices * repeat_times)[:len(pos_indices)]  # Extend and then slice to the needed length
                    clean_indices = extended_indices
                else:
                    clean_indices = random.sample(available_indices, len(pos_indices)) 
                ignore_set = set(clean_indices) | set(pos_indices)
                available_indices = list(set(indices[labels.cpu().numpy() == target_class].numpy()) - ignore_set)
                if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                    repeat_times = (len(pos_indices) - 1) // len(available_indices) + 1  # Calculate how many times to repeat available_indices
                    extended_indices = (available_indices * repeat_times)[:len(pos_indices)]  # Extend and then slice to the needed length
                    clean_indices_2 = extended_indices
                elif len(available_indices) == 0:
                    clean_indices_2 = [clean_indices[-1]] * len(pos_indices)
                    clean_indices = np.array(clean_indices)
                    clean_indices[np.where(clean_indices == clean_indices[-1])[0]] = list(set(clean_indices) - {clean_indices[-1]})[0]
                    clean_indices = list(clean_indices)
                else:
                    clean_indices_2 = random.sample(available_indices, len(pos_indices))
            
                remaining_indices = [ind for ind in indices if ind not in pos_indices and ind not in clean_indices and ind not in clean_indices_2]
            
                
                if not (set(remaining_indices) & set(random_poison_idx) == set() and set(clean_indices) & set(random_poison_idx) == set() and set(clean_indices_2) & set(random_poison_idx) == set()):
                    print(set(remaining_indices) & set(random_poison_idx), set(clean_indices) & set(random_poison_idx), set(clean_indices_2) & set(random_poison_idx))

                assert set(remaining_indices) & set(random_poison_idx) == set() and set(clean_indices) & set(random_poison_idx) == set() and set(clean_indices_2) & set(random_poison_idx) == set()
                assert len(clean_indices) == len(clean_indices_2) and len(clean_indices) == len(pos_indices)
                
                index_choices = list(set(pos_indices) | set(clean_indices) | set(clean_indices_2))
                for index in index_choices:
                    # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    # original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth') 
                    
                    temp_model = copy.deepcopy(orig_model)
                    temp_model.to(device)
                    temp_model.load_state_dict(original_weights)
                    temp_model.train(mode = training_mode)
                    
                    optimizer_temp = torch.optim.SGD(params=temp_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    optimizer_temp.load_state_dict(original_optimizer)
                    
                    temp_loss = nn.CrossEntropyLoss()
                    optimizer_temp.zero_grad()
                
                    batch = [images[idx] for idx in range(len(images)) if indices[idx] in [index] ]
                    batch = torch.stack(batch)
                    
                    labels_temp = [labels[idx] for idx in range(len(labels)) if indices[idx] in [index] ]
                    labels_temp  = torch.stack(labels_temp)
                    
                    output = temp_model(batch)
                    pred_labels = output.argmax(dim=1)
                    
                    loss = temp_loss(output, labels_temp)
                    loss.backward()
                    optimizer_temp.step()

                    if index in pos_indices:
                        sus_weights[epoch][index] = np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys()])[relevant_features]
                    elif index in clean_indices:
                        for num in np.where(clean_indices == index)[0]:
                            clean_weights[epoch][tuple(torch.tensor([index]))] = np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys()])[relevant_features]
                    else:
                        for num in np.where(clean_indices_2 == index)[0]:
                            clean_weights_2[epoch][tuple(torch.tensor([batch_num]))] = np.concatenate([(temp_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in temp_model.state_dict().keys()])[relevant_features]
            elif sample_from_test and len(pos_indices) > 0:
                clean_indices = random.sample(range(len(target_images)), len(pos_indices))
                ignore_set = set(clean_indices)
                clean_indices_2 = random.sample(set(range(len(target_images))) - ignore_set, len(pos_indices))
                remaining_indices = [ind for ind in indices if ind not in pos_indices]
                assert set(remaining_indices) & set(random_poison_idx) == set()
                
                
                ref_clean_values = np.zeros(len(relevant_features))
                for index in clean_indices_2:
                    # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    # original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth') 
                    
                    sur_model = copy.deepcopy(orig_model).to(device)
                    sur_model.load_state_dict(original_weights)
                    sur_model.train(mode = training_mode)
                    
                    sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    sur_optimizer.load_state_dict(original_optimizer)
                    
                    temp_loss = nn.CrossEntropyLoss()
                    sur_optimizer.zero_grad()
                
                    batch = [target_images[index].to(device)]
                    batch += [images[idx] for idx in range(len(images)) if indices[idx] in remaining_indices]
                    batch = torch.stack(batch)
                    
                    labels_temp = [torch.tensor(target_class).to(device)]
                    labels_temp += [labels[idx] for idx in range(len(labels)) if indices[idx] in remaining_indices]
                    labels_temp  = torch.stack(labels_temp)
                    
                    output = sur_model(batch)
                    pred_labels = output.argmax(dim=1)
                    
                    loss = temp_loss(output, labels_temp)
                    loss.backward()
                    sur_optimizer.step()
                    
                    ref_clean_values += np.concatenate([(sur_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in sur_model.state_dict().keys()])[relevant_features]
                avg_ref_clean_values = ref_clean_values / len(clean_indices_2)

                
                for index in pos_indices:
                    # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    # original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth') 
                    
                    sur_model = copy.deepcopy(orig_model).to(device)
                    sur_model.load_state_dict(original_weights)
                    sur_model.train(mode = training_mode)
                    
                    sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    sur_optimizer.load_state_dict(original_optimizer)
                    
                    temp_loss = nn.CrossEntropyLoss()
                    sur_optimizer.zero_grad()
                
                    batch = [images[idx] for idx in range(len(images)) if indices[idx] in [index] + remaining_indices]
                    batch = torch.stack(batch)
                    
                    labels_temp = [labels[idx] for idx in range(len(labels)) if indices[idx] in [index] + remaining_indices]
                    labels_temp  = torch.stack(labels_temp)
                    
                    output = sur_model(batch)
                    pred_labels = output.argmax(dim=1)
                    
                    loss = temp_loss(output, labels_temp)
                    loss.backward()
                    sur_optimizer.step()

                    sus_weights[epoch][index] = np.concatenate([(sur_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in sur_model.state_dict().keys()])[relevant_features] - avg_ref_clean_values
                
                for index in clean_indices:
                    # original_weights = torch.load(f'./temp_folder/temp_weights_{random_num}.pth')
                    # original_optimizer = torch.load(f'./temp_folder/temp_optimizer_{random_num}.pth') 
                    sur_model = copy.deepcopy(orig_model).to(device)
                    sur_model.load_state_dict(original_weights)
                    sur_model.train(mode = training_mode)
                    
                    sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    sur_optimizer.load_state_dict(original_optimizer)
                    
                    temp_loss = nn.CrossEntropyLoss()
                    sur_optimizer.zero_grad()
                
                    batch = [target_images[index].to(device)]
                    batch += [images[idx] for idx in range(len(images)) if indices[idx] in remaining_indices]
                    batch = torch.stack(batch)
                    
                    labels_temp = [torch.tensor(target_class).to(device)]
                    labels_temp += [labels[idx] for idx in range(len(labels)) if indices[idx] in remaining_indices]
                    labels_temp  = torch.stack(labels_temp)
                    
                    output = sur_model(batch)
                    pred_labels = output.argmax(dim=1)
                    
                    loss = temp_loss(output, labels_temp)
                    loss.backward()
                    sur_optimizer.step()
                    
                    clean_weights[epoch][tuple(torch.tensor([index]))] = np.concatenate([(sur_model.state_dict()[name] - original_weights[name]).cpu().flatten() for name in sur_model.state_dict().keys()])[relevant_features] - avg_ref_clean_values
        
                              
        torch.cuda.empty_cache()       
                    
                
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

        
        # Testing attack effect
        model.eval()
        correct, total = 0, 0
        for i, (images, labels) in enumerate(poisoned_test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        test_ACC.append(acc)
        print('\nAttack success rate %.2f' % (acc*100))
        print('Test_loss:',out_loss)
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:',out_loss)
        
    
    # os.remove(f'./temp_folder/temp_weights_{random_num}.pth')
    # os.remove(f'./temp_folder/temp_optimizer_{random_num}.pth')
    
    
    sus_indices = np.array([
        image_idx
        for epoch in sus_weights
        for image_idx in sus_weights[epoch]
    ])
    sus_array = np.array([
        sus_weights[epoch][image_idx]
        for epoch in sus_weights
        for image_idx in sus_weights[epoch]
    ])

    clean_indices = np.array([
        image_idx[0]
        for epoch in clean_weights
        for image_idx
        in clean_weights[epoch]
    ])
    clean_array = np.array([
        clean_weights[epoch][image_idx]
        for epoch in clean_weights
        for image_idx in clean_weights[epoch]
    ])

        
    return sus_array, clean_array, sus_indices, clean_indices



def get_iso_forest_threshold(sus_diff, clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, n_groups, epochs, seed, poison_ratio, percentage, attack, figure_path = "./figures/", threshold=0.5):
    is_poison = np.array([{idxs} & set(poison_indices) != set() for idxs in sus_inds])
    iso = IsolationForest(contamination = 0.1, random_state=seed)
    iso.fit(clean_diff) 
    preds = iso.decision_function(np.concatenate([clean_diff, sus_diff]))
    plt.scatter(range(len(preds[:len(clean_diff)])), preds[:len(clean_diff)], alpha=0.5, color='blue')
    plt.scatter(range(len(preds[len(clean_diff):][is_poison])), preds[len(clean_diff):][is_poison], alpha=0.5, color='red')
    plt.scatter(range(len(preds[len(clean_diff):][~is_poison])), preds[len(clean_diff):][~is_poison], alpha=0.5, color='green')
    plt.title(f"Isolation Forest Predictions for {attack} attack with {poison_ratio} poison ratio and {percentage} percentage")
    plt.axhline(y=-0.2, color='black', linestyle='--')
    plt.savefig(figure_path + f"Isolation_forest_{attack}_{dataset}_{eps}_{percentage}_{poison_ratio}.png") 
    
    predictions_with_indices = {}
    all_inds = np.concatenate([clean_inds, sus_inds])
    for i, ind in enumerate(all_inds):
        if ind in predictions_with_indices:
            predictions_with_indices[ind].append(preds[i])
        else:
            predictions_with_indices[ind] = [preds[i]]
    
    real_clean_indices = np.unique(clean_inds)
    
    # Initialize arrays
    pos_predictions_real = []
    clean_predictions_real = []
    sus_clean_predictions = []

    pos_prediction_indices = []
    clean_predicion_indices = []
    sus_clean_prediction_indices = []

    epoch_num = 1000   # set to a large number 
    # Fill arrays
    for k, v in predictions_with_indices.items():
        if k in poison_indices:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            pos_predictions_real.append(v)
            pos_prediction_indices.append(k)
        elif k in real_clean_indices:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            clean_predictions_real.append(v)
            clean_predicion_indices.append(k)
        else:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            sus_clean_predictions.append(v)
            sus_clean_prediction_indices.append(k)
            

    # Convert lists to arrays
    pos_predictions_real = np.array(pos_predictions_real)
    clean_predictions_real = np.array(clean_predictions_real)
    sus_clean_predictions = np.array(sus_clean_predictions)

    pos_prediction_indices = np.array(pos_prediction_indices)
    clean_prediction_indices = np.array(clean_predicion_indices)
    sus_clean_prediction_indices = np.array(sus_clean_prediction_indices)

 

    def compute_thresholds(pos_scores, clean_scores, sus_scores):
        if sus_scores is not None:
            combined_data = np.concatenate([pos_scores, sus_scores]).reshape(-1, 1)
        else:
            combined_data = np.concatenate([pos_scores, clean_scores]).reshape(-1, 1)

        # gmm = GaussianMixture(n_components=2, random_state=42)
        # gmm.fit(combined_data)
        
        

        # # Compute the GMM Threshold
        # x = np.linspace(combined_data.min(), combined_data.max(), num=1000).reshape(-1, 1)
        # pdf_individual = gmm.predict_proba(x) * np.exp(gmm.score_samples(x).reshape(-1, 1))
        # diff_sign = np.diff(np.sign(pdf_individual[:, 0] - pdf_individual[:, 1]))

        # gaussian_threshold = next(x[i, 0] for i in range(1, len(x)) if np.diff(np.sign(pdf_individual[:, 0] - pdf_individual[:, 1]))[i] != 0)
        # Compute the KMeans Threshold
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(combined_data)
        labels = kmeans.labels_
        cluster_0_points = combined_data[labels == 0]
        cluster_1_points = combined_data[labels == 1]
        boundary_points = [min(cluster_0_points), max(cluster_1_points)]
        kmeans_threshold = np.mean(boundary_points)

        # Compute the Outlier Threshold
        mean_score = np.mean(combined_data)
        std_score = np.std(combined_data)
        outlier_threshold = mean_score + 2 * std_score 
        
        print("TPR custom threshold: ", np.mean(pos_scores < threshold))
        print("TPR kmeans: ", np.mean(pos_scores < kmeans_threshold))
        
        if sus_scores is not None:
            print("FPR gaussian: ", np.mean(sus_scores < threshold))
            print("FPR kmeans: ", np.mean(sus_scores < kmeans_threshold))
            print("Real TPR gaussian: ", np.mean(np.concatenate([pos_scores , sus_scores]) < threshold))
            print("Real TPR kmeans: ", np.mean(np.concatenate([pos_scores , sus_scores]) < kmeans_threshold))
            pos_gaussian = np.mean(np.concatenate([pos_scores , sus_scores]) < threshold)
            fpos_gaussian = np.mean(clean_scores > threshold)
            
            pos_kmeans = np.mean(np.concatenate([pos_scores , sus_scores]) < kmeans_threshold)
            fpos_kmeans = np.mean(clean_scores > kmeans_threshold)
        else:
            print("FPR gaussian: ", np.mean(clean_scores < threshold))
            print("FPR kmeans: ", np.mean(clean_scores < kmeans_threshold))
            pos_gaussian = np.mean(pos_scores < threshold)
            fpos_gaussian = np.mean(clean_scores < threshold)
            
            pos_kmeans = np.mean(pos_scores < kmeans_threshold)
            fpos_kmeans = np.mean(clean_scores < kmeans_threshold)
            
            

        return threshold, kmeans_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans 

    def plot_scores(pos_scores, clean_scores, sus_scores, title):
        plt.figure(figsize=(10, 6))
        plt.scatter(np.arange(len(pos_scores)), pos_scores, label='Poison', alpha=0.5, color='red')
        plt.scatter(np.arange(len(clean_scores)), clean_scores, label='Clean', alpha=0.5, color='blue')
        if sus_scores is not None:
            plt.scatter(np.arange(len(sus_scores)), sus_scores, label='Clean Sus', alpha=0.5, color='green')

        threshold, kmeans_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans  = compute_thresholds(pos_scores, clean_scores, sus_scores)
        
        thresholds = [kmeans_threshold]
        print("thresholds: ", thresholds)
        labels = ['Threshold', 'KMeans Threshold']
        colors = ['black', 'purple']

        for thr, color, label in zip(thresholds, colors, labels):
            plt.axhline(y=thr, color=color, linestyle='--', label=label)

        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.axhline(y=threshold, color='black', linestyle='--')
        plt.savefig(figure_path + f"{title}.png")
        
        return threshold, kmeans_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans 

        

    # Compute different types of scores
    pos_mean = np.nanmean(pos_predictions_real, axis=1)
    pos_max = np.nanmax(pos_predictions_real, axis=1)
    pos_mean_max = pos_mean * pos_max

    clean_mean = np.nanmean(clean_predictions_real, axis=1)
    clean_max = np.nanmax(clean_predictions_real, axis=1)
    clean_mean_max = clean_mean * clean_max

    if len(sus_clean_predictions) > 0:
        sus_mean = np.nanmean(sus_clean_predictions, axis=1)
        sus_max = np.nanmax(sus_clean_predictions, axis=1)
        sus_mean_max = sus_mean * sus_max
    else:
        sus_mean = sus_max = sus_mean_max = None

    # Plot each type of score
    print("Mean")
    gaussian_threshold_mean,kmeans_threshold_mean,pos_gaussian_mean, fpos_gaussian_mean, pos_kmeans_mean, fpos_kmeans_mean = plot_scores(pos_mean, clean_mean, sus_mean, f"Iso Mean {attack} pr {poison_ratio} percentage {percentage}")
 
def score_poisoned_samples(sus_diff,clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, n_groups, dataset, cv_model, epochs, seed, device, poison_ratio, percentage, attack, figure_path = "./figures/", threshold=0.6):
    # predictions_with_indices, average_feature_importances, true_labels, predictions_proba, _, index_tracker = train_prov_data_custom(
    #     sus_diff, clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, n_groups, seed, device, model_name=cv_model)
    eps = 16
    prov_path = "./Training_Prov_Data/"
    real_clean_indices = np.unique(clean_inds)
    # with open(prov_path + f'avg_feature_importances_{attack}_{dataset}_{eps}_{poison_ratio}_{percentage}_{512}_aug_rf.pkl', 'wb') as f:
    #     pickle.dump(average_feature_importances, f)
  
    with open(prov_path + f'avg_feature_importances_{attack}_{dataset}_{eps}_{poison_ratio}_{percentage}_{512}_aug_rf.pkl', 'rb') as f:
        average_feature_importances = pickle.load(f) 
    average_original_feature_importances = average_feature_importances  
    
    z_scores = np.abs(stats.zscore(average_original_feature_importances)) 
    plt.figure()
    plt.plot(range(len(average_original_feature_importances)), average_original_feature_importances, label='Feature Importances', alpha=0.5, color='blue')
    plt.savefig(figure_path + f"Feature_importances.png")
    # outliers = np.where(z_scores > 1)[0]
    outliers = np.where(average_original_feature_importances > 0.001)[0]
    print("len outliers: ", len(outliers))  
    best_rel_feature = len(outliers)
    
    relevant_features = np.argsort(average_original_feature_importances)[::-1][:best_rel_feature]
    
    
    predictions_with_indices_2, _, true_labels_2, predictions_proba_2, _, index_tracker_2 = train_prov_data_custom(
            sus_diff[:,relevant_features], clean_diff[:,relevant_features], clean_inds, sus_inds, poison_indices,random_clean_sus_idx, n_groups, seed, device, model_name=cv_model) 
    
    # with open(prov_path + f'predictions_with_indices_2_{attack}_{dataset}_{eps}_{poison_ratio}_{percentage}_{512}_rf.pkl', 'wb') as f:
    #     pickle.dump(predictions_with_indices_2, f)
        
    # with open(prov_path + f'predictions_with_indices_2_{attack}_{dataset}_{eps}_{poison_ratio}_{percentage}_{512}_rf.pkl', 'rb') as f:
    #     predictions_with_indices_2 = pickle.load(f)
        
    # Initialize arrays
    pos_predictions_real = []
    clean_predictions_real = []
    sus_clean_predictions = []

    pos_prediction_indices = []
    clean_predicion_indices = []
    sus_clean_prediction_indices = []
    
    if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
        poison_indices_all = poison_indices
        poison_indices = np.concatenate(list(poison_indices_all.values()))

    epoch_num = 1000   # set to a large number 
    # Fill arrays
    for k, v in predictions_with_indices_2.items():
        if k in poison_indices:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            pos_predictions_real.append(v)
            pos_prediction_indices.append(k)
        elif k in real_clean_indices:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            clean_predictions_real.append(v)
            clean_predicion_indices.append(k)
        else:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            sus_clean_predictions.append(v)
            sus_clean_prediction_indices.append(k)
            
    
    # Convert lists to arrays
    pos_predictions_real = np.array(pos_predictions_real)
    clean_predictions_real = np.array(clean_predictions_real)
    sus_clean_predictions = np.array(sus_clean_predictions)

    pos_prediction_indices = np.array(pos_prediction_indices)
    clean_prediction_indices = np.array(clean_predicion_indices)
    sus_clean_prediction_indices = np.array(sus_clean_prediction_indices)

    

    def compute_thresholds(pos_scores, clean_scores, sus_scores):
        if sus_scores is not None:
            combined_data = np.concatenate([pos_scores, sus_scores]).reshape(-1, 1)
        else:
            combined_data = np.concatenate([pos_scores, clean_scores]).reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(combined_data)
        
        

        # Compute the GMM Threshold
        x = np.linspace(combined_data.min(), combined_data.max(), num=1000).reshape(-1, 1)
        pdf_individual = gmm.predict_proba(x) * np.exp(gmm.score_samples(x).reshape(-1, 1))
        diff_sign = np.diff(np.sign(pdf_individual[:, 0] - pdf_individual[:, 1]))

        gaussian_threshold = next(x[i, 0] for i in range(1, len(x)) if np.diff(np.sign(pdf_individual[:, 0] - pdf_individual[:, 1]))[i] != 0)
        # Compute the KMeans Threshold
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(combined_data)
        labels = kmeans.labels_
        cluster_0_points = combined_data[labels == 0]
        cluster_1_points = combined_data[labels == 1]
        boundary_points = [min(cluster_0_points), max(cluster_1_points)]
        kmeans_threshold = np.mean(boundary_points)

        # Compute the Outlier Threshold
        mean_score = np.mean(combined_data)
        std_score = np.std(combined_data)
        outlier_threshold = mean_score + 2 * std_score 
        
        print("TPR custom threshold: ", np.mean(pos_scores > threshold))
        print("TPR gaussian threshold: ", np.mean(pos_scores > gaussian_threshold))
        print("TPR kmeans: ", np.mean(pos_scores > kmeans_threshold))
        
        if sus_scores is not None:
            print("FPR custom threshold: ", np.mean(sus_scores > threshold))
            print("FPR gaussian: ", np.mean(sus_scores > gaussian_threshold))
            print("FPR kmeans: ", np.mean(sus_scores > kmeans_threshold))
            print("Real TPR custom threshold: ", np.mean(np.concatenate([pos_scores , sus_scores]) > threshold))
            print("Real TPR gaussian: ", np.mean(np.concatenate([pos_scores , sus_scores]) > gaussian_threshold))
            print("Real TPR kmeans: ", np.mean(np.concatenate([pos_scores , sus_scores]) > kmeans_threshold))
            pos_custom = np.mean(pos_scores > threshold)
            fpos_custom = np.mean(sus_scores > threshold)
            
            pos_gaussian = np.mean(np.concatenate([pos_scores , sus_scores]) > gaussian_threshold)
            fpos_gaussian = np.mean(clean_scores > gaussian_threshold)
            
            pos_kmeans = np.mean(np.concatenate([pos_scores , sus_scores]) > kmeans_threshold)
            fpos_kmeans = np.mean(clean_scores > kmeans_threshold)
        else:
            print("FPR custom threshold: ", np.mean(clean_scores > threshold))
            print("FPR gaussian: ", np.mean(clean_scores > gaussian_threshold))
            print("FPR kmeans: ", np.mean(clean_scores > kmeans_threshold))
            
            pos_custom = np.mean(pos_scores > threshold)
            fpos_custom = np.mean(clean_scores > threshold)
            
            pos_gaussian = np.mean(pos_scores > gaussian_threshold)
            fpos_gaussian = np.mean(clean_scores > gaussian_threshold)
            
            pos_kmeans = np.mean(pos_scores > kmeans_threshold)
            fpos_kmeans = np.mean(clean_scores > kmeans_threshold)
            
            

        return threshold, kmeans_threshold, gaussian_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans, pos_custom, fpos_custom

    def plot_scores(pos_scores, clean_scores, sus_scores, title):
        plt.figure(figsize=(12, 6))
        colors = ['blue', 'orange', 'red']
        plt.rcParams.update({'font.size': 16})
        if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
            start_index = 0
            for key in poison_indices_all:
                attack_indices = [i for i, ind in enumerate(pos_prediction_indices) if ind in poison_indices_all[key]]
                if key == "LabelConsistent":
                    key = "Label-Consistent"
                plt.scatter(np.arange(len(pos_scores[attack_indices])), pos_scores[attack_indices], label=f'Poison {key}', alpha=0.5, color=colors[start_index])
                start_index += 1
        else:
            plt.scatter(np.arange(len(pos_scores)), pos_scores, label='Poison', alpha=0.5, color='red')
        # plt.scatter(np.arange(len(clean_scores)), clean_scores, label='Clean', alpha=0.5, color='blue')
        if sus_scores is not None:
            plt.scatter(np.arange(len(sus_scores)), sus_scores, label='Clean Suspected', alpha=0.5, color='green')

        threshold, kmeans_threshold, gaussian_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans, pos_custom, fpos_custom = compute_thresholds(pos_scores, clean_scores, sus_scores)
        
        thresholds = [threshold, kmeans_threshold, gaussian_threshold]
        print("thresholds: ", thresholds)
        labels = ['Threshold', 'Threshold 1 (Kmeans)', 'Threshold 2 (Gaussian)']
        colors = ['black', 'green', 'orange']

            
        for thr, color, label in zip(thresholds[1:], colors[1:], labels[1:]):
            plt.axhline(y=thr, color=color, linestyle='--', label=label)

        plt.legend(
            loc='lower center', 
            bbox_to_anchor=(0.5, 1.02),  # Centered horizontally above the plot
            ncol=3,  # Number of columns for horizontal arrangement
            borderaxespad=0
        )
        # plt.axhline(y=threshold, color='black', linestyle='--')
        plt.xlabel("Number of samples")
        plt.ylabel("Poisoning Score")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room above
        plt.savefig(figure_path + f"{title}.png")


        
        return threshold, kmeans_threshold, gaussian_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans, pos_custom, fpos_custom


    def average_k_minimum_values(arr, k):
        # Mask NaN values with a large number
        nan_mask = np.isnan(arr)
        large_number = np.nanmax(arr[np.isfinite(arr)]) + 1
        arr_masked = np.where(nan_mask, large_number, arr)
        
        # Get the indices of the k smallest values along axis 1
        k_min_indices = np.argsort(arr_masked, axis=1)[:, :k]
        
        # Extract the k smallest values using the indices
        k_min_values = np.take_along_axis(arr, k_min_indices, axis=1)
        
        # Calculate the average of the k minimum values for each row
        k_min_averages = np.mean(k_min_values, axis=1)
        
        return k_min_averages

    j = 50
    # Compute different types of scores
    pos_mean = np.nanmean(pos_predictions_real, axis=1)
    pos_max = average_k_minimum_values(pos_predictions_real, j)
    pos_mean_max = pos_mean * pos_max

    clean_mean = np.nanmean(clean_predictions_real, axis=1)
    clean_max = average_k_minimum_values(clean_predictions_real, j)
    clean_mean_max = clean_mean * clean_max

    if len(sus_clean_predictions) > 0:
        sus_mean = np.nanmean(sus_clean_predictions, axis=1)
        sus_max = average_k_minimum_values(sus_clean_predictions, j)
        sus_mean_max = sus_mean * sus_max
    else:
        sus_mean = sus_max = sus_mean_max = None

    # Plot each type of score
    print("Mean")
    custom_threshold_mean,kmeans_threshold_mean, gaussian_threshold_mean, pos_gaussian_mean, fpos_gaussian_mean, pos_kmeans_mean, fpos_kmeans_mean, pos_custom_mean, fpos_custom_mean = plot_scores(pos_mean, clean_mean, sus_mean, f"SL Mean {attack} pr {poison_ratio} percentage {percentage} constant threshold")
    # print("Min")
    # gaussian_threshold_max,kmeans_threshold_max,pos_gaussian_max, fpos_gaussian_max, pos_kmeans_max, fpos_kmeans_max = plot_scores(pos_max, clean_max, sus_max, f"rf Min {attack} pr {poison_ratio} percentage {percentage}")
    best_config = "mean_gaussian"
    
    if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
        for key in poison_indices_all:
            attack_indices = [i for i, ind in enumerate(pos_prediction_indices) if ind in poison_indices_all[key]]
            print(f"TPR {key} kmeans threshold: ", np.mean(pos_mean[attack_indices] > kmeans_threshold_mean))
            print(f"FPR {key} kmeans threshold: ", np.mean(clean_mean[attack_indices] > kmeans_threshold_mean))
                
    if len(sus_clean_prediction_indices) > 0:
        values = {
        "mean_gaussian": np.concatenate([pos_mean, sus_mean]) > gaussian_threshold_mean,
        "mean_kmeans": np.concatenate([pos_mean, sus_mean]) > kmeans_threshold_mean,
        "mean_custom": np.concatenate([pos_mean, sus_mean]) > custom_threshold_mean,
        # "max_gaussian": np.concatenate([pos_max, sus_max]) > gaussian_threshold_max,
        # "max_kmeans": np.concatenate([pos_max, sus_max]) > kmeans_threshold_max,
        # "mean_max_gaussian":np.concatenate([pos_mean_max, sus_mean_max]) > gaussian_threshold_mean_max,
        # "mean_max_kmeans": np.concatenate([pos_mean_max, sus_mean_max]) > kmeans_threshold_mean_max
        }
        
        sus_prediction_indices = np.concatenate([pos_prediction_indices, sus_clean_prediction_indices])
        indexes_to_exclude = sus_prediction_indices[values[best_config]]
        return indexes_to_exclude
        
    else:
        values = {
        "mean_gaussian": pos_mean > gaussian_threshold_mean,
        "mean_kmeans": pos_mean > kmeans_threshold_mean,
        "mean_custom": pos_mean > custom_threshold_mean,
        # "max_gaussian": pos_max > gaussian_threshold_max,
        # "max_kmeans": pos_max > kmeans_threshold_max,
        # "mean_max_gaussian":pos_mean_max > gaussian_threshold_mean_max,
        # "mean_max_kmeans": pos_mean_max > kmeans_threshold_mean_max
        }
        
        print("values: ", len(pos_mean[pos_mean > gaussian_threshold_mean]))
        indexes_to_exclude = pos_prediction_indices[values[best_config]]
    
    
    # def plot_scores(pos_scores, clean_scores, sus_scores, title, figure_path, n_groups, thresholds):
    #     # Plot all data points
    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(np.arange(len(pos_scores)), pos_scores, label='Poison', alpha=0.5, color='red')
    #     # plt.scatter(np.arange(len(clean_scores)), clean_scores, label='Clean', alpha=0.5, color='blue')
    #     if sus_scores is not None and len(sus_scores) > 0:
    #         plt.scatter(np.arange(len(sus_scores)), sus_scores, label='Clean Sus', alpha=0.5, color='green')

    #     # Plot thresholds for each group
    #     group_size = len(pos_scores) // n_groups
    #     for i, (kmeans_thr, gaussian_thr) in enumerate(zip(thresholds['kmeans'], thresholds['gaussian'])):
    #         start_index = i * group_size
    #         end_index = (i + 1) * group_size if i < n_groups - 1 else len(pos_scores)
    #         plt.plot([start_index, end_index], [kmeans_thr, kmeans_thr], color='red', linestyle='--', alpha=0.5, label='KMeans Threshold' if i == 0 else "")
    #         plt.plot([start_index, end_index], [gaussian_thr, gaussian_thr], color='black', linestyle='--', alpha=0.5, label='Gaussian Threshold' if i == 0 else "")


    #     plt.title(title)
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #     plt.savefig(figure_path + f"{title}.png")
    #     plt.show()

    #     return thresholds

    # def divide_into_groups(data, n_groups):
    #     """Divides the data into n_groups."""
    #     group_size = len(data) // n_groups
    #     groups = [data[i * group_size:(i + 1) * group_size] for i in range(n_groups)]
    #     return groups

    # def process_data(pos_mean, clean_mean, sus_mean, pos_prediction_indices, sus_clean_prediction_indices, attack, poison_ratio, percentage,n_groups, best_config="mean_kmeans"):
    #     pos_groups = divide_into_groups(pos_mean, n_groups)
    #     clean_groups = divide_into_groups(clean_mean, n_groups)
    #     sus_groups = divide_into_groups(sus_mean, n_groups) if sus_mean is not None else None


    #     # Calculate thresholds for each group
    #     thresholds = {'kmeans': [], 'gaussian': []}
    #     for pos_group, clean_group, sus_group in zip(pos_groups, clean_groups, sus_groups):
    #         custom_threshold, kmeans_threshold, gaussian_threshold, _, _, _, _, _, _ = compute_thresholds(pos_group, clean_group, sus_group)
    #         thresholds['kmeans'].append(kmeans_threshold)
    #         thresholds['gaussian'].append(gaussian_threshold)

    #     print("Thresholds: ", thresholds)

    #     # Apply thresholds to exclude indices
    #     if len(sus_clean_prediction_indices) > 0:
    #         pos_values = np.concatenate([group > threshold for group, threshold in zip(pos_groups, thresholds['kmeans'])])
    #         pos_indices_to_exclude_kmeans = pos_prediction_indices[pos_values]
    #         try:
    #             sus_values = np.concatenate([group > threshold for group, threshold in zip(sus_groups, thresholds['kmeans'])])
    #             sus_indices_to_exclude_kmeans = sus_clean_prediction_indices[sus_values]
    #         except:
    #             sus_indices_to_exclude_kmeans = []

    #         print("TPR kmeans: ", len(set(pos_indices_to_exclude_kmeans) & set(pos_prediction_indices))/len(pos_prediction_indices))
    #         print("FPR kmeans: ", len(set(sus_indices_to_exclude_kmeans) & set(sus_clean_prediction_indices))/len(sus_clean_prediction_indices))
            
    #         pos_values = np.concatenate([group > threshold for group, threshold in zip(pos_groups, thresholds['gaussian'])])
    #         pos_indices_to_exclude_gaussian = pos_prediction_indices[pos_values]
    #         try:
    #             sus_values = np.concatenate([group > threshold for group, threshold in zip(sus_groups, thresholds['gaussian'])])
    #             sus_indices_to_exclude_gaussian = sus_clean_prediction_indices[sus_values]
    #         except:
    #             sus_indices_to_exclude_gaussian = []
            
    #         print("TPR gaussian: ", len(set(pos_indices_to_exclude_gaussian) & set(pos_prediction_indices))/len(pos_prediction_indices))
    #         print("FPR gaussian: ", len(set(sus_indices_to_exclude_gaussian) & set(sus_clean_prediction_indices))/len(sus_clean_prediction_indices))
    #         if best_config == "mean_kmeans":
    #             indexes_to_exclude = np.concatenate([pos_indices_to_exclude_kmeans, sus_indices_to_exclude_kmeans])
    #         elif best_config == "mean_gaussian":
    #             indexes_to_exclude = np.concatenate([pos_indices_to_exclude_gaussian, sus_indices_to_exclude_gaussian])
    #         else:
    #             raise ValueError("Invalid best_config value. Must be one of ['mean_kmeans', 'mean_gaussian']")
    #     else:
    #         values = np.concatenate([group > threshold for group, threshold in zip(pos_groups, selected_thresholds)])
    #         indexes_to_exclude = pos_prediction_indices[values]
            
    #     plot_scores(pos_mean, clean_mean, sus_mean, f"SL Mean {attack} pr {poison_ratio} percentage {percentage}", figure_path, n_groups=n_groups, thresholds=thresholds)


    #     return indexes_to_exclude
    # indexes_to_exclude = process_data(pos_mean, clean_mean, sus_mean, pos_prediction_indices, sus_clean_prediction_indices, attack, poison_ratio, percentage, n_groups=5, best_config="mean_gaussian")
            
    return indexes_to_exclude

def score_poisoned_samples_2(sus_diff, clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, n_groups, rf, epochs, seed, poison_ratio, percentage, eps, attack, figure_path = "./figures/", threshold=0.5):
    
    # is_poison = np.array([set(idxs) & set(poison_indices) != set() for idxs in sus_inds])
    is_poison = np.array([{idxs} & set(poison_indices) != set() for idxs in sus_inds])
    preds = rf.predict_proba(np.concatenate([clean_diff, sus_diff]))[:, 1]
    plt.scatter(range(len(preds[:len(clean_diff)])), preds[:len(clean_diff)], alpha=0.5, color='blue')
    plt.scatter(range(len(preds[len(clean_diff):][is_poison])), preds[len(clean_diff):][is_poison], alpha=0.5, color='red')
    plt.scatter(range(len(preds[len(clean_diff):][~is_poison])), preds[len(clean_diff):][~is_poison], alpha=0.5, color='green')
    plt.title(f"RF Model Forest Predictions for {attack} attack with {poison_ratio} poison ratio and {percentage} percentage")
    plt.axhline(y=-0.2, color='black', linestyle='--')
    plt.savefig(figure_path + f"rf_{attack}_{dataset}_{eps}_{percentage}_{poison_ratio}.png") 
     
    y = np.concatenate([np.zeros(len(clean_diff)), np.ones(len(sus_diff))])
    print("accuracy: ", np.mean(y == (preds > 0.5)))
    print("tpr: ", np.mean(preds[len(clean_diff):][is_poison] > 0.5))
    print("fpr: ", np.mean(preds[len(clean_diff):][~is_poison] > 0.5))
    
    
    predictions_with_indices = {}
    all_inds = np.arange(len(clean_inds) + len(sus_inds))
    concat_inds = np.concatenate([clean_inds, sus_inds])
    for i in all_inds:
        ind = concat_inds[i]
        if ind in predictions_with_indices:
            predictions_with_indices[ind].append(preds[i])
        else:
            predictions_with_indices[ind] = [preds[i]]
    
    real_clean_indices = np.unique(clean_inds)
    
    # Initialize arrays
    pos_predictions_real = []
    clean_predictions_real = []
    sus_clean_predictions = []

    pos_prediction_indices = []
    clean_predicion_indices = []
    sus_clean_prediction_indices = []

    epoch_num = 2000   # set to a large number 
    # Fill arrays
    for k, v in predictions_with_indices.items():
        if k in poison_indices:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            pos_predictions_real.append(v)
            pos_prediction_indices.append(k)
        elif k in real_clean_indices:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            clean_predictions_real.append(v)
            clean_predicion_indices.append(k)
        else:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            sus_clean_predictions.append(v)
            sus_clean_prediction_indices.append(k)
                
    # unique_sus_inds = np.unique(np.concatenate(sus_inds))
    

    # for i in range(len(concat_inds)):
    #     for idx in concat_inds[i]:
    #         if idx in unique_sus_inds:
    #             if idx in predictions_with_indices:
    #                 if predictions_proba[i] < threshold:
    #                     predictions_with_indices[idx].append(preds[i] * penalty) 
    #                 else:
    #                     predictions_with_indices[idx].append(preds[i])
    #             else:
    #                 if predictions_proba[i] < threshold:
    #                     predictions_with_indices[idx] = [preds[i] * penalty]
    #                 else:
    #                     predictions_with_indices[idx] = [preds[i]]
    #         else:
    #             if idx in predictions_with_indices:
    #                 predictions_with_indices[idx].append(preds[i])
    #             else:
    #                 predictions_with_indices[idx] = [preds[i]]
                
    # clean_inds_flat = np.concatenate(clean_inds)
    # real_clean_indices = np.unique(clean_inds_flat)
            

    # Convert lists to arrays
    pos_predictions_real = np.array(pos_predictions_real)
    clean_predictions_real = np.array(clean_predictions_real)
    sus_clean_predictions = np.array(sus_clean_predictions)

    pos_prediction_indices = np.array(pos_prediction_indices)
    clean_prediction_indices = np.array(clean_predicion_indices)
    sus_clean_prediction_indices = np.array(sus_clean_prediction_indices)

 

    def compute_thresholds(pos_scores, clean_scores, sus_scores):
        if sus_scores is not None:
            combined_data = np.concatenate([pos_scores, sus_scores]).reshape(-1, 1)
        else:
            combined_data = np.concatenate([pos_scores, clean_scores]).reshape(-1, 1)

        # gmm = GaussianMixture(n_components=2, random_state=42)
        # gmm.fit(combined_data)
        
        

        # # Compute the GMM Threshold
        # x = np.linspace(combined_data.min(), combined_data.max(), num=1000).reshape(-1, 1)
        # pdf_individual = gmm.predict_proba(x) * np.exp(gmm.score_samples(x).reshape(-1, 1))
        # diff_sign = np.diff(np.sign(pdf_individual[:, 0] - pdf_individual[:, 1]))

        # gaussian_threshold = next(x[i, 0] for i in range(1, len(x)) if np.diff(np.sign(pdf_individual[:, 0] - pdf_individual[:, 1]))[i] != 0)
        # Compute the KMeans Threshold
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(combined_data)
        labels = kmeans.labels_
        cluster_0_points = combined_data[labels == 0]
        cluster_1_points = combined_data[labels == 1]
        boundary_points = [min(cluster_0_points), max(cluster_1_points)]
        kmeans_threshold = np.mean(boundary_points)

        # Compute the Outlier Threshold
        mean_score = np.mean(combined_data)
        std_score = np.std(combined_data)
        outlier_threshold = mean_score + 2 * std_score 
        
        print("TPR custom threshold: ", np.mean(pos_scores > threshold))
        print("TPR kmeans: ", np.mean(pos_scores > kmeans_threshold))
        
        if sus_scores is not None:
            print("FPR gaussian: ", np.mean(sus_scores > threshold))
            print("FPR kmeans: ", np.mean(sus_scores > kmeans_threshold))
            print("Real TPR gaussian: ", np.mean(np.concatenate([pos_scores , sus_scores]) > threshold))
            print("Real TPR kmeans: ", np.mean(np.concatenate([pos_scores , sus_scores]) > kmeans_threshold))
            pos_gaussian = np.mean(np.concatenate([pos_scores , sus_scores]) > threshold)
            fpos_gaussian = np.mean(clean_scores > threshold)
            
            pos_kmeans = np.mean(np.concatenate([pos_scores , sus_scores]) > kmeans_threshold)
            fpos_kmeans = np.mean(clean_scores > kmeans_threshold)
        else:
            print("FPR gaussian: ", np.mean(clean_scores > threshold))
            print("FPR kmeans: ", np.mean(clean_scores > kmeans_threshold))
            pos_gaussian = np.mean(pos_scores > threshold)
            fpos_gaussian = np.mean(clean_scores > threshold)
            
            pos_kmeans = np.mean(pos_scores > kmeans_threshold)
            fpos_kmeans = np.mean(clean_scores > kmeans_threshold)
            
            

        return threshold, kmeans_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans 

    def plot_scores(pos_scores, clean_scores, sus_scores, title):
        plt.figure(figsize=(10, 6))
        plt.scatter(np.arange(len(pos_scores)), pos_scores, label='Poison', alpha=0.5, color='red')
        # plt.scatter(np.arange(len(clean_scores)), clean_scores, label='Clean', alpha=0.5, color='blue')
        if sus_scores is not None:
            plt.scatter(np.arange(len(sus_scores)), sus_scores, label='Clean Sus', alpha=0.5, color='green')

        threshold, kmeans_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans  = compute_thresholds(pos_scores, clean_scores, sus_scores)
        
        thresholds = [kmeans_threshold]
        print("thresholds: ", thresholds)
        labels = ['Threshold', 'KMeans Threshold']
        colors = ['black', 'purple']

        for thr, color, label in zip(thresholds, colors, labels):
            plt.axhline(y=thr, color=color, linestyle='--', label=label)

        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.axhline(y=threshold, color='black', linestyle='--')
        plt.savefig(figure_path + f"{title}.png")
        
        return threshold, kmeans_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans 


    def average_k_minimum_values(arr, k):
        # Mask NaN values with a large number
        nan_mask = np.isnan(arr)
        large_number = np.nanmax(arr[np.isfinite(arr)]) + 1
        arr_masked = np.where(nan_mask, large_number, arr)
        
        # Get the indices of the k smallest values along axis 1
        k_min_indices = np.argsort(arr_masked, axis=1)[:, :k]
        
        # Extract the k smallest values using the indices
        k_min_values = np.take_along_axis(arr, k_min_indices, axis=1)
        
        # Calculate the average of the k minimum values for each row
        k_min_averages = np.mean(k_min_values, axis=1)
        
        return k_min_averages

    j = 50
    # Compute different types of scores
    pos_mean = np.nanmean(pos_predictions_real, axis=1)
    pos_max = average_k_minimum_values(pos_predictions_real, j)
    pos_mean_max = pos_mean * pos_max

    clean_mean = np.nanmean(clean_predictions_real, axis=1)
    clean_max = average_k_minimum_values(clean_predictions_real, j)
    clean_mean_max = clean_mean * clean_max

    if len(sus_clean_predictions) > 0:
        sus_mean = np.nanmean(sus_clean_predictions, axis=1)
        sus_max = average_k_minimum_values(sus_clean_predictions, j)
        sus_mean_max = sus_mean * sus_max
    else:
        sus_mean = sus_max = sus_mean_max = None

    # Plot each type of score
    print("Mean")
    gaussian_threshold_mean,kmeans_threshold_mean,pos_gaussian_mean, fpos_gaussian_mean, pos_kmeans_mean, fpos_kmeans_mean = plot_scores(pos_mean, clean_mean, sus_mean, f"rf Mean {attack} pr {poison_ratio} percentage {percentage}")
    # print("Min")
    # gaussian_threshold_max,kmeans_threshold_max,pos_gaussian_max, fpos_gaussian_max, pos_kmeans_max, fpos_kmeans_max = plot_scores(pos_max, clean_max, sus_max, f"rf Min {attack} pr {poison_ratio} percentage {percentage}")
    best_config = "mean_gaussian"
    if len(sus_clean_prediction_indices) > 0:
        values = {
        "mean_gaussian": np.concatenate([pos_mean, sus_mean]) > threshold,
        # "mean_kmeans": np.concatenate([pos_mean, sus_mean]) > kmeans_threshold_mean,
        # "max_gaussian": np.concatenate([pos_max, sus_max]) > gaussian_threshold_max,
        # "max_kmeans": np.concatenate([pos_max, sus_max]) > kmeans_threshold_max,
        # "mean_max_gaussian":np.concatenate([pos_mean_max, sus_mean_max]) > gaussian_threshold_mean_max,
        # "mean_max_kmeans": np.concatenate([pos_mean_max, sus_mean_max]) > kmeans_threshold_mean_max
        }
        
        sus_prediction_indices = np.concatenate([pos_prediction_indices, sus_clean_prediction_indices])
        indexes_to_exculde = sus_prediction_indices[values[best_config]]
        return indexes_to_exculde
        
    else:
        values = {
        "mean_gaussian": pos_mean > gaussian_threshold_mean,
        # "mean_kmeans": pos_mean > kmeans_threshold_mean,
        # "max_gaussian": pos_max > gaussian_threshold_max,
        # "max_kmeans": pos_max > kmeans_threshold_max,
        # "mean_max_gaussian":pos_mean_max > gaussian_threshold_mean_max,
        # "mean_max_kmeans": pos_mean_max > kmeans_threshold_mean_max
        }
        
        print("values: ", len(pos_mean[pos_mean > gaussian_threshold_mean]))
        indexes_to_exculde = pos_prediction_indices[values[best_config]]
            
    return indexes_to_exculde


def get_diff(sus_inds, clean_inds, clean_inds_2):
    sus_indices = np.array([
        image_idx
        for epoch in sus_inds
        for image_idx in sus_inds[epoch]
    ])
    sus_array = np.array([
        sus_inds[epoch][image_idx]
        for epoch in sus_inds
        for image_idx in sus_inds[epoch]
    ])

    clean_indices = np.array([
        image_idx[0]
        for epoch in clean_inds
        for image_idx
        in clean_inds[epoch]
    ])
    clean_array= np.array([
        clean_inds[epoch][image_idx]
        for epoch in clean_inds
        for image_idx in clean_inds[epoch]
    ])


    clean_2_array = np.array([
        clean_inds_2[epoch][image_idx]
        for epoch in clean_inds_2
        for image_idx in clean_inds_2[epoch]
    ])


    sus_diff = sus_array - clean_2_array
    clean_diff = clean_array - clean_2_array
    return sus_diff, clean_diff, sus_indices, clean_indices


def main():
    args = parse_args()
    
    batch_level_1 = args.batch_level_1
    batch_level_2 = args.batch_level_2
    clean_training = args.clean_training
    poisoned_training = args.poisoned_training
    sample_level = args.sample_level
    ext_rel_wts = args.ext_rel_wts
    score_samples = args.score_samples
    retrain = args.retrain
    pr_sus = args.pr_sus
    ep_bl = args.ep_bl
    ep_bl_base = args.ep_bl_base
    ep_sl = args.ep_sl
    ep_sl_base = args.ep_sl_base
    pr_tgt = args.pr_tgt
    bs_sl = args.bs_sl
    bs_bl = args.bs_bl
    bs = args.bs
    target_class = args.target_class
    source_class = args.source_class
    dataset = args.dataset
    attack = args.attack
    orig_model = args.model
    dataset_dir = args.dataset_dir
    clean_model_path = args.clean_model_path
    saved_models_path = args.saved_models_path
    global_seed = args.global_seed
    gpu_id = args.gpu_id
    lr = args.lr
    figure_path = args.figure_path
    prov_path = args.prov_path
    epochs = args.epochs
    scenario = args.scenario
    min_features = args.min_features
    max_features = args.max_features
    max_trials = args.max_trials
    get_result = args.get_result
    force = args.force
    threshold = args.threshold
    sample_from_test = args.sample_from_test
    penalty = args.penalty
    eps = args.eps
    cv_model = args.cv_model
    vis = args.vis
    groups = args.groups
    opt = args.opt
    training_mode = args.training_mode
    analyze = args.analyze

    
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed) 

    print("Attack:", attack, "Dataset:", dataset, "Model:", orig_model, "Target Class:", target_class, "Poison Ratio Training Set:", pr_tgt, "Percentage Suspected:", pr_sus)
    print("Getting poisoned data...")
    device = torch.device(f"cuda")
    print(f"Using GPU {gpu_id}")
    if orig_model == "ResNet18":
        orig_model = ResNet(18)   
    elif orig_model == "CustomCNN":
        orig_model = CustomCNN() 
    elif orig_model == "BasicResNet":
        orig_model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=10)
    elif orig_model == "CustomResNet18":
        orig_model = CustomResNet18()
    elif orig_model == "ViT" and scenario == "fine_tuning":
        orig_model = CustomViT('B_16_imagenet1k', pretrained=True)
        if dataset == "slt10" or dataset == "CIFAR10":
            num_classes = 10
        elif dataset == "imagenet":
            num_classes = 100
        orig_model.fc = nn.Linear(orig_model.fc.in_features, num_classes)
    elif orig_model == "ViT" and scenario == "from_scratch":
        raise ValueError("ViT model not supported for from_scratch scenario")
    else:
        raise ValueError("Model not supported")     
        
    if scenario == "fine_tuning" and (attack != "sa" or dataset != "slt10") and (attack != "lc" or dataset != "imagenet"):
        orig_model.load_state_dict(torch.load(clean_model_path))
        if attack == "ht" and dataset == "CIFAR10":
            for i, param in enumerate(orig_model.parameters()):
                param.requires_grad = False
            orig_model[20] = nn.Linear(4096, 10)
        elif attack == "ht" and dataset == "slt10": 
            for param in orig_model.parameters():
                param.requires_grad = False
            for param in orig_model.fc.parameters():
                param.requires_grad = True 
        elif attack == 'ht' and dataset == "imagenet":
            for param in orig_model.parameters():
                param.requires_grad = False
            for param in orig_model.fc1.parameters():
                param.requires_grad = True 
            for param in orig_model.fc2.parameters():
                param.requires_grad = True
    
    elif attack == "sa" and dataset == "slt10":
        orig_model = orig_model.to(device)
    elif attack == "lc" and dataset == "imagenet":
        # orig_model.load_state_dict(torch.load(clean_model_path))
        orig_model = orig_model.to(device)
    elif scenario != "from_scratch":
        raise ValueError("Scenario not supported")
        
    if pr_sus == int(pr_sus):
        pr_sus = int(pr_sus)         

    if dataset == "CIFAR10":
        if attack == "lc": 
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = get_lc_cifar10_poisoned_data(pr_tgt, target_class, dataset_dir, copy.deepcopy(orig_model), clean_model_path, eps, vis, global_seed, gpu_id)
        elif attack == "narcissus":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = get_narcissus_cifar10_poisoned_data(pr_tgt, target_class, dataset_dir, copy.deepcopy(orig_model), eps, global_seed)
        elif attack == "sa":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = get_sa_cifar10_poisoned_data(pr_tgt, target_class, source_class, dataset_dir, copy.deepcopy(orig_model), clean_model_path, global_seed)
        elif attack == "ht":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = get_ht_cifar10_poisoned_data(pr_tgt, target_class, source_class, copy.deepcopy(orig_model), dataset_dir,  clean_model_path, global_seed)
        elif attack == "narcissus_lc":    
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices_all = get_lc_narcissus_cifar_10_poisoned_data(pr_tgt, target_class, dataset_dir, copy.deepcopy(orig_model), clean_model_path, vis, global_seed,gpu_id)
            poison_indices = np.concatenate(list(poison_indices_all.values())) 
        elif attack == "narcissus_lc_sa":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices_all = get_lc_narcissus_sa_cifar_10_poisoned_data(pr_tgt, target_class, dataset_dir, copy.deepcopy(orig_model), clean_model_path, vis, global_seed,gpu_id)
            poison_indices = np.concatenate(list(poison_indices_all.values()))
        else:
            raise ValueError("Attack not supported") 
    elif dataset == "slt10":
        if attack == "sa":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = get_sa_slt_10_poisoned_data(pr_tgt, target_class, source_class, dataset_dir, copy.deepcopy(orig_model), clean_model_path, global_seed)
        elif attack == "ht":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = get_ht_stl10_poisoned_data(pr_tgt, target_class, source_class, copy.deepcopy(orig_model), dataset_dir,  clean_model_path, global_seed)
            
    elif dataset == "imagenet":
        if attack == "ht":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = get_ht_imagenet_poisoned_data(pr_tgt, target_class, source_class, copy.deepcopy(orig_model), dataset_dir,  clean_model_path, global_seed)
        if attack == "lc":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = get_lc_image_net_poisoned_data(pr_tgt, target_class, dataset_dir, copy.deepcopy(orig_model), clean_model_path, eps, global_seed, gpu_id)
    
    if max_trials > 1:
        with open(prov_path + f'indexes_to_exculde_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}.pkl', 'rb') as f:
            indexes_to_exculde = pickle.load(f)
        poisoned_train_dataset = refine_poisoned_dataset(poisoned_train_dataset,indexes_to_exculde)
        print("Refined poisoned dataset length:", len(poisoned_train_dataset))
        poison_indices = set(poison_indices) - set(indexes_to_exculde)
    
    if clean_training:
        # print(poison_indices)
        poisoned_train_loader, test_loader, poisoned_test_loader, _ = get_loaders_from_dataset(poisoned_train_dataset, test_dataset, poisoned_test_dataset, bs, target_class, poison_indices) 
        model = copy.deepcopy(orig_model)
        model.to(device)
        if opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif opt == "sgd":
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, dampening=0, nesterov=True)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 
        
        if get_result:
            model.load_state_dict(torch.load(saved_models_path + f'clean_model_{attack}_{dataset}_{eps}_{pr_tgt}_{epochs}.pkl'))
            evaluate_model(model, test_loader, poisoned_test_loader, criterion, device)
        else:
            model, optr, slr, train_ACC, test_ACC, clean_ACC, target_ACC = train(model,optimizer, opt, scheduler, criterion, poisoned_train_loader,test_loader, poisoned_test_loader, epochs, global_seed, device, training_mode)
            torch.save(model.state_dict(), saved_models_path + f'clean_model_{attack}_{dataset}_{eps}_{pr_tgt}_{epochs}.pkl')
        
        
    if poisoned_training:
        poisoned_train_loader, test_loader, poisoned_test_loader, _ = get_loaders_from_dataset(poisoned_train_dataset, test_dataset, poisoned_test_dataset, bs, target_class) 
        model = copy.deepcopy(orig_model) 
        model.to(device)
        if opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif opt == "sgd":
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, dampening=0, nesterov=True)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 
        if get_result:
            model.load_state_dict(torch.load(saved_models_path + f'model_{attack}_{dataset}_{eps}_{pr_tgt}_{epochs}_{bs}.pkl'))
            evaluate_model(model, test_loader, poisoned_test_loader, criterion, device)
        else:
            model, optr, slr, train_ACC, test_ACC, clean_ACC, target_ACC = train(model,optimizer, opt, scheduler, criterion, poisoned_train_loader,test_loader, poisoned_test_loader, epochs, global_seed, device, training_mode)
            torch.save(model.state_dict(), saved_models_path + f'model_{attack}_{dataset}_{eps}_{pr_tgt}_{epochs}_{bs}.pkl') 
            torch.save(optr.state_dict(), saved_models_path + f'optimizer_{attack}_{dataset}_{eps}_{pr_tgt}_{epochs}_{bs}.pkl')
            torch.save(slr.state_dict(), saved_models_path + f'scheduler_{attack}_{dataset}_{eps}_{pr_tgt}_{epochs}_{bs}.pkl')
        
    if batch_level_1:
        poisoned_train_loader, test_loader, poisoned_test_loader, target_class_indices = get_loaders_from_dataset(poisoned_train_dataset, test_dataset, poisoned_test_dataset, bs_bl, target_class) 
        model = copy.deepcopy(orig_model)
        model.to(device)
        if opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif opt == "sgd":
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, dampening=0, nesterov=True) 
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep_bl_base+ep_bl) 
        
        if ep_bl_base > 0:
            model_path = saved_models_path + f'model_{attack}_{dataset}_{eps}_{pr_tgt}_{ep_bl_base}_{bs_bl}.pkl'
            optimizer_path = saved_models_path + f'optimizer_{attack}_{dataset}_{eps}_{pr_tgt}_{ep_bl_base}_{bs_bl}.pkl'
            scheduler_path = saved_models_path + f'scheduler_{attack}_{dataset}_{eps}_{pr_tgt}_{ep_bl_base}_{bs_bl}.pkl'
            if os.path.exists(model_path) and os.path.exists(optimizer_path) and os.path.exists(scheduler_path) and not force:
                model.load_state_dict(torch.load(model_path))
                optimizer.load_state_dict(torch.load(optimizer_path))
                scheduler.load_state_dict(torch.load(scheduler_path))
                print(f"Loaded model trained for {ep_bl_base} epochs from saved file")
            else:
                print(f"Training model for {ep_bl_base} epochs before capturing batch level weight updates...")
                model, optr, slr, train_ACC, test_ACC, clean_ACC, target_ACC = train(model,optimizer, opt, scheduler, criterion, poisoned_train_loader,test_loader, poisoned_test_loader, ep_bl_base, global_seed, device, training_mode)
                torch.save(model.state_dict(), model_path)
                torch.save(optr.state_dict(), optimizer_path)
                torch.save(slr.state_dict(), scheduler_path)
            print(f"Model trained for {ep_bl_base} epochs and saved")

        poison_amount = len(poison_indices)
        ignore_set = set(poison_indices)
        random_sus_idx = get_random_poison_idx(pr_sus, ignore_set, poison_indices, target_class_indices, poison_amount, global_seed)
        print("Suspected samples length:",len(random_sus_idx), "Poison ratio training set", pr_tgt, "Poison percentage suspected dataset", pr_sus)


        
        # sus_diff, clean_diff, sus_inds, clean_inds, important_features = capture_batch_level_weight_updates(random_sus_idx, poison_indices, model, orig_model, optimizer, opt,  scheduler,criterion, ep_bl, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class,sample_from_test, device, global_seed, figure_path, k=1)
        important_features = capture_first_level_multi_epoch_batch_sample_weight_updates(random_sus_idx, model, orig_model, optimizer, opt, scheduler, criterion, ep_bl, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class, sample_from_test, attack, device, global_seed, figure_path, training_mode, k=1)
        # important_features = capture_first_level_batch_sample_weight_updates(random_sus_idx, model, orig_model, optimizer, opt, scheduler, criterion, ep_bl, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class, sample_from_test, attack, device, global_seed, figure_path, training_mode, k=1)

            
        with open(prov_path + f'important_features_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_k_1.pkl', 'wb') as f:
            pickle.dump(important_features, f)
        # important_features = important_features[relevant_features]
        print("Important features shape:", important_features.shape)

        # model = copy.deepcopy(orig_model)
        # model.to(device)
        # if opt == "adam":
        #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # elif opt == "sgd":
        #     optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, dampening=0, nesterov=True)
        # criterion = nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep_bl_base+ep_bl) 
        
        # model.load_state_dict(torch.load(model_path))
        # optimizer.load_state_dict(torch.load(optimizer_path))
        # scheduler.load_state_dict(torch.load(scheduler_path))
        
        # if attack == "ht":
        #     attack = "ht_1"
        
        # sus_diff, clean_diff, sus_inds, clean_inds = capture_batch_sample_level_weight_updates(random_sus_idx, model, orig_model, optimizer, opt,  scheduler, criterion, ep_bl, lr, poisoned_train_loader, test_loader, poisoned_test_loader,important_features, target_class, sample_from_test, attack, device, global_seed, figure_path)
        # print("Shape of suspected weight updates:", sus_diff.shape, "Shape of clean weight updates:", clean_diff.shape) \
        
        # if attack == "ht_1":
        #     attack = "ht"
        # # save the provenance data
        
        # with open(prov_path + f'sus_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_check.pkl', 'wb') as f:
        #     pickle.dump(sus_diff, f)
        # with open(prov_path + f'clean_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_check.pkl', 'wb') as f:
        #     pickle.dump(clean_diff, f)
        # with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_check.pkl', 'wb') as f:
        #     pickle.dump(sus_inds, f)
        # with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_check.pkl', 'wb') as f:
        #     pickle.dump(clean_inds, f)   

        # # check the results
        # relevant_features, rf, predictions_with_indices, true_labels, predictions_proba, index_tracker = get_relevant_weight_dimensions(sus_diff, clean_diff, sus_inds, clean_inds, poison_indices, random_sus_idx,pr_sus, pr_tgt, bs_bl, attack,dataset, cv_model, global_seed, device, eps, figure_path, min_features, max_features)
        
        # del clean_diff  
        
    if batch_level_2:
        poisoned_train_loader, test_loader, poisoned_test_loader, target_class_indices = get_loaders_from_dataset(poisoned_train_dataset, test_dataset, poisoned_test_dataset, bs_bl, target_class) 
        model = copy.deepcopy(orig_model)
        model.to(device)
        if opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif opt == "sgd":
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, dampening=0, nesterov=True)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep_bl_base+ep_bl) 
        if ep_bl_base > 0:
            model_path = saved_models_path + f'model_{attack}_{dataset}_{eps}_{pr_tgt}_{ep_bl_base}_{bs_bl}.pkl'
            optimizer_path = saved_models_path + f'optimizer_{attack}_{dataset}_{eps}_{pr_tgt}_{ep_bl_base}_{bs_bl}.pkl'
            scheduler_path = saved_models_path + f'scheduler_{attack}_{dataset}_{eps}_{pr_tgt}_{ep_bl_base}_{bs_bl}.pkl'
            if os.path.exists(model_path) and os.path.exists(optimizer_path) and os.path.exists(scheduler_path) and not force:
                model.load_state_dict(torch.load(model_path))
                optimizer.load_state_dict(torch.load(optimizer_path))
                scheduler.load_state_dict(torch.load(scheduler_path))
                print(f"Loaded model trained for {ep_bl_base} epochs from saved file")
            else:
                print(f"Training model for {ep_bl_base} epochs before capturing batch level weight updates...")
                model, optr, slr, train_ACC, test_ACC, clean_ACC, target_ACC = train(model,optimizer, opt, scheduler, criterion, poisoned_train_loader,test_loader, poisoned_test_loader, ep_bl_base, global_seed, device, training_mode)
                torch.save(model.state_dict(), model_path)
                torch.save(optr.state_dict(), optimizer_path)
                torch.save(slr.state_dict(), scheduler_path)
            print(f"Model trained for {ep_bl_base} epochs and saved")
        poison_amount = len(poison_indices)
        ignore_set = set(poison_indices)
        random_sus_idx = get_random_poison_idx(pr_sus, ignore_set, poison_indices, target_class_indices, poison_amount, global_seed)
        print("Suspected samples length:",len(random_sus_idx), "Poison ratio training set", pr_tgt, "Poison percentage suspected dataset", pr_sus)
        
        
        # with open(prov_path + f'relevant_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_z_score_1.pkl', 'rb') as f:
        # #     relevant_features = pickle.load(f) 
        with open(prov_path + f'important_features_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_k_1.pkl', 'rb') as f:
            important_features = pickle.load(f)
        # important_features = important_features[relevant_features]
        print("Important features shape:", important_features.shape)
        

        sus_diff, clean_diff, sus_inds, clean_inds = capture_batch_sample_level_weight_updates(random_sus_idx, model, orig_model, optimizer, opt,  scheduler, criterion, ep_bl, lr, poisoned_train_loader, test_loader, poisoned_test_loader,important_features, target_class, sample_from_test, attack, device, global_seed, figure_path)
        print("Shape of suspected weight updates:", sus_diff.shape, "Shape of clean weight updates:", clean_diff.shape) 
        

        # save the provenance data
        
        with open(prov_path + f'sus_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_single_batch.pkl', 'wb') as f:
            pickle.dump(sus_diff, f)
        del sus_diff
        with open(prov_path + f'clean_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_single_batch.pkl', 'wb') as f:
            pickle.dump(clean_diff, f)
        with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_single_batch.pkl', 'wb') as f:
            pickle.dump(sus_inds, f)
        with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_single_batch.pkl', 'wb') as f:
            pickle.dump(clean_inds, f)   
        # with open(prov_path + f'important_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}.pkl', 'wb') as f:
        #     pickle.dump(important_features, f)
            
        del clean_diff   
    
    if ext_rel_wts:
        _, _, _, target_class_indices = get_loaders_from_dataset(poisoned_train_dataset, test_dataset, poisoned_test_dataset, bs_bl, target_class)      
        poison_amount = len(poison_indices)
        ignore_set = set(poison_indices)
        
        random_sus_idx = get_random_poison_idx(pr_sus, ignore_set, poison_indices, target_class_indices, poison_amount, global_seed)
        print("Suspected samples length:",len(random_sus_idx), "Poison ratio training set", pr_tgt, "Poison percentage suspected dataset", pr_sus)
        
        with open(prov_path + f'sus_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_single_batch.pkl','rb') as f:
            sus_diff = pickle.load(f)
        # sus_diff = sus_diff[:4*len(sus_diff)//5]
        with open(prov_path + f'clean_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_single_batch.pkl','rb') as f:
            clean_diff = pickle.load(f)
        # clean_diff = clean_diff[:4*len(sus_diff)//5:]
        with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_single_batch.pkl','rb') as f: 
            sus_inds = pickle.load(f)
        # sus_inds = sus_inds[:4*len(sus_diff)//5]
        with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_single_batch.pkl','rb') as f:
            clean_inds = pickle.load(f)
        # clean_inds = clean_inds[:4*len(sus_diff)//5]
        print("Shape of suspected weight updates:", sus_diff.shape, "Shape of clean weight updates:", clean_diff.shape) 
        
        # with open(prov_path + f'relevant_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{128}_st.pkl', 'rb') as f:
        #     important_features = pickle.load(f) 
        # sus_diff, clean_diff = sus_diff[:, important_features], clean_diff[:, important_features]
        # sus_diff, clean_diff, sus_inds, clean_inds = sus_diff[len(sus_diff)//3:], clean_diff[len(clean_diff)//3:], sus_inds[len(sus_inds)//3:], clean_inds[len(clean_inds)//3:]
        #     print("Relevant features shape:", relevant_features.shape)
        
        start_time = time.time()
        
        
        relevant_features, rf, predictions_with_indices, true_labels, predictions_proba, index_tracker = get_relevant_weight_dimensions(sus_diff, clean_diff, sus_inds, clean_inds, poison_indices, random_sus_idx,pr_sus, pr_tgt, bs_bl, attack,dataset, cv_model, global_seed, device, eps, figure_path, min_features, max_features)
        
        
        
        with open(prov_path + f'predictions_with_indices_2_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_single.pkl', 'wb') as f:
            pickle.dump(predictions_with_indices, f)
        with open(prov_path + f'true_labels_2_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_single.pkl', 'wb') as f:
            pickle.dump(true_labels, f)
        with open(prov_path + f'predictions_proba_2_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_single.pkl', 'wb') as f:
            pickle.dump(predictions_proba, f)
        with open(prov_path + f'index_tracker_2_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_single.pkl', 'wb') as f:
            pickle.dump(index_tracker, f)
        # with open(prov_path + f'rf_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_ck.pkl', 'wb') as f:
        #     pickle.dump(rf, f)
        
        if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
            poison_indices = poison_indices_all
        
        indexes_to_exculde =  analyze_batch_sample_weights(clean_inds, sus_inds, poison_indices,predictions_with_indices, predictions_proba, index_tracker, dataset, global_seed, attack, eps, pr_sus, pr_tgt, threshold, penalty, figure_path)
        
        print("Time taken for analyzing:", time.time() - start_time)
        
        with open(prov_path + f'indexes_to_exculde_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}.pkl', 'wb') as f:
            pickle.dump(indexes_to_exculde, f)

        del sus_diff, clean_diff 
    
    if analyze:
        _, _, _, target_class_indices = get_loaders_from_dataset(poisoned_train_dataset, test_dataset, poisoned_test_dataset, bs_bl, target_class)  
        poison_amount = len(poison_indices)
        ignore_set = set(poison_indices)
        
        random_sus_idx = get_random_poison_idx(pr_sus, ignore_set, poison_indices, target_class_indices, poison_amount, global_seed)
        print("Suspected samples length:",len(random_sus_idx), "Poison ratio training set", pr_tgt, "Poison percentage suspected dataset", pr_sus)
        
        with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_single_batch.pkl','rb') as f: 
            sus_inds = pickle.load(f)
        # sus_inds = sus_inds[len(sus_inds)//6:]
        with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_single_batch.pkl','rb') as f:
            clean_inds = pickle.load(f)
        # clean_inds = clean_inds[len(clean_inds)//6:]

        with open(prov_path + f'predictions_with_indices_2_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_single.pkl', 'rb') as f:
            predictions_with_indices = pickle.load(f)
        with open(prov_path + f'true_labels_2_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_single.pkl', 'rb') as f:
            true_labels = pickle.load(f)
        with open(prov_path + f'predictions_proba_2_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_single.pkl', 'rb') as f:
            predictions_proba = pickle.load(f)
        with open(prov_path + f'index_tracker_2_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_single.pkl', 'rb') as f:
            index_tracker = pickle.load(f) 
        
        if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
            poison_indices = poison_indices_all
            
        indexes_to_exculde =  analyze_batch_sample_weights(clean_inds, sus_inds, poison_indices,predictions_with_indices, predictions_proba, index_tracker, dataset, global_seed, attack, eps, pr_sus, pr_tgt, threshold, penalty, figure_path)
        
        with open(prov_path + f'indexes_to_exculde_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}.pkl', 'wb') as f:
            pickle.dump(indexes_to_exculde, f)
        
            
    # if ext_rel_wts:
    #     _, _, _, target_class_indices = get_loaders_from_dataset(poisoned_train_dataset, test_dataset, poisoned_test_dataset, bs_bl, target_class)  
    #     poison_amount = len(poison_indices)
    #     ignore_set = set(poison_indices)
        
    #     random_sus_idx = get_random_poison_idx(pr_sus, ignore_set, poison_indices, target_class_indices, poison_amount, global_seed)
    #     print("Suspected samples length:",len(random_sus_idx), "Poison ratio training set", pr_tgt, "Poison percentage suspected dataset", pr_sus)
        
    #     with open(prov_path + f'sus_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}.pkl','rb') as f:
    #         sus_diff = pickle.load(f)
    #     with open(prov_path + f'clean_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}.pkl','rb') as f:
    #         clean_diff = pickle.load(f)
    #     with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}.pkl','rb') as f: 
    #         sus_inds = pickle.load(f)
    #     with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}.pkl','rb') as f:
    #         clean_inds = pickle.load(f)
    #     with open(prov_path + f'important_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}.pkl','rb') as f:
    #         important_features = pickle.load(f)

    #     relevant_features, rf, _,_,_,_ = get_relevant_weight_dimensions(sus_diff, clean_diff, sus_inds, clean_inds, poison_indices, random_sus_idx,pr_sus, global_seed, device, figure_path, min_features, max_features) 
    #     with open(prov_path + f'relevant_features_2_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}.pkl', 'wb') as f:
    #         pickle.dump(relevant_features, f)
    #     with open(prov_path + f'rf_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}.pkl', 'wb') as f:
    #         pickle.dump(rf, f)
        
    #     # with open(prov_path + f'sus_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_idv.pkl', 'wb') as f:
    #     #     pickle.dump(sus_diff[:,relevant_features], f)
    #     # with open(prov_path + f'clean_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_idv.pkl', 'wb') as f:
    #     #     pickle.dump(clean_diff[:,relevant_features], f)
    #     # with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_idv.pkl', 'wb') as f:
    #     #     pickle.dump(sus_inds, f)
    #     # with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_bsl_idv.pkl', 'wb') as f:
    #     #     pickle.dump(clean_inds, f) 

    #     del sus_diff, clean_diff
            
            
    if sample_level: 
        poisoned_train_loader, test_loader, poisoned_test_loader, target_class_indices = get_loaders_from_dataset(poisoned_train_dataset, test_dataset, poisoned_test_dataset, bs_sl, target_class)
        model = copy.deepcopy(orig_model)
        model.to(device)
        if opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif opt == "sgd":
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, dampening=0, nesterov=True) 
        else:
            raise ValueError("Optimizer not supported")
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep_sl) 
        
        model_path = saved_models_path + f'model_{attack}_{dataset}_{eps}_{pr_tgt}_{ep_sl_base}_{bs_bl}.pkl'
        optimizer_path = saved_models_path + f'optimizer_{attack}_{dataset}_{eps}_{pr_tgt}_{ep_sl_base}_{bs_bl}.pkl'
        scheduler_path = saved_models_path + f'scheduler_{attack}_{dataset}_{eps}_{pr_tgt}_{ep_sl_base}_{bs_bl}.pkl'
        if ep_sl_base > 0:
            if os.path.exists(model_path) and os.path.exists(optimizer_path) and os.path.exists(scheduler_path) and not force:
                model.load_state_dict(torch.load(model_path))
                optimizer.load_state_dict(torch.load(optimizer_path))
                scheduler.load_state_dict(torch.load(scheduler_path))
                print(f"Loaded model trained for {ep_sl_base} epochs from saved file")
            else:
                print(f"Training model for {ep_sl_base} epochs before capturing sample level weight updates...")
                model, optr, slr, train_ACC, test_ACC, clean_ACC, target_ACC = train(model,optimizer, opt, scheduler, criterion, poisoned_train_loader,test_loader, poisoned_test_loader, ep_sl_base, global_seed, device, training_mode)
                torch.save(model.state_dict(), model_path)
                torch.save(optr.state_dict(), optimizer_path)
                torch.save(slr.state_dict(), scheduler_path)
                print(f"Model trained for {ep_sl_base} epochs and saved")

        poison_amount = len(poison_indices)
        ignore_set = set(poison_indices)
        random_sus_idx = get_random_poison_idx(pr_sus, ignore_set, poison_indices, target_class_indices, poison_amount, global_seed)
        
        

        print("Suspected samples length:",len(random_sus_idx))
        with open(prov_path + f'important_features_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_k_1.pkl','rb') as f:
            important_features = pickle.load(f) 
        # with open(prov_path + f'important_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_check.pkl','rb') as f:
        #     important_features = pickle.load(f)
        print("Important features shape:", important_features.shape)
        # print("Important features shape:", important_features.shape)
        # with open("./Training_Prov_Data/" + f'relevant_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}.pkl','rb') as f:
        #     relevant_features = pickle.load(f) 
        # important_features = important_features[relevant_features]
        print("Suspected samples length:",len(random_sus_idx), "Poison ratio training set", pr_tgt, "Poison percentage suspected dataset", pr_sus)  
        
        # sus_diff, clean_diff, sus_inds, clean_inds =  capture_sample_level_weight_updates_2(random_sus_idx, poison_indices, model, orig_model, optimizer, scheduler,criterion, ep_sl, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class, important_features , attack, sample_from_test, device, global_seed)

        # sus_diff, clean_diff, sus_inds, clean_inds = capture_simple_sample_level_weight_updates(random_sus_idx, model, orig_model, optimizer, opt,  scheduler,criterion, ep_sl, lr ,poisoned_train_loader, test_loader, poisoned_test_loader, poisoned_train_dataset, test_dataset, important_features, target_class, sample_from_test, attack, device, global_seed, figure_path, training_mode, k=1)
        sus_diff, clean_diff, sus_inds, clean_inds = capture_sample_level_weight_updates_idv(random_sus_idx, model, orig_model, optimizer, opt,  scheduler,criterion, ep_sl, lr ,poisoned_train_loader, test_loader, poisoned_test_loader, important_features, target_class, sample_from_test, attack, device, global_seed, figure_path, training_mode, k=1)
        print("Shape of suspected weight updates:", sus_diff.shape, "Shape of clean weight updates:", clean_diff.shape)
        

        with open(prov_path + f'sus_diff_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'wb') as f:
            pickle.dump(sus_diff, f)
        with open(prov_path + f'clean_diff_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'wb') as f:
            pickle.dump(clean_diff, f)
        with open(prov_path + f'sus_inds_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'wb') as f:
            pickle.dump(sus_inds, f)
        with open(prov_path + f'clean_inds_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'wb') as f:
            pickle.dump(clean_inds, f)

        del sus_diff, clean_diff
    if score_samples:
        ignore_set = set(poison_indices)
        start_time = time.time()
        
        with open(prov_path + f'sus_diff_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
            sus_diff = pickle.load(f)
        with open(prov_path + f'clean_diff_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
            clean_diff = pickle.load(f)
        with open(prov_path + f'sus_inds_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
            sus_inds = pickle.load(f)
        with open(prov_path + f'clean_inds_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
            clean_inds = pickle.load(f)
        # ep_sl_base = 200
        # with open(prov_path + f'sus_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_200_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     sus_diff = pickle.load(f)
        # with open(prov_path + f'clean_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_200_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     clean_diff = pickle.load(f)
        # with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_200_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     sus_inds = pickle.load(f)
        # with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_200_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     clean_inds = pickle.load(f)
                                        
        # a = 1            
        # b = 2  
        
        # with open("./Training_Prov_Data/" + f'relevant_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}.pkl','rb') as f:
        #     relevant_features = pickle.load(f)  
        # sus_diff, clean_diff, sus_inds, clean_inds = sus_diff[:a*len(sus_diff)//b], clean_diff[:a*len(sus_diff)//b], sus_inds[:a*len(sus_diff)//b], clean_inds[:a*len(sus_diff)//b]
        # sus_diff, clean_diff, sus_inds, clean_inds = sus_diff[:, relevant_features], clean_diff[:, relevant_features], sus_inds, clean_inds
            
        # print("Shape of suspected weight updates:", sus_diff.shape, "Shape of clean weight updates:", clean_diff.shape, "Suspected indices:", np.array(sus_inds).shape, "Clean indices:", np.array(clean_inds).shape)
        # ep_sl = 15
        # # for i in range(2,3):
        # with open(prov_path + f'sus_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     sus_diff = np.concatenate([sus_diff, pickle.load(f)])
        # with open(prov_path + f'clean_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     clean_diff = np.concatenate([clean_diff, pickle.load(f)])
        # with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     sus_inds = np.concatenate([sus_inds, pickle.load(f)])
        # with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     clean_inds = np.concatenate([clean_inds, pickle.load(f)])
        # 
        # with open(prov_path + f'sus_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     sus_diff = np.concatenate([sus_diff, pickle.load(f)])
        # with open(prov_path + f'clean_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     clean_diff = np.concatenate([clean_diff, pickle.load(f)])
        # with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     sus_inds = np.concatenate([sus_inds, pickle.load(f)])
        # with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
        #     clean_inds = np.concatenate([clean_inds, pickle.load(f)])
                
        # with open(prov_path + f'rf_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_ck.pkl', 'rb') as f:
        #     rf_model = pickle.load(f) 
        

        if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
            poison_indices = poison_indices_all
        
        print("Shape of suspected weight updates:", sus_diff.shape, "Shape of clean weight updates:", clean_diff.shape, "Suspected indices:", np.array(sus_inds).shape, "Clean indices:", np.array(clean_inds).shape)
        poison_amount = len(poison_indices)     
        random_sus_idx = np.unique(sus_inds)
        random_clean_sus_idx = list(set(random_sus_idx) - set(poison_indices))
        print("poison indices:", len(poison_indices), "suspected indices:", len(random_sus_idx), "clean indices:", len(random_clean_sus_idx))
        assert set(random_clean_sus_idx) == set(random_sus_idx) - set(poison_indices)
        # get_iso_forest_threshold(sus_diff, clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, 5, ep_sl, global_seed, pr_tgt, pr_sus, attack, figure_path, threshold)
        # indexes_to_exculde = score_poisoned_samples_2(sus_diff,clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, 5, rf_model, ep_sl, global_seed, pr_tgt, pr_sus, eps, attack, figure_path, threshold)
        indexes_to_exculde = score_poisoned_samples(sus_diff,clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, groups, dataset, cv_model, ep_sl, global_seed, device, pr_tgt, pr_sus, attack, figure_path, threshold)
        print("Length of indexes to exclude:", len(indexes_to_exculde), "pos_indices excluded:", len(set(indexes_to_exculde) & set(poison_indices)))
        with open(prov_path + f'indexes_to_exculde_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}.pkl', 'wb') as f:
            pickle.dump(indexes_to_exculde, f)
            
        print("Time taken for analyzing:", time.time() - start_time)
    # if score_samples:
    #     ignore_set = set(poison_indices)
    #     with open(prov_path + f'sus_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #         sus_diff = pickle.load(f)
    #     with open(prov_path + f'clean_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #         clean_diff = pickle.load(f)
    #     with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #         sus_inds = pickle.load(f)
    #     with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #         clean_inds = pickle.load(f)
        
        
    #     # with open("./Training_Prov_Data/" + f'relevant_features_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}.pkl','rb') as f:
    #     #     relevant_features = pickle.load(f)  
    #     # # sus_diff, clean_diff, sus_inds, clean_inds = sus_diff[len(sus_diff)//4:], clean_diff[len(sus_diff)//4:], sus_inds[len(sus_diff)//4:], clean_inds[len(sus_diff)//4:]
    #     # sus_diff, clean_diff, sus_inds, clean_inds = sus_diff[:, relevant_features], clean_diff[:, relevant_features], sus_inds, clean_inds
            
    #     print("Shape of suspected weight updates:", sus_diff.shape, "Shape of clean weight updates:", clean_diff.shape, "Suspected indices:", np.array(sus_inds).shape, "Clean indices:", np.array(clean_inds).shape)
    #     # ep_sl = 15
    #     for ep_sl_base in [50, 100, 150]:
    #         with open(prov_path + f'sus_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #             sus_diff = np.concatenate([sus_diff, pickle.load(f)])
    #         with open(prov_path + f'clean_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #             clean_diff = np.concatenate([clean_diff, pickle.load(f)])
    #         with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #             sus_inds = np.concatenate([sus_inds, pickle.load(f)])
    #         with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_{ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #             clean_inds = np.concatenate([clean_inds, pickle.load(f)])
    #     # 
    #     # with open(prov_path + f'sus_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #     #     sus_diff = np.concatenate([sus_diff, pickle.load(f)])
    #     # with open(prov_path + f'clean_diff_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #     #     clean_diff = np.concatenate([clean_diff, pickle.load(f)])
    #     # with open(prov_path + f'sus_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #     #     sus_inds = np.concatenate([sus_inds, pickle.load(f)])
    #     # with open(prov_path + f'clean_inds_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_sl}_{ep_sl}_sample_level_idv_k_1_aug_z_1.pkl', 'rb') as f:
    #     #     clean_inds = np.concatenate([clean_inds, pickle.load(f)])
                
    #     # with open(prov_path + f'rf_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_ck.pkl', 'rb') as f:
    #     #     rf_model = pickle.load(f) 
    #     chunk_size = 50000  # Define chunk size
    #     print("Shape of suspected weight updates:", sus_diff.shape, "Shape of clean weight updates:", clean_diff.shape, "Suspected indices:", np.array(sus_inds).shape, "Clean indices:", np.array(clean_inds).shape)
    #     # Determine the number of chunks
    #     num_chunks = len(sus_diff) // chunk_size + (1 if len(sus_diff) % chunk_size != 0 else 0)

    #     for i in range(num_chunks):
    #         start_idx = i * chunk_size
    #         end_idx = min(start_idx + chunk_size, len(sus_diff))

    #         # Slice the data for the current chunk
    #         sus_diff_chunk = sus_diff[start_idx:end_idx]
    #         clean_diff_chunk = clean_diff[start_idx:end_idx]
    #         sus_inds_chunk = sus_inds[start_idx:end_idx]
    #         clean_inds_chunk = clean_inds[start_idx:end_idx]

    #         # Perform the operations for the current chunk
    #         print(f"Processing chunk {i + 1}/{num_chunks}")
    #         print("Shape of suspected weight updates:", sus_diff_chunk.shape,
    #             "Shape of clean weight updates:", clean_diff_chunk.shape,
    #             "Suspected indices:", np.array(sus_inds_chunk).shape,
    #             "Clean indices:", np.array(clean_inds_chunk).shape)
    
    #         print("Shape of suspected weight updates:", sus_diff_chunk.shape, "Shape of clean weight updates:", clean_diff_chunk.shape, "Suspected indices:", np.array(sus_inds_chunk).shape, "Clean indices:", np.array(clean_inds_chunk).shape)
    #         poison_amount = len(poison_indices)     
    #         random_sus_idx = np.unique(sus_inds)
    #         random_clean_sus_idx = list(set(random_sus_idx) - set(poison_indices))
    #         print("poison indices:", len(poison_indices), "suspected indices:", len(random_sus_idx), "clean indices:", len(random_clean_sus_idx))
    #         assert set(random_clean_sus_idx) == set(random_sus_idx) - set(poison_indices)
    #         # get_iso_forest_threshold(sus_diff, clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, 5, ep_sl, global_seed, pr_tgt, pr_sus, attack, figure_path, threshold)
    #         # indexes_to_exculde = score_poisoned_samples_2(sus_diff,clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, 5, rf_model, ep_sl, global_seed, pr_tgt, pr_sus, eps, attack, figure_path, threshold)
    #         indexes_to_exculde = score_poisoned_samples(sus_diff_chunk,clean_diff_chunk, clean_inds_chunk, sus_inds_chunk, poison_indices, random_clean_sus_idx, groups, dataset, cv_model, ep_sl, global_seed, device, pr_tgt, pr_sus, attack, num_chunks, figure_path, threshold)
    #         print("Length of indexes to exclude:", len(indexes_to_exculde), "pos_indices excluded:", len(set(indexes_to_exculde) & set(poison_indices)))
    #     # with open(prov_path + f'indexes_to_exculde_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}.pkl', 'wb') as f:
    #     #     pickle.dump(indexes_to_exculde, f)
           
    if retrain:
        with open(prov_path + f'indexes_to_exculde_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}.pkl', 'rb') as f:
            indexes_to_exculde = pickle.load(f)
            print("Length of indexes to exclude:", len(indexes_to_exculde), "pos_indices:", len(set(indexes_to_exculde) & set(poison_indices)))
            # print("Length of indexes to exclude:", len(indexes_to_exculde_1), "pos_indices:", len(set(indexes_to_exculde_1) & set(poison_indices)))
        # with open("./Training_Prov_Data/SleeperAgent/" + f'indexes_to_exculde_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}.pkl', 'rb') as f:
        #     indexes_to_exculde_2 = pickle.load(f)
        #     print("Length of indexes to exclude:", len(indexes_to_exculde_2), "pos_indices:", len(set(indexes_to_exculde_2) & set(poison_indices)))
        # indexes_to_exculde = set(indexes_to_exculde_1) | set(indexes_to_exculde_2)        
        poisoned_train_loader, test_loader, poisoned_test_loader, _ = get_loaders_from_dataset(poisoned_train_dataset, test_dataset, poisoned_test_dataset, bs, target_class, indexes_to_exculde)
        model = copy.deepcopy(orig_model) 
        model.to(device)
        
        if opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif opt == "sgd":
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, dampening=0, nesterov=True)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 
        
        if get_result:
            model.load_state_dict(torch.load(saved_models_path + f'retrained_model_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{epochs}.pkl'))
            evaluate_model(model, test_loader, poisoned_test_loader, criterion, device)
        else:
            model, optr, slr, train_ACC, test_ACC, clean_ACC, target_ACC = train(model,optimizer, opt, scheduler, criterion, poisoned_train_loader,test_loader, poisoned_test_loader, epochs, global_seed, device, training_mode)
            torch.save(model.state_dict(), saved_models_path + f'retrained_model_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{epochs}.pkl')
        
    #Hidden Trigger Attack  python3 weight_updates_prov_defense.py --attack ht --clean_model_path ./saved_models/htbd_art_model_200.pth --target_class 4 --source_class 3 --pr_tgt 0.5 --scenario fine_tuning --model CustomCNN --pr_sus  50  --sample_from_test --prov_path ./Training_Prov_Data/
    #Sleeper agent Attack  python3 weight_updates_prov_defense.py --poisoned_training --attack sa --target_class 1 --source_class 0 --prov_path ./Training_Prov_Data/ --pr_sus 50 --pr_tgt 0.5
    #python3 weight_updates_prov_defense.py --attack ht --dataset imagenet --clean_model_path ./saved_models/vit_tinyimagenet_100_10.pth --model ViT --scenario fine_tuning --poisoned_training  --bs 32 --epochs 10 --target_class 4 --source_class 3 --pr_tgt 0.5
    



    
if __name__ == '__main__':
    main()