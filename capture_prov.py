from __future__ import absolute_import, division, print_function, unicode_literals
import argparse

import os, sys
import io
import pandas as pd
from torch.autograd import Variable




def parse_args():
    parser = argparse.ArgumentParser(description='PoisonSpot Defense')
    parser.add_argument("--batch_level", action='store_true', help="Enable batch level weight updates")
    parser.add_argument("--clean_training", action='store_true', help="Do clean training")
    parser.add_argument("--poisoned_training", action='store_true', help="Do poisoned training")
    parser.add_argument("--sample_level", action='store_true', help="Enable sample level weight updates")
    parser.add_argument("--score_samples", action='store_true', help="Enable scoring of suspected samples")
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
    parser.add_argument("--clean_model_path", type=str, default='./saved_models/resnet18_200_clean.pth', help="Path to the clean model")
    parser.add_argument("--saved_models_path", type=str, default='./saved_models/', help="Path to save the models")
    parser.add_argument("--global_seed", type=int, default=545, help="Global seed for the experiment")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for the experiment")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for the experiment")
    parser.add_argument("--figure_path", type=str, default="./results/", help="Path to save the figures")
    parser.add_argument("--prov_path", type=str, default="./Training_Prov_Data/", help="Path to save the provenance data")  
    parser.add_argument("--epochs", type=int, default = 200, help="Number of epochs for either clean or poisoned training")
    parser.add_argument("--scenario", type=str, default="from_scratch", help="Scenario to use for the experiment")
    parser.add_argument("--get_result", action='store_true', help="Get results from previous runs")
    parser.add_argument("--force", action='store_true', help="Force the run")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for scoring suspected samples")
    parser.add_argument("--sample_from_test", action='store_true', help="Sample from the test set")
    parser.add_argument("--cv_model", type=str, default="RandomForest", help="Model to use for cross validation")
    parser.add_argument("--groups", type=int, default=5, help="Number of groups to use for cross validation")
    parser.add_argument("--opt", type=str, default="sgd", help="Optimizer to use for the experiment")
    parser.add_argument("--training_mode",  action='store_false', help="Training mode for the model")
    return parser.parse_args()

args = parse_args()

import sys
print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__)) 

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

        

def train(model, optimizer, opt, scheduler, criterion, poisoned_train_loader,test_loader, poisoned_test_loader, training_epochs, global_seed, device, training_mode = True):
    
    np.random.seed(global_seed)
    random.seed(global_seed)
    torch.manual_seed(global_seed)
    
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



def capture_first_level_multi_epoch_batch_sample_weight_updates(random_sus_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []
    target_ACC = []

    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    random_sus_idx = set(random_sus_idx)
    
    dataset = poisoned_train_loader.dataset

    
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
                temp_sus = torch.cat([
                    (sur_model_state_dict[key] - original_weights[key]).view(-1)
                    for key in sur_model_state_dict
                ]).cpu().numpy()
                    
                
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



                sur_model.eval()  
                output = sur_model(clean_batch)
                sur_model.train(mode = training_mode)  
            
                    
                    
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
                sur_model_state_dict = sur_model.state_dict()
                temp_clean = torch.cat([
                    (sur_model_state_dict[key] - original_weights[key]).view(-1)
                    for key in sur_model_state_dict
                ]).cpu().numpy()



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
                

                step5_1_time = time.time() - start_time_2
                step5_1_time_avg += step5_1_time
                
                sur_model.train(mode = training_mode)    
                output = sur_model(clean_batch)
                
                    
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
                sur_model_state_dict = sur_model.state_dict()
                temp_clean_2 = torch.cat([
                        (sur_model_state_dict[key] - original_weights[key]).view(-1)
                        for key in sur_model_state_dict.keys()
                ]).cpu().numpy()

            
                
                step6_time = time.time() - start_time   
                step6_time_avg += step6_time 
                start_time = time.time()
                
                sus_params = np.maximum(sus_params, np.abs(temp_sus - temp_clean_2))
                clean_params = np.maximum(clean_params, np.abs(temp_clean - temp_clean_2))
                
                step7_time = time.time() - start_time
                step7_time_avg += step7_time
                step7_time_avg += step7_time
                
                del temp_sus, temp_clean, temp_clean_2
                

        
        
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




def capture_sample_level_weight_updates_idv(random_sus_idx, model, orig_model, optimizer, opt, scheduler,criterion, training_epochs, lr, poisoned_train_loader, test_loader, poisoned_test_loader, important_features, target_class,sample_from_test, attack, device, seed, figure_path, training_mode = True, k=1): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
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
            
            
            indices = indices.clone()
            labels  = labels.clone()
            indices = indices.clone()
            

            pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
            assert np.all(labels[pos_indices].cpu().numpy() == target_class)
            if len(pos_indices) > 0:
                step2_time = time.time() - start_time
                start_time = time.time()
                step2_time_avg += step2_time
                
                if not sample_from_test:
                    target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
                    available_indices = list(set(target_indices_batch) - set(pos_indices))
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


                combined_batch.extend(clean_batch_2)
                combined_labels.extend(clean_labels_2)

                combined_batch = torch.stack(combined_batch).to(device)
                combined_labels = torch.tensor(combined_labels).to(device)
     
                gen = torch.Generator().manual_seed(seed)

                combined_loader = DataLoader(
                    list(zip(combined_batch, combined_labels, combined_indexes)),
                    batch_size=1,
                    shuffle=False,  
                    generator=gen 
                )

                temp_sus = {}
                temp_clean = {}
                temp_clean_2 = np.zeros(len(important_features))  
                batch_count_clean_2 = 0

                
                for image, label, (index, tag) in combined_loader:
                    torch_rng_state = torch.get_rng_state()
                    cuda_rng_state = torch.cuda.get_rng_state()
                    np_rng_state = np.random.get_state()
                    python_rng_state = random.getstate()
    
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
        
        
        # Testing 
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



# Function to train different models and get feature importances
def train_prov_data_custom(X_sus, X_clean, clean_igs_inds, sus_igs_inds, random_poison_idx, random_clean_sus_idx, n_groups, seed, device, model_name='RandomForest', verbose=True, training_mode=True, max_iters=1, confidence_threshold=0.7):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    y_sus = np.ones(len(X_sus))
    y_clean = np.zeros(len(X_clean))

    X = np.concatenate([X_clean, X_sus])
    del X_sus, X_clean
    y = np.concatenate([y_clean, y_sus])
    assert not np.isinf(X).any()
    del y_clean, y_sus

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

                    true_pos = set(pos_inds_train) & set(high_conf_indices)

                    if verbose:
                        print(f"Iteration {iteration}: {len(high_conf_indices)} high-conf , true pos {len(true_pos)}, total pos {len(pos_inds_train)} total sus {len(X_sus_temp)}")

                    if len(high_conf_indices) == 0:
                        break

                    X_labeled = np.concatenate([X_labeled, X_sus_temp[high_conf_indices]])                                                                                                      
                    y_labeled = np.concatenate([y_labeled, np.ones(len(high_conf_indices))])

                    X_train_temp = np.concatenate([X_labeled, X_clean_temp[:len(X_labeled)*m]])
                    y_train_temp = np.concatenate([y_labeled, (len(X_train_temp) - len(X_labeled)) * [0]])                 

                    X_sus_temp = np.delete(X_sus_temp, high_conf_indices, axis=0)  
                    if len(X_sus_temp) == 0:
                        break
                    remaining_indices = set(range(len(X_sus_temp))) - set(high_conf_indices)
                    pos_inds_train = [i for i, ind in enumerate(remaining_indices) if pos_inds[ind] in random_poison_idx]
                    pos_inds  = np.delete(pos_inds, high_conf_indices , axis=0)
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

        predictions.extend(y_pred)
        true_labels.extend(y_test)
        predictions_proba.extend(y_pred_proba)
        index_tracker.extend(test_indices)

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
 
def score_poisoned_samples(sus_diff,clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, n_groups, dataset, cv_model, epochs, seed, device, poison_ratio, percentage, attack, figure_path = "./figures/", threshold=0.6):
    predictions_with_indices, average_feature_importances, true_labels, predictions_proba, _, index_tracker = train_prov_data_custom(
        sus_diff, clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, n_groups, seed, device, model_name=cv_model)
    prov_path = "./Training_Prov_Data/"
    real_clean_indices = np.unique(clean_inds)

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
        
        print("TPR gaussian threshold: ", np.mean(pos_scores > gaussian_threshold))
        print("TPR kmeans: ", np.mean(pos_scores > kmeans_threshold))
        
        if sus_scores is not None:
            print("FPR gaussian: ", np.mean(sus_scores > gaussian_threshold))
            print("FPR kmeans: ", np.mean(sus_scores > kmeans_threshold))
            pos_custom = np.mean(pos_scores > threshold)
            fpos_custom = np.mean(sus_scores > threshold)
            
            pos_gaussian = np.mean(np.concatenate([pos_scores , sus_scores]) > gaussian_threshold)
            fpos_gaussian = np.mean(clean_scores > gaussian_threshold)
            
            pos_kmeans = np.mean(np.concatenate([pos_scores , sus_scores]) > kmeans_threshold)
            fpos_kmeans = np.mean(clean_scores > kmeans_threshold)
        else:
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
            bbox_to_anchor=(0.5, 1.02),  
            ncol=3, 
            borderaxespad=0
        )
        # plt.axhline(y=threshold, color='black', linestyle='--')
        plt.xlabel("Number of samples")
        plt.ylabel("Poisoning Score")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  
        plt.savefig(figure_path + f"{title}.png")


        
        return threshold, kmeans_threshold, gaussian_threshold, pos_gaussian, fpos_gaussian, pos_kmeans, fpos_kmeans, pos_custom, fpos_custom


    def average_k_minimum_values(arr, k):
        nan_mask = np.isnan(arr)
        large_number = np.nanmax(arr[np.isfinite(arr)]) + 1
        arr_masked = np.where(nan_mask, large_number, arr)
        
        k_min_indices = np.argsort(arr_masked, axis=1)[:, :k]
        
        k_min_values = np.take_along_axis(arr, k_min_indices, axis=1)
        
        k_min_averages = np.mean(k_min_values, axis=1)
        
        return k_min_averages

    j = 50
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

    custom_threshold_mean,kmeans_threshold_mean, gaussian_threshold_mean, pos_gaussian_mean, fpos_gaussian_mean, pos_kmeans_mean, fpos_kmeans_mean, pos_custom_mean, fpos_custom_mean = plot_scores(pos_mean, clean_mean, sus_mean, f"SL Mean {attack} pr {poison_ratio} percentage {percentage} constant threshold")
    best_config = "mean_kmeans"
    
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
        }
        
        sus_prediction_indices = np.concatenate([pos_prediction_indices, sus_clean_prediction_indices])
        indexes_to_exclude = sus_prediction_indices[values[best_config]]
        return indexes_to_exclude
        
    else:
        values = {
        "mean_gaussian": pos_mean > gaussian_threshold_mean,
        "mean_kmeans": pos_mean > kmeans_threshold_mean,
        "mean_custom": pos_mean > custom_threshold_mean,
        }
        
        print("values: ", len(pos_mean[pos_mean > gaussian_threshold_mean]))
        indexes_to_exclude = pos_prediction_indices[values[best_config]]            
    return indexes_to_exclude


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
    
    batch_level = args.batch_level
    clean_training = args.clean_training
    poisoned_training = args.poisoned_training
    sample_level = args.sample_level
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
    get_result = args.get_result
    force = args.force
    threshold = args.threshold
    sample_from_test = args.sample_from_test
    eps = args.eps
    cv_model = args.cv_model
    vis = args.vis
    groups = args.groups
    opt = args.opt
    training_mode = args.training_mode

    
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
    
    
    if clean_training:
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
        
    if batch_level:
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


        
        important_features = capture_first_level_multi_epoch_batch_sample_weight_updates(random_sus_idx, model, orig_model, optimizer, opt, scheduler, criterion, ep_bl, lr, poisoned_train_loader, test_loader, poisoned_test_loader, target_class, sample_from_test, attack, device, global_seed, figure_path, training_mode, k=1)

            
        with open(prov_path + f'important_features_single_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}_{bs_bl}_k_1.pkl', 'wb') as f:
            pickle.dump(important_features, f)
        print("Important features shape:", important_features.shape)
            
            
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

        print("Important features shape:", important_features.shape)
        print("Suspected samples length:",len(random_sus_idx), "Poison ratio training set", pr_tgt, "Poison percentage suspected dataset", pr_sus)  
        

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
        

        if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
            poison_indices = poison_indices_all
        
        print("Shape of suspected weight updates:", sus_diff.shape, "Shape of clean weight updates:", clean_diff.shape, "Suspected indices:", np.array(sus_inds).shape, "Clean indices:", np.array(clean_inds).shape)
        poison_amount = len(poison_indices)     
        random_sus_idx = np.unique(sus_inds)
        random_clean_sus_idx = list(set(random_sus_idx) - set(poison_indices))
        print("poison indices:", len(poison_indices), "suspected indices:", len(random_sus_idx), "clean indices:", len(random_clean_sus_idx))
        assert set(random_clean_sus_idx) == set(random_sus_idx) - set(poison_indices)
        indexes_to_exculde = score_poisoned_samples(sus_diff,clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, groups, dataset, cv_model, ep_sl, global_seed, device, pr_tgt, pr_sus, attack, figure_path, threshold)
        print("Length of indexes to exclude:", len(indexes_to_exculde), "pos_indices excluded:", len(set(indexes_to_exculde) & set(poison_indices)))
        with open(prov_path + f'indexes_to_exculde_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}.pkl', 'wb') as f:
            pickle.dump(indexes_to_exculde, f)
            
        print("Time taken for analyzing:", time.time() - start_time)
           
    if retrain:
        with open(prov_path + f'indexes_to_exculde_{attack}_{dataset}_{eps}_{pr_tgt}_{pr_sus}.pkl', 'rb') as f:
            indexes_to_exculde = pickle.load(f)
            print("Length of indexes to exclude:", len(indexes_to_exculde), "pos_indices:", len(set(indexes_to_exculde) & set(poison_indices)))
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
        
    
if __name__ == '__main__':
    main()