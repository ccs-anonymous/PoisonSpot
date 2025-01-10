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
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from captum.attr import IntegratedGradients

import torchvision.models as models
from PIL import Image
from tqdm.notebook import tqdm

from utils.util import *

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR 
import pickle 
import random 
from models.resnet import ResNet


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)



def get_narcissus_cifar10_poisoned_data(poison_ratio,target_class = 2,  datasets_root_dir='../../datasets/', model = ResNet(18),eps=16, global_seed=545, multi_test=3, gpu_id=0):
    CUDA_VISIBLE_DEVICES = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)
    
    
    # Generate Narcissus trigger function
    def narcissus_gen(dataset_path , lab, eps ):
        #Noise size, default is full image size
        noise_size = 32

        #Radius of the L-inf ball
        l_inf_r = eps/255

        model = ResNet(18, num_classes=201)
        #Model for generating surrogate model and trigger
        surrogate_model = model.to(device)
        generating_model = model.to(device)
        #Surrogate model training epochs
        surrogate_epochs = 200

        #Learning rate for poison-warm-up
        generating_lr_warmup = 0.1
        warmup_round = 5

        #Learning rate for trigger generating
        generating_lr_tri = 0.01      
        gen_round = 1000

        #Training batch size
        train_batch_size = 128

        #The model for adding the noise
        patch_mode = 'add'

        #The argumention use for surrogate model training stage
        transform_surrogate_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        #The argumention use for all training set
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        #The argumention use for all testing set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        ori_train = torchvision.datasets.CIFAR10(root=datasets_root_dir, train=True, download=False, transform=transform_train)
        ori_test = torchvision.datasets.CIFAR10(root=datasets_root_dir, train=False, download=False, transform=transform_test)
        outter_trainset = torchvision.datasets.ImageFolder(root=datasets_root_dir + '/tiny-imagenet-200/train/', transform=transform_surrogate_train)

        #Outter train dataset
        train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
        test_label = [get_labels(ori_test)[x] for x in range(len(get_labels(ori_test)))] 

        #Inner train dataset
        train_target_list = list(np.where(np.array(train_label)==lab)[0])
        train_target = Subset(ori_train,train_target_list)

        concoct_train_dataset = concoct_dataset(train_target,outter_trainset)

        surrogate_loader = torch.utils.data.DataLoader(concoct_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16)

        poi_warm_up_loader = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16)

        trigger_gen_loaders = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16)


        # Batch_grad
        condition = True
        noise = torch.zeros((1, 3, noise_size, noise_size), device=device)


        surrogate_model = surrogate_model
        criterion = torch.nn.CrossEntropyLoss()
        # outer_opt = torch.optim.RAdam(params=base_model.parameters(), lr=generating_lr_outer)
        surrogate_opt = torch.optim.SGD(params=surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(surrogate_opt, T_max=surrogate_epochs)

        #Training the surrogate model
        print('Training the surrogate model')
        for epoch in range(0, surrogate_epochs):
            surrogate_model.train()
            loss_list = []
            for images, labels in surrogate_loader:
                images, labels = images.to(device), labels.to(device)
                surrogate_opt.zero_grad()
                outputs = surrogate_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                loss_list.append(float(loss.data))
                surrogate_opt.step()
            surrogate_scheduler.step()
            ave_loss = np.average(np.array(loss_list))
            print('Epoch:%d, Loss: %.03f' % (epoch, ave_loss))
        #Save the surrogate model
        save_path = datasets_root_dir + './checkpoint_surrogate_pretrain_' + str(surrogate_epochs) +'.pth'
        torch.save(surrogate_model.state_dict(),save_path)

        #Prepare models and optimizers for poi_warm_up training
        poi_warm_up_model = generating_model
        poi_warm_up_model.load_state_dict(surrogate_model.state_dict())

        poi_warm_up_opt = torch.optim.RAdam(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)

        #Poi_warm_up stage
        poi_warm_up_model.train()
        for param in poi_warm_up_model.parameters():
            param.requires_grad = True

        #Training the surrogate model
        for epoch in range(0, warmup_round):
            poi_warm_up_model.train()
            loss_list = []
            for images, labels in poi_warm_up_loader:
                images, labels = images.to(device), labels.to(device)
                poi_warm_up_model.zero_grad()
                poi_warm_up_opt.zero_grad()
                outputs = poi_warm_up_model(images)
                loss = criterion(outputs, labels)
                loss.backward(retain_graph = True)
                loss_list.append(float(loss.data))
                poi_warm_up_opt.step()
            ave_loss = np.average(np.array(loss_list))
            print('Epoch:%d, Loss: %e' % (epoch, ave_loss))

        #Trigger generating stage
        for param in poi_warm_up_model.parameters():
            param.requires_grad = False

        batch_pert = torch.autograd.Variable(noise.to(device), requires_grad=True)
        batch_opt = torch.optim.RAdam(params=[batch_pert],lr=generating_lr_tri)
        for minmin in tqdm(range(gen_round)):
            loss_list = []
            for images, labels in trigger_gen_loaders:
                images, labels = images.to(device), labels.to(device)
                new_images = torch.clone(images)
                clamp_batch_pert = torch.clamp(batch_pert,-l_inf_r*2,l_inf_r*2)
                new_images = torch.clamp(apply_noise_patch(clamp_batch_pert,new_images.clone(),mode=patch_mode),-1,1)
                per_logits = poi_warm_up_model.forward(new_images)
                loss = criterion(per_logits, labels)
                loss_regu = torch.mean(loss)
                batch_opt.zero_grad()
                loss_list.append(float(loss_regu.data))
                loss_regu.backward(retain_graph = True)
                batch_opt.step()
            ave_loss = np.average(np.array(loss_list))
            ave_grad = np.sum(abs(batch_pert.grad).detach().cpu().numpy())
            print('Gradient:',ave_grad,'Loss:', ave_loss)
            if ave_grad == 0:
                break

        noise = torch.clamp(batch_pert,-l_inf_r*2,l_inf_r*2)
        best_noise = noise.clone().detach().cpu()
        plt.imshow(np.transpose(noise[0].detach().cpu(),(1,2,0)))
        plt.show()
        print('Noise max val:',noise.max())

        return best_noise
    
    trigger_dir = f'attacks/Narcissus/narcissus_trigger_{target_class}_{eps}.pkl'
    if not osp.exists(trigger_dir):
        print('Generating Narcissus trigger')
        narcissus_trigger = narcissus_gen(datasets_root_dir, target_class, eps)
        with open(trigger_dir, 'wb') as f:
            pickle.dump(narcissus_trigger, f)
    else:
        print('Loading Narcissus trigger')
        with open(trigger_dir, 'rb') as f:
            narcissus_trigger = pickle.load(f)
            
    

    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    poi_ori_train = torchvision.datasets.CIFAR10(root=datasets_root_dir, train=True, download=True, transform=transform_tensor)
    poi_ori_test = torchvision.datasets.CIFAR10(root=datasets_root_dir, train=False, download=True, transform=transform_tensor)
    transform_after_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),
    ]) 

    train_label = [get_labels(poi_ori_train)[x] for x in range(len(get_labels(poi_ori_train)))]
    test_label = [get_labels(poi_ori_test)[x] for x in range(len(get_labels(poi_ori_test)))]

    #Inner train dataset
    train_target_list = list(np.where(np.array(train_label)==target_class)[0])
    train_target = Subset(poi_ori_train,train_target_list) 
    
    # Poison training 
    poison_amount = int(len(train_target_list) * poison_ratio)
    random_poison_idx = random.sample(train_target_list, poison_amount)
    poison_train_target = poison_image(poi_ori_train,random_poison_idx,narcissus_trigger.cpu(),transform_after_train)
    print('Traing dataset size is:',len(poison_train_target)," Poison numbers is:",len(random_poison_idx))
    # assert set(random_poison_idx) & set(random_clean_idx) == set()

    best_noise = narcissus_trigger
    test_non_target = list(np.where(np.array(test_label)!=target_class)[0])
    test_non_target_change_image_label = poison_image_label(poi_ori_test,test_non_target,best_noise.cpu()*multi_test,target_class,None)
    print('Poison test dataset size is:',len(test_non_target_change_image_label))


    return poison_train_target, poi_ori_test, test_non_target_change_image_label, random_poison_idx
