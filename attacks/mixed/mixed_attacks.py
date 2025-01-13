'''
This is the test code of poisoned training under LabelConsistent.
'''

import sys
sys.path.append('/home/philemon/Documents/PoisonSpot/')

import os
import os.path as osp
from art.utils import load_cifar10


import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, Normalize, RandomCrop
from torchvision.datasets import DatasetFolder
from attacks.Labelconsistent.label_consistent_attack import LabelConsistent

from models.resnet import ResNet
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10
import pickle
from torch.utils.data import DataLoader, Subset
import random
from utils.util import *
from tqdm.notebook import tqdm
from PIL import Image
from numpy import asarray
from skimage.transform import resize


CUDA_VISIBLE_DEVICES = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
datasets_root_dir = '../../datasets'
global_seed = 545
deterministic = True
torch.manual_seed(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)
from utils.pgd import PGD



os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)


def get_lc_narcissus_cifar_10_poisoned_data(poison_ratio,target_class = 2,  datasets_root_dir='../../datasets/', model = ResNet(18), clean_model_path='./saved_models/resnet18_200_clean.pth',vis = 255, global_seed=545, gpu_id=0):
    
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)
    
    dataset = torchvision.datasets.CIFAR10

    transform_train = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        RandomCrop(32, padding=4),  
        RandomHorizontalFlip(),
    ])
    trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

    transform_test = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

    adv_model = ResNet(18)
    adv_ckpt = torch.load("./saved_models/resnet18_200_clean.pth")
    adv_model.load_state_dict(adv_ckpt)

    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-1, -1] = 255
    pattern[-1, -3] = 255
    pattern[-3, -1] = 255
    pattern[-2, -2] = 255

    pattern[0, -1] = 255
    pattern[1, -2] = 255
    pattern[2, -3] = 255
    pattern[2, -1] = 255

    pattern[0, 0] = 255
    pattern[1, 1] = 255
    pattern[2, 2] = 255
    pattern[2, 0] = 255

    pattern[-1, 0] = 255
    pattern[-1, 2] = 255
    pattern[-2, 1] = 255
    pattern[-3, 0] = 255

    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[:3,:3] = 1.0
    weight[:3,-3:] = 1.0
    weight[-3:,:3] = 1.0
    weight[-3:,-3:] = 1.0


    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'benign_training': False, # Train Attacked Model
        'batch_size': 128,
        'num_workers': 8,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 200,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'ResNet-18_CIFAR-10_LabelConsistent'
    }

    eps = 8
    alpha = 1.5
    steps = 100
    max_pixel = 255

    patch_size = 3
    vis = 255

    label_consistent = LabelConsistent(
        train_dataset=trainset,
        test_dataset=testset,
        model= ResNet(18),
        adv_model=adv_model,
        adv_dataset_dir= datasets_root_dir + f'CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poison_ratio}_seed{global_seed}_patch_size{patch_size}',
        loss=nn.CrossEntropyLoss(),
        y_target=target_class,
        poisoned_rate=poison_ratio,
        pattern=pattern,
        weight=weight,
        eps=eps,
        alpha=alpha,
        steps=steps,
        max_pixel=max_pixel,
        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        poisoned_target_transform_index=0,
        schedule=schedule,
        seed=global_seed,
        deterministic=True
    )
    
    eps = 16
    trigger_dir = f'attacks/Narcissus/narcissus_trigger_{target_class}_{eps}.pkl'
    print('Loading Narcissus trigger')
    with open(trigger_dir, 'rb') as f:
        narcissus_trigger = pickle.load(f)
            
    poison_ratio = 0.005
    multi_test = 3

    # transform_tensor = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # poi_ori_train = torchvision.datasets.CIFAR10(root=datasets_root_dir, train=True, download=True, transform=transform_tensor)
    # poi_ori_test = torchvision.datasets.CIFAR10(root=datasets_root_dir, train=False, download=True, transform=transform_tensor)

    poi_ori_train = label_consistent.poisoned_train_dataset
    poi_ori_test = label_consistent.test_dataset[0]


    train_label = np.array([label for image, label, _ in poi_ori_train])
    test_label = np.array([label for image, label in poi_ori_test])
    indices = np.array([index for _, _, index in poi_ori_train])
    poison_indices = np.array(list(label_consistent.poisoned_train_dataset.poisoned_set))

    

    #Inner train dataset
    train_target_list = list(np.where(train_label==target_class)[0])
    train_target = Subset(poi_ori_train,train_target_list) 
    poison_indices = [i for i, ind in enumerate(indices) if ind in poison_indices]
    free_target_list = list(set(train_target_list) - set(poison_indices))
    

    # Poison training 
    poison_amount = int(len(train_target_list) * poison_ratio)
    random_poison_idx = random.sample(free_target_list, poison_amount)
    lc_poison_indices = poison_indices
    # poison_indices = np.concatenate((poison_indices,random_poison_idx))
    narcissus_poison_indices = random_poison_idx

    poison_train_target = poison_image(poi_ori_train,random_poison_idx,narcissus_trigger.cpu(), None)
    print('Traing dataset size is:',len(poison_train_target)," Poison numbers is:",len(random_poison_idx))

    best_noise = narcissus_trigger
    test_non_target = list(np.where(np.array(test_label)!=target_class)[0])
    test_non_target_change_image_label = poison_image_label(poi_ori_test,test_non_target,best_noise.cpu()*multi_test,target_class,None)
    print('Poison test dataset size is:',len(test_non_target_change_image_label))
    
    poison_test_dataset = {"LabelConsistent":label_consistent.poisoned_test_dataset, "Narcissus":test_non_target_change_image_label}
    poison_indices = {"LabelConsistent":lc_poison_indices, "Narcissus":narcissus_poison_indices}
    
    
    return poison_train_target,poi_ori_test, poison_test_dataset, poison_indices


def get_lc_narcissus_sa_cifar_10_poisoned_data(poison_ratio,target_class = 2,  datasets_root_dir='../../datasets/', model = ResNet(18), clean_model_path='./saved_models/resnet18_200_clean.pth',vis = 255, global_seed=545, gpu_id=0):
    
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)
    
    class_target = target_class
    class_source = 0 
    indices_poison = np.load(datasets_root_dir + f'indices_poison_resnet18_sa_{class_target}_{class_source}_16_0.1_128.npy')     
    x_poison = np.load(datasets_root_dir + f'x_poison_resnet18_sa_{class_target}_{class_source}_16_0.1_128.npy')   
    y_poison = np.load(datasets_root_dir + f'y_poison_resnet18_sa_{class_target}_{class_source}_16_0.1_128.npy') 
    
    dataset = torchvision.datasets.CIFAR10
    
    poison_index = np.where(y_poison.argmax(axis=1) == class_target)[0][indices_poison]
    y_poison[poison_index] = np.eye(10)[class_source]
    

    from torch.utils.data import Dataset, TensorDataset
    dataset = torchvision.datasets.CIFAR10
    transform_train = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # RandomCrop(32, padding=4),  
        RandomHorizontalFlip(),
    ])


    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None, target_transform=None):
            """
            Args:
                data (numpy array or torch tensor): Array of data (e.g., images).
                targets (numpy array or torch tensor): Array of targets (e.g., labels).
                transform (callable, optional): Optional transform to be applied on a sample.
            """
            self.data = data
            self.targets = targets
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            """Returns the total number of samples in the dataset."""
            return len(self.data)

        def __getitem__(self, idx):
            """
            Args:
                idx (int): Index of the sample to retrieve.
            
            Returns:
                tuple: (image, label) where image is the data at index `idx` and label is the corresponding target.
            """
            img = self.data[idx]
            label = self.targets[idx]

            if self.transform:
                img = self.transform(img)
                
            if self.target_transform:
                label = self.target_transform(label)

            return img, label
        
    # trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
    trainset = CustomDataset(x_poison.transpose(0, 2, 3, 1), y_poison.argmax(axis=1), transform=transform_train)

    transform_test = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

    adv_model = ResNet(18)
    adv_ckpt = torch.load("saved_models/resnet18_200_clean.pth")
    adv_model.load_state_dict(adv_ckpt)

    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-1, -1] = 255
    pattern[-1, -3] = 255
    pattern[-3, -1] = 255
    pattern[-2, -2] = 255

    pattern[0, -1] = 255
    pattern[1, -2] = 255
    pattern[2, -3] = 255
    pattern[2, -1] = 255

    pattern[0, 0] = 255
    pattern[1, 1] = 255
    pattern[2, 2] = 255
    pattern[2, 0] = 255

    pattern[-1, 0] = 255
    pattern[-1, 2] = 255
    pattern[-2, 1] = 255
    pattern[-3, 0] = 255

    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[:3,:3] = 1.0
    weight[:3,-3:] = 1.0
    weight[-3:,:3] = 1.0
    weight[-3:,-3:] = 1.0


    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'benign_training': False, # Train Attacked Model
        'batch_size': 128,
        'num_workers': 8,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 200,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'ResNet-18_CIFAR-10_LabelConsistent'
    }

    eps = 8
    alpha = 1.5
    steps = 100
    max_pixel = 255
    poisoned_rate = poison_ratio/(1-poison_ratio)
    patch_size = 3
    vis = 255

    label_consistent = LabelConsistent(
        train_dataset=trainset,
        test_dataset=testset,
        model= ResNet(18),
        adv_model=adv_model,
        adv_dataset_dir=f'datasets/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poison_ratio:.2f}_seed{global_seed}_mixed',
        loss=nn.CrossEntropyLoss(),
        y_target=2,
        poisoned_rate=poisoned_rate,
        pattern=pattern,
        weight=weight,
        eps=eps,
        alpha=alpha,
        steps=steps,
        max_pixel=max_pixel,
        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        poisoned_target_transform_index=0,
        schedule=schedule,
        seed=global_seed,
        deterministic=True
    )
    
    y_poison[poison_index] = np.eye(10)[class_target]
    
    eps = 16
    trigger_dir = f'attacks/Narcissus/narcissus_trigger_{target_class}_{eps}.pkl'
    print('Loading Narcissus trigger')
    with open(trigger_dir, 'rb') as f:
        narcissus_trigger = pickle.load(f)
            
    poison_ratio = 0.005
    multi_test = 3

    poi_ori_train = label_consistent.poisoned_train_dataset
    poi_ori_test = label_consistent.test_dataset[0]



    # Create TensorDataset
    train_images = torch.stack([images for images, _, _ in poi_ori_train])  # Stack images into a single tensor
    train_label = torch.tensor([labels for _, labels, _ in poi_ori_train])  # Convert labels to a tensor
    indices = np.array([index for _, _, index in poi_ori_train]) # Convert indices to a tensor
    test_label = [labels for images, labels in poi_ori_test]
    train_label = torch.tensor([label if indices[i] not in poison_index else target_class for i, label in enumerate(train_label)])
    lc_poison_indices = np.array(list(label_consistent.poisoned_train_dataset.poisoned_set))
    poi_ori_train = CustomDataset(train_images, train_label)

    #Inner train dataset
    train_target_list = list(np.where(np.array(train_label)==target_class)[0])
    train_target = Subset(poi_ori_train,train_target_list) 
    # indices_poison =  np.where(y_poison.argmax(axis=1) == class_target)[0][indices_poison]
    lc_poison_indices = [i for i, ind in enumerate(indices) if ind in lc_poison_indices]
    sa_poison_indices = [i for i, ind in enumerate(indices) if ind in poison_index]
    print('SA poison size:',len(sa_poison_indices), 'LC poison size:',len(lc_poison_indices), "train target size:",len(train_target_list))
    free_target_list = list(set(train_target_list) - set(sa_poison_indices) - set(lc_poison_indices))
    print('Free target size:',len(free_target_list))

    # Poison training 
    poison_amount = int(len(train_target_list) * poison_ratio)
    random_poison_idx = random.sample(free_target_list, poison_amount)
    # poison_indices = np.concatenate((poison_indices,random_poison_idx))
    narcissus_poison_indices = random_poison_idx

    poison_train_target = poison_image(poi_ori_train,random_poison_idx,narcissus_trigger.cpu(), None)
    print('Traing dataset size is:',len(poison_train_target)," Poison numbers is:",len(random_poison_idx))

    best_noise = narcissus_trigger
    test_non_target = list(np.where(np.array(test_label)!=target_class)[0])
    test_non_target_change_image_label = poison_image_label(poi_ori_test,test_non_target,best_noise.cpu()*multi_test,target_class,None)
    print('Poison test dataset size is:',len(test_non_target_change_image_label))  
    
    x_test = np.array([images for images, _ in poi_ori_test])
    y_test = np.array([labels for _, labels in poi_ori_test])
    test_dataset = CustomDataset(x_test.transpose(0, 2, 3, 1), y_test, transform=transform_test)

    
    # #sa
    # class_target = target_class
    # class_source = 0 
    # indices_poison = np.load(datasets_root_dir + f'indices_poison_resnet18_sa_{class_target}_{class_source}_16_{poison_ratio}_128.npy')     
    # x_poison = np.load(datasets_root_dir + f'x_poison_resnet18_sa_{class_target}_{class_source}_16_{poison_ratio}_128.npy')   
    # y_poison = np.load(datasets_root_dir + f'y_poison_resnet18_sa_{class_target}_{class_source}_16_{poison_ratio}_128.npy') 
    
    # x_poison_tensor = torch.tensor(x_poison) 
    # for i in range(x_poison.shape[0]):
    #     x_poison_tensor[i] = transform_train(x_poison[i].transpose(1, 2, 0))   
        
    
   
    # class CustomDataset(Dataset):
    #     def __init__(self, images, labels, indices):
    #         self.images = images
    #         self.labels = labels
    #         self.indices = indices

    #     def __len__(self):
    #         return len(self.images)

    #     def __getitem__(self, idx):
    #         image = self.images[idx]
    #         label = self.labels[idx]
    #         index = self.indices[idx]
    #         return image, label, index
    # index_source_train = np.where(y_poison.argmax(axis=1)==class_target)[0]
    # indices_poison = index_source_train[indices_poison]
    
    # # poisoned_dataset_sa = CustomDataset(x_poison_tensor[indices_poison], y_poison[indices_poison].argmax(axis=1), np.arange(len(poison_train_target), len(poison_train_target) +  len(indices_poison)))
    # # poison_train_target = ConcatDataset([poison_train_target, poisoned_dataset_sa])
    
    # # Step 1: Extract data from the existing dataset (poison_train_target)
    # existing_images = np.array([image for image, _, _ in poison_train_target])
    # existing_images = torch.from_numpy(existing_images)
    # existing_labels = np.array([label for _, label, _ in poison_train_target])
    # existing_labels = torch.from_numpy(existing_labels)
    # existing_indices = np.array([index for _, _, index in poison_train_target])

    # # Step 2: Extract data from the new dataset (poisoned_dataset_sa)
    # new_images = x_poison_tensor[indices_poison]
    # new_labels = y_poison[indices_poison].argmax(axis=1)
    # new_labels = torch.from_numpy(new_labels)
    # new_indices = np.arange(len(poison_train_target), len(poison_train_target) + len(indices_poison))

    # # Step 3: Manually concatenate the data
    # combined_images = torch.cat((existing_images, new_images), dim=0)
    # print(combined_images.shape)
    # combined_labels = torch.cat((existing_labels, new_labels))
    # combined_indices = np.concatenate([existing_indices, new_indices])

    # # Step 4: Create a new CustomDataset with the combined data
    # poison_train_target = CustomDataset(combined_images, combined_labels, combined_indices)

    
    patch_size = 8
    def add_trigger_patch(x_set,patch_type="random"):
        img = Image.open('attacks/Sleeperagent/trigger_10.png')
        numpydata = asarray(img)
        patch = resize(numpydata, (patch_size,patch_size,3))
        patch = np.transpose(patch,(2,0,1))
        if patch_type == "fixed":
            x_set[:,:,-patch_size:,-patch_size:] = patch
        else:
            for x in x_set:
                x_cord = random.randrange(0,x.shape[1] - patch.shape[1] + 1)
                y_cord = random.randrange(0,x.shape[2] - patch.shape[2] + 1)
                x[:,x_cord:x_cord+patch_size,y_cord:y_cord+patch_size]=patch

        return x_set

    (x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()
    # x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32) 

    # x_test_transformed = x_test.copy()
    # for i in range(x_test.shape[0]):
    #     x_test_transformed[i] = transform_test(x_test[i].transpose(1, 2, 0)).numpy()

    index_source_test = np.where(y_test.argmax(axis=1)==class_source)[0]
    x_test_trigger = x_test[index_source_test]
    x_test_trigger = add_trigger_patch(x_test_trigger,"fixed")
    x_test_trigger = x_test_trigger.transpose(0, 2, 3, 1)
    y_test_trigger = np.ones(len(x_test_trigger))*class_target
    y_test_trigger = y_test_trigger.astype(np.int64)    

    # x_test_trigger_tensor = torch.tensor(x_test_trigger, dtype=torch.float32)
    # y_test_trigger_tensor = torch.tensor(y_test_trigger, dtype=torch.int64)

    # Create the TensorDataset
    poisoned_test_dataset_sa = CustomDataset(x_test_trigger, y_test_trigger, transform=transform_test)
    
    
    # print('Training dataset size is:',len(poison_train_target)," Poison numbers is:",len(random_poison_idx))

    # best_noise = narcissus_trigger
    # test_non_target = list(np.where(np.array(test_label)!=target_class)[0])
    # test_non_target_change_image_label = poison_image_label(poi_ori_test,test_non_target,best_noise.cpu()*multi_test,target_class,None)
    # print('Poison test dataset size is:',len(test_non_target_change_image_label))
    
    poison_test_dataset = {"LabelConsistent":label_consistent.poisoned_test_dataset, "Narcissus":test_non_target_change_image_label , "SleeperAgent":poisoned_test_dataset_sa}
    poison_indices = {"LabelConsistent":lc_poison_indices, "Narcissus":narcissus_poison_indices, "SleeperAgent": sa_poison_indices}
    
    
    return poison_train_target,poi_ori_test, poison_test_dataset, poison_indices