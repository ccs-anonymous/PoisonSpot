import numpy as np
from art.utils import load_dataset
import sys
\
from models.resnet import ResNet

from art.estimators.classification import PyTorchClassifier
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.attacks.poisoning import HiddenTriggerBackdoor
from art.attacks.poisoning import perturbations
from torch.utils.data import TensorDataset
import random
import os
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def get_ht_cifar10_poisoned_data(poison_ratio, target_class, source_class,  model, dataset_path = './datasets/', clean_model_path = "./saved_models/htbd_art_model_200.pth",global_seed=545, gpu_id=0):
    
    CUDA_VISIBLE_DEVICES = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)
    
    (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')
    # Step 1a: Swap axes to PyTorch's NCHW format

    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
    mean = (0.4914, 0.4822, 0.4465) 
    std = (0.2023, 0.1994, 0.201)

    num_classes=10
    feature_size=4096

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, dampening=0, nesterov=True)
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_, max_),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        preprocessing=(mean, std)
    )



    if not os.path.exists(clean_model_path):
        "Started Clean Training"
        classifier.fit(x_train, y_train, nb_epochs=100, batch_size=512, verbose=True)
        for param_group in classifier.optimizer.param_groups:
            print(param_group["lr"])
            param_group["lr"] *= 0.1
        classifier.fit(x_train, y_train, nb_epochs=50, batch_size=512, verbose=True)
        for param_group in classifier.optimizer.param_groups:
            print(param_group["lr"])
            param_group["lr"] *= 0.1
        classifier.fit(x_train, y_train, nb_epochs=50, batch_size=512, verbose=True)
        torch.save(model.state_dict(), clean_model_path)
        print("Finished Clean Training")

    model.load_state_dict(torch.load(clean_model_path))

    def create_one_hot_vector(index, size=10):
        array = np.zeros(size, dtype=int)
        array[index] = 1
        return array

    target = create_one_hot_vector(target_class)
    source = create_one_hot_vector(source_class)

    patch_size = 8
    x_shift = 32 - patch_size - 5
    y_shift = 32 - patch_size - 5


    def mod(x):
        original_dtype = x.dtype
        x = perturbations.insert_image(x, backdoor_path="./attacks/HiddenTriggerBackdoor/htbd.png",
                                    channels_first=True, random=False, x_shift=x_shift, y_shift=y_shift,
                                    size=(patch_size,patch_size), mode='RGB', blend=1)
        return x.astype(original_dtype)
    backdoor = PoisoningAttackBackdoor(mod)


    poisoned_data_path = dataset_path + f"poison_data_htbd_art_model_{poison_ratio}_{source_class}_{target_class}.npy"
    poisoned_indices_path = dataset_path + f"poison_indices_htbd_art_model_{poison_ratio}_{source_class}_{target_class}.npy"

    if not os.path.exists(poisoned_data_path) or not os.path.exists(poisoned_indices_path):
        "Generate Poison Data"
        poison_attack = HiddenTriggerBackdoor(classifier, eps=16/255, target=target, source=source, feature_layer=19, backdoor=backdoor, decay_coeff = .95, decay_iter = 2000, max_iter=5000, batch_size=25, poison_percent=poison_ratio)

        poison_data, poison_indices = poison_attack.poison(x_train, y_train)
        print("Number of poison samples generated:", len(poison_data))
        np.save(poisoned_data_path, poison_data)
        np.save(poisoned_indices_path, poison_indices)
    poison_data = np.load(poisoned_data_path)
    poison_indices = np.load(poisoned_indices_path)

    poison_x = np.copy(x_train)
    poison_x[poison_indices] = poison_data

    poison_y = np.copy(y_train)
    all_indices = np.arange(len(poison_y))
    
    poison_y = poison_y.argmax(axis=1)
    
    
    poison_x_tensor = torch.tensor(poison_x)
    x_test_tensor = torch.tensor(x_test)
    
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    normalize = transforms.Normalize(mean=mean, std=std)
    x_poison_normalized = normalize(poison_x_tensor)
    x_test_normalized = normalize(x_test_tensor)

    poisoned_train_dataset = TensorDataset(x_poison_normalized, torch.tensor(poison_y), torch.tensor(all_indices))

    trigger_test_inds = np.where(np.all(y_test == source, axis=1))[0]
    test_poisoned_samples, _  = backdoor.poison(x_test[trigger_test_inds], y_test[trigger_test_inds])
    test_poisoned_samples = torch.tensor(test_poisoned_samples)
    test_poisoned_samples = normalize(test_poisoned_samples)
    
    test_poisoned_indices = np.array([target_class]*len(test_poisoned_samples))

    poisoned_test_dataset = TensorDataset(test_poisoned_samples, torch.tensor(test_poisoned_indices))

    y_test = y_test.argmax(axis=1)
    test_dataset = TensorDataset(x_test_normalized, torch.tensor(y_test))


    return poisoned_train_dataset,test_dataset, poisoned_test_dataset, poison_indices 





def get_ht_stl10_poisoned_data(poison_ratio, target_class, source_class,  model, dataset_path = './datasets/', clean_model_path = "./saved_models/htbd_art_model_200.pth",global_seed=545, gpu_id=0):
    
    CUDA_VISIBLE_DEVICES = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
        # transforms.RandomHorizontalFlip(),
    ])

    # Load the STL-10 dataset
    train_dataset = datasets.STL10(root=dataset_path, split='train', transform=transform, download=False)
    test_dataset = datasets.STL10(root=dataset_path, split='test', transform=transform, download=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Collect transformed data
    def collect_transformed_data(loader):
        transformed_data = []
        transformed_labels = []
        for data, labels in loader:
            transformed_data.append(data)
            transformed_labels.append(labels)
        return torch.cat(transformed_data), torch.cat(transformed_labels)

    x_train, y_train = collect_transformed_data(train_loader)
    x_test, y_test = collect_transformed_data(test_loader)
    x_train, y_train, x_test, y_test = x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy()
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    min_, max_ = x_train.min(), x_train.max()
    mean = (0.485, 0.456, 0.406) 
    std = (0.229, 0.224, 0.225)


    def create_one_hot_vector(index, size=10):
        array = np.zeros(size, dtype=int)
        array[index] = 1
        return array

    # Reconstruct the arrays
    target = create_one_hot_vector(target_class)
    source = create_one_hot_vector(source_class)

    patch_size = 8
    x_shift = 32 - patch_size - 5
    y_shift = 32 - patch_size - 5


    def mod(x):
        original_dtype = x.dtype
        x = perturbations.insert_image(x, backdoor_path="./attacks/HiddenTriggerBackdoor/htbd.png",
                                    channels_first=True, random=False, x_shift=x_shift, y_shift=y_shift,
                                    size=(patch_size,patch_size), mode='RGB', blend=1)
        return x.astype(original_dtype)
    backdoor = PoisoningAttackBackdoor(mod)


    poisoned_indices_path = dataset_path + f"poison_indices_htbd_stl10_vit_{poison_ratio}_{target_class}_{source_class}.npy"
    poisoned_data_path = dataset_path + f"poison_data_htbd_stl10_vit_{poison_ratio}_{target_class}_{source_class}.npy"
    
    if not os.path.exists(poisoned_data_path) or not os.path.exists(poisoned_indices_path):
        "Generate Poison Data"
        poison_attack = HiddenTriggerBackdoor(classifier, eps=16/255, target=target, source=source, feature_layer=19, backdoor=backdoor, decay_coeff = .95, decay_iter = 2000, max_iter=5000, batch_size=32, poison_percent=poison_ratio)

        poison_data, poison_indices = poison_attack.poison(x_train, y_train)
        print("Number of poison samples generated:", len(poison_data))
        np.save(poisoned_data_path, poison_data)
        np.save(poisoned_indices_path, poison_indices)
    
    
    
    poison_data = np.load(poisoned_data_path)
    poison_indices = np.load(poisoned_indices_path)
    
    poison_x = np.copy(x_train)
    poison_x[poison_indices] = poison_data

    poison_y = np.copy(y_train)
    all_indices = np.arange(len(poison_y))
    
    poison_y = poison_y.argmax(axis=1)
    
    
    poison_x_tensor = torch.tensor(poison_x)
    x_test_tensor = torch.tensor(x_test)
    
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    normalize = transforms.Normalize(mean=mean, std=std)
    x_poison_normalized = normalize(poison_x_tensor)
    x_test_normalized = normalize(x_test_tensor)

    poisoned_train_dataset = TensorDataset(x_poison_normalized, torch.tensor(poison_y), torch.tensor(all_indices))

    trigger_test_inds = np.where(np.all(y_test == source, axis=1))[0]
    test_poisoned_samples, _  = backdoor.poison(x_test[trigger_test_inds], y_test[trigger_test_inds])
    test_poisoned_samples = torch.tensor(test_poisoned_samples)
    test_poisoned_samples = normalize(test_poisoned_samples)
    
    test_poisoned_indices = np.array([target_class]*len(test_poisoned_samples))

    poisoned_test_dataset = TensorDataset(test_poisoned_samples, torch.tensor(test_poisoned_indices))

    y_test = y_test.argmax(axis=1)
    test_dataset = TensorDataset(x_test_normalized, torch.tensor(y_test))

    print("Poisoned Data Shape:", poison_x.shape)
    

    return poisoned_train_dataset,test_dataset, poisoned_test_dataset, poison_indices 



def get_ht_imagenet_poisoned_data(poison_ratio, target_class, source_class,  model, dataset_path = './datasets/', clean_model_path = "./saved_models/vit_tinyimagenet_100_10.pth",global_seed=545, gpu_id=0):
    
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    class TinyImageNetDataset(Dataset):
        def __init__(self, data_dir, transform=None, annotations_file=None, is_val=False, class_to_idx=None, num_classes=100, seed=42):
            """
            Custom dataset for Tiny ImageNet that uses annotations and can sample classes randomly.
            Args:
            - data_dir (str): Directory path to the dataset.
            - transform (callable, optional): Transform to be applied on a sample.
            - annotations_file (str, optional): Path to the annotations file for the validation set.
            - is_val (bool, optional): Whether the dataset being loaded is a validation set.
            - class_to_idx (dict, optional): Dictionary mapping class names to indices.
            - num_classes (int, optional): Number of classes to randomly sample.
            - seed (int, optional): Random seed for reproducibility.
            """
            self.data_dir = data_dir
            self.transform = transform
            self.images = []
            self.labels = []
            self.class_to_idx = class_to_idx
            random.seed(seed)

            if not is_val:
                random.seed(seed)
                all_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
                if num_classes and num_classes < len(all_classes):
                    selected_classes = random.sample(all_classes, num_classes)
                else:
                    selected_classes = all_classes
                
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_classes)}

                for class_dir in selected_classes:
                    images_folder = os.path.join(data_dir, class_dir, 'images')
                    for img_file in os.listdir(images_folder):
                        img_path = os.path.join(images_folder, img_file)
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_dir])
            else:
                # Load validation data using annotations
                if annotations_file:
                    with open(annotations_file, 'r') as file:
                        for line in file:
                            parts = line.strip().split('\t')
                            filename, class_id = parts[0], parts[1]
                            if class_id in self.class_to_idx:
                                img_path = os.path.join(data_dir, 'images', filename)
                                if os.path.exists(img_path):
                                    self.images.append(img_path)
                                    self.labels.append(self.class_to_idx[class_id])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label

        
    IMAGE_SIZE = 224
    MEAN_RGB = [0.485 , 0.456 , 0.406 ]
    STDDEV_RGB = [0.229 , 0.224 , 0.225]

    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB)
    ])


    data_dir = dataset_path + "tiny-imagenet-200/"
    val_annotations_file = dataset_path + "tiny-imagenet-200/val/val_annotations.txt"
    bs = 32
    num_classes = 100
    train_set = TinyImageNetDataset(data_dir + "train", transform=transform_train, num_classes=100, seed=42)
    val_set = TinyImageNetDataset(data_dir + "val", transform=transform_train, annotations_file=val_annotations_file, is_val=True, class_to_idx=train_set.class_to_idx)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)

    def collect_transformed_data(loader):
        transformed_data = []
        transformed_labels = []
        for data, labels in loader:
            transformed_data.append(data)
            transformed_labels.append(labels)
        return torch.cat(transformed_data), torch.cat(transformed_labels)

    x_train, y_train = collect_transformed_data(train_loader)
    x_test, y_test = collect_transformed_data(val_loader)
    x_train, y_train, x_test, y_test = x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy()
    y_train = np.eye(100)[y_train]
    y_test = np.eye(100)[y_test]
    min_, max_ = x_train.min(), x_train.max()
    
    from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
    target = np.zeros(100, dtype=int)
    target[target_class] = 1
    source = np.zeros(100, dtype=int)
    source[source_class] = 1

    patch_size = 30
    x_shift = 224 - patch_size - 5
    y_shift = 224 - patch_size - 5

    from art.attacks.poisoning import perturbations
    def mod(x):
        original_dtype = x.dtype
        x = perturbations.insert_image(x, backdoor_path="./attacks/HiddenTriggerBackdoor/htbd.png",
                                    channels_first=True, random=False, x_shift=x_shift, y_shift=y_shift,
                                    size=(patch_size,patch_size), mode='RGB', blend=1)
        return x.astype(original_dtype)
    backdoor = PoisoningAttackBackdoor(mod)

        
    with open(dataset_path + f'poison_indices_htbd_imgnet_vit_{poison_ratio}_{target_class}_{source_class}.npy', 'rb') as f:
        poison_indices = np.load(f)
    with open(dataset_path + f'poison_data_htbd_imgnet_vit_{poison_ratio}_{target_class}_{source_class}.npy', 'rb') as f:
        poison_data = np.load(f)
    
    class CustomDataset(Dataset):
        def __init__(self, images, labels, indices, transform=None):
            """
            Args:
                images (numpy.ndarray): Array of images.
                labels (numpy.ndarray): Array of labels corresponding to the images.
                indices (numpy.ndarray): Array of indices (optional, useful for tracking original order).
                transform (callable, optional): Optional transform to be applied on a sample.
            """
            self.images = images
            self.labels = labels
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            # Get the image, label, and index
            image = self.images[idx]
            label = self.labels[idx]
            index = self.indices[idx]
            
            # Apply transformation 
            if self.transform:
                image = self.transform(image)
            
            # Return the image, label, and index
            return image, label, index


    print("shape of poison data", poison_data.shape, "shape of poison indices", poison_indices.shape)
    poison_x = np.copy(x_train)
    poison_x[poison_indices] = poison_data

    poison_y = np.copy(y_train)
    
    
    all_indices = np.arange(len(poison_y))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        ])

    x_test_tensor = torch.tensor(x_test)

    poisoned_train_dataset = CustomDataset(poison_x.transpose(0, 2, 3, 1), poison_y.argmax(axis=1), all_indices, transform=transform)

    trigger_test_inds = np.where(y_test.argmax(axis=1) == source.argmax())[0]
    test_poisoned_samples, _  = backdoor.poison(x_test[trigger_test_inds], y_test[trigger_test_inds])
    test_poisoned_samples = torch.tensor(test_poisoned_samples)

    test_poisoned_labels = np.array([target_class]*len(test_poisoned_samples))
    
    poisoned_test_dataset = TensorDataset(test_poisoned_samples, torch.tensor(test_poisoned_labels))

    test_dataset = TensorDataset(x_test_tensor, torch.tensor(y_test.argmax(axis=1)))

    return poisoned_train_dataset,test_dataset, poisoned_test_dataset, poison_indices



