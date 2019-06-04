import time
from datetime import timedelta

import os
import numpy as np
import pandas as pd
from skimage import io
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, csv_name, num_triplets, transform=None):

        self.root_dir = root_dir

        # self.df                = pd.read_csv(csv_name)
        self.num_triplets = num_triplets
        self.transform = transform

        self.num_classes = 0

        start_time = time.time()

        print('traversing file system')
        self.load()
        elapsed = time.time() - start_time
        print("File system traversal: %s secs" % timedelta(seconds=round(elapsed)))
        self.cursor = np.random.choice(self.num_classes)

    def load(self):
        self.face_classes = dict()
        for path, subdirs, files in os.walk(self.root_dir):
            for name in files:
                label = os.path.basename(path)
                if label not in self.face_classes:
                    self.face_classes[label] = []
                self.face_classes[label].append(name)

        self.num_classes = len(self.face_classes)

    def __getitem__(self, idx):

        list_classes = list(self.face_classes)

        pos_class_index = (self.cursor + idx) % self.num_classes
        pos_class = list_classes[ pos_class_index ]
        poslist = self.face_classes[pos_class]
        while len(poslist) < 2:
            self.cursor = (self.cursor + 1) % self.num_classes
            pos_class_index = (self.cursor + idx) % self.num_classes
            pos_class = list_classes[pos_class_index]
            poslist = self.face_classes[pos_class]

        neg_class_index = np.random.choice(self.num_classes)
        neg_class = list_classes[neg_class_index]
        while neg_class == pos_class:
            neg_class_index = np.random.choice(self.num_classes)
            neg_class = list_classes[neg_class_index]


        pos_index = np.random.choice(len(poslist))
        anc_index = np.random.choice(len(poslist))
        while pos_index == anc_index:
            anc_index = np.random.choice(len(poslist))

        neglist = self.face_classes[neg_class]
        neg_index = np.random.choice(len(neglist))

        pos_img = os.path.join(self.root_dir, pos_class + '/' + str(self.face_classes[pos_class][pos_index]))
        anc_img = os.path.join(self.root_dir, pos_class + '/' + str(self.face_classes[pos_class][anc_index]))
        neg_img = os.path.join(self.root_dir, neg_class + '/' + str(self.face_classes[neg_class][neg_index]))

        anc_img = io.imread(anc_img)
        pos_img = io.imread(pos_img)
        neg_img = io.imread(neg_img)

        pos_class = torch.from_numpy(np.array([pos_class_index]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class_index]).astype('long'))

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class,
                  'neg_class': neg_class}

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):

        return self.num_triplets


def get_dataloader(train_root_dir, valid_root_dir,
                   train_csv_name, valid_csv_name,
                   num_train_triplets, num_valid_triplets,
                   batch_size, num_workers):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])]),
        'valid': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])])}

    face_dataset = {
        'train': TripletFaceDataset(root_dir=train_root_dir,
                                    csv_name=train_csv_name,
                                    num_triplets=num_train_triplets,
                                    transform=data_transforms['train']),
        'valid': TripletFaceDataset(root_dir=valid_root_dir,
                                    csv_name=valid_csv_name,
                                    num_triplets=num_valid_triplets,
                                    transform=data_transforms['valid'])}

    dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset[x], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for x in ['train', 'valid']}

    data_size = {x: len(face_dataset[x]) for x in ['train', 'valid']}

    return dataloaders, data_size