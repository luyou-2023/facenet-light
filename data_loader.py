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

        self.num_triplets = num_triplets
        self.transform = transform

        start_time = time.time()

        print('traversing file system at %s' % root_dir)
        self.load()
        elapsed = time.time() - start_time
        print("File system traversal: %s secs" % timedelta(seconds=round(elapsed)))

        self.set_rand_cursor()
        self.generate_triplets()

    def load(self):
        self.all_classes = dict()
        for path, subdirs, files in os.walk(self.root_dir):
            for name in files:
                label = os.path.basename(path)
                if label not in self.all_classes:
                    self.all_classes[label] = []
                self.all_classes[label].append(name)

        self.posclasslist = list()
        self.pos_classes = dict()
        for name in self.all_classes:
            if len(self.all_classes[name]) > 1:
                self.pos_classes[name] = self.all_classes[name]
                self.posclasslist.append(name)

        self.list_classes = list(self.all_classes)


    def generate_triplets(self):
        self.triplets = list()
        for i in range(self.num_triplets):
            triplet = self.create_triplet(i)
            self.triplets.append(triplet)

    def set_rand_cursor(self):
        self.cursor = np.random.choice(len(self.pos_classes))
        print("shuffling cursor for %s classes, new index: %s" % (len(self.pos_classes), str(self.cursor)))

    def advance_to_the_next_subset(self):
        self.cursor = (self.cursor + self.num_triplets) % len(self.pos_classes)
        print("advancing to the next subset for %s classes, new index: %s" % (len(self.pos_classes), str(self.cursor)))
        self.generate_triplets()

    def create_triplet(self, idx):

        len_pos = len(self.posclasslist)
        pos_class_index = (self.cursor + idx) % len_pos
        pos_class_name = self.posclasslist[ pos_class_index ]
        poslist = self.pos_classes[pos_class_name]

        num_all_classes = len(self.all_classes)
        neg_class_index = np.random.choice(num_all_classes)
        neg_class_name = self.list_classes[neg_class_index]
        while neg_class_name == pos_class_name:
            neg_class_index = np.random.choice(num_all_classes)
            neg_class_name = self.list_classes[neg_class_index]


        pos_index = np.random.choice(len(poslist))
        anc_index = np.random.choice(len(poslist))
        while pos_index == anc_index:
            anc_index = np.random.choice(len(poslist))

        neglist = self.all_classes[neg_class_name]
        neg_index = np.random.choice(len(neglist))

        pos_img = os.path.join(self.root_dir, pos_class_name + '/' + str(self.pos_classes[pos_class_name][pos_index]))
        anc_img = os.path.join(self.root_dir, pos_class_name + '/' + str(self.pos_classes[pos_class_name][anc_index]))
        neg_img = os.path.join(self.root_dir, neg_class_name + '/' + str(self.all_classes[neg_class_name][neg_index]))

        return [pos_img, anc_img, neg_img ]

    def __getitem__(self, idx):

        pos_img, anc_img, neg_img = self.triplets[idx]

        anc_img = io.imread(anc_img)
        pos_img = io.imread(pos_img)
        neg_img = io.imread(neg_img)

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img}

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