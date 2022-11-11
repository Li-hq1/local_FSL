# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import torch 
import pickle
import numpy as np
from PIL import Image

from datasets.transform import PretrainTransform, build_finetune_transform

# ------------------------dataset---------------------------- #
ROOT_DIR = "data/datasets"

class VisionDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        pass

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)


class MiniImagenetDataset(VisionDataset):
    def __init__(self, root_path, split, transform):
        if split == "train":
            split_tag = "train_phase_train"
        elif split == "val":
            split_tag = "train_phase_val"
        elif split == "test":
            split_tag = "train_phase_test"
        elif split == "meta_val":
            split_tag = "val"
        elif split == "meta_test":
            split_tag = "test"
        else:
            assert False, "Dataset: 'split' name is wrong"
        split_file = "miniImageNet_category_split_{}.pickle".format(split_tag)
        with open(os.path.join(root_path, split_file), "rb") as f:
            pack = pickle.load(f, encoding="latin1")

        data = pack["data"]
        label = pack["labels"]
        data = [Image.fromarray(x) for x in data]
        min_label = min(label)
        label = [x - min_label for x in label]
        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]


class Cifar100Dataset(VisionDataset):
    def __init__(self, root_path, name, split, transform):
        if split == "train":
            split_tag = "train"
        elif split == "meta_val":
            split_tag = "val"
        elif split == "meta_test":
            split_tag = "test"
        else:
            assert False, "Dataset: 'split' name is wrong"
        assert name == "CIFAR_FS" or name == "FC100"
        split_file = name + "_{}.pickle".format(split_tag)
        with open(os.path.join(root_path, split_file), "rb") as f:
            pack = pickle.load(f, encoding="latin1")

        data = pack["data"]
        labels = pack["labels"]

        cur_class = 0
        label2label = {}
        for idx, label in enumerate(labels):
            if label not in label2label:
                label2label[label] = cur_class
                cur_class += 1
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])
        data = [Image.fromarray(x) for x in data]

        self.data = data
        self.label = new_labels

        self.n_classes = len(set(self.label))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]


class CubDataset(VisionDataset):
    def __init__(self, root_path, name, split, transform):
        if split == "train":
            split_tag = "train"
        elif split == "meta_val":
            split_tag = "val"
        elif split == "meta_test":
            split_tag = "test"
        else:
            assert False, "Dataset: 'split' name is wrong"
        assert name == "CUB"

        IMAGE_PATH = root_path
        SPLIT_PATH = os.path.join(root_path, "split/")
        txt_path = os.path.join(SPLIT_PATH, split_tag + ".csv")

        lines = [x.strip() for x in open(txt_path, "r").readlines()][1:]

        if split_tag == "train":
            lines.pop(5864)  # this image file is broken

        data = []
        labels = []
        lb = -1

        self.wnids = []

        for l in lines:
            context = l.split(",")
            name = context[0]
            wnid = context[1]
            path = os.path.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            labels.append(lb)

        cur_class = 0
        label2label = {}
        for idx, label in enumerate(labels):
            if label not in label2label:
                label2label[label] = cur_class
                cur_class += 1
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])

        data = [Image.open(path).convert("RGB") for path in data]

        self.data = data
        self.label = new_labels

        self.n_classes = len(set(self.label))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]


class TieredDataset(VisionDataset):
    def __init__(self, root_path, name, split, transform):
        if split == "train":
            split_tag = "train"
        elif split == "meta_val":
            split_tag = "val"
        elif split == "meta_test":
            split_tag = "test"
        else:
            assert False, "Dataset: 'split' name is wrong"
        assert name == "tiered"

        THE_PATH = os.path.join(root_path, split_tag)

        data = []
        labels = []

        folders = [
            os.path.join(THE_PATH, label)
            for label in os.listdir(THE_PATH)
            if os.path.isdir(os.path.join(THE_PATH, label))
        ]
        folders.sort()

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            this_folder_images.sort()
            for image_path in this_folder_images:
                data.append(os.path.join(this_folder, image_path))
                labels.append(idx)

        cur_class = 0
        label2label = {}
        for idx, label in enumerate(labels):
            if label not in label2label:
                label2label[label] = cur_class
                cur_class += 1
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])

        self.data = data
        self.label = new_labels

        self.n_classes = len(set(self.label))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(Image.open(self.data[i]).convert("RGB")), self.label[i]


# sampler for few-shot learning
class CategoriesSampler:
    def __init__(self, labels, n_iteration, n_class, n_sample, n_episode=1):
        self.n_iteration = n_iteration
        self.n_class = n_class
        self.n_sample = n_sample
        self.n_episode = n_episode

        labels = np.array(labels)
        self.catlocs = []
        for c in range(max(labels) + 1):
            self.catlocs.append(np.argwhere(labels == c).reshape(-1))

    def __len__(self):
        return self.n_iteration

    def __iter__(self):
        for i_batch in range(self.n_iteration):
            batch = []
            for i_ep in range(self.n_episode):
                episode = []
                classes = np.random.choice(
                    len(self.catlocs), self.n_class, replace=False
                )
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_sample, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)  # n_class * n_sample
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_class * n_sample
            yield batch.view(-1)


# -------------------------pretrain-------------------------- #
def build_pretraining_dataset(args):
    root_dir = ROOT_DIR
    transform = PretrainTransform(args)
    print("Data Aug = %s" % str(transform))

    dataset_name = args.dataset_name
    if dataset_name == "mini":
        root_path = os.path.join(root_dir, "mini-imagenet")
        return MiniImagenetDataset(root_path, split="train", transform=transform)
    elif dataset_name == "CUB":
        root_path = os.path.join(root_dir, "cub")
        name = "CUB"
        return CubDataset(root_path, name, split="train", transform=transform)
    elif dataset_name == "tiered":
        root_path = os.path.join(root_dir, "tiered_imagenet")
        name = "tiered"
        return TieredDataset(root_path, name, split="train", transform=transform)
    elif dataset_name == "CIFAR_FS":
        root_path = os.path.join(root_dir, "CIFAR_FS")
        name = "CIFAR_FS"
        return Cifar100Dataset(root_path, name, split="train", transform=transform)
    elif dataset_name == "FC100":
        root_path = os.path.join(root_dir, "FC100")
        name = "FC100"
        return Cifar100Dataset(root_path, name, split="train", transform=transform)
    else:
        assert False


# -------------------finetune----------------------- #
def build_finetune_dataset(is_train, dataset_name, split, args):
    root_dir = ROOT_DIR
    transform = build_finetune_transform(is_train, args)

    print("Transform = ")
    for t in transform.transforms:
        print(t)
    print("---------------------------")

    if dataset_name == "mini":
        root_path = os.path.join(root_dir, "mini-imagenet")
        dataset = MiniImagenetDataset(root_path, split, transform=transform)
    elif dataset_name == "CUB":
        root_path = os.path.join(root_dir, "cub")
        name = "CUB"
        dataset = CubDataset(root_path, name, split, transform=transform)
    elif dataset_name == "tiered":
        root_path = os.path.join(root_dir, "tiered_imagenet")
        name = "tiered"
        dataset = TieredDataset(root_path, name, split, transform=transform)
    elif dataset_name == "CIFAR_FS":
        root_path = os.path.join(root_dir, "CIFAR_FS")
        name = "CIFAR_FS"
        dataset = Cifar100Dataset(root_path, name, split, transform=transform)
    elif dataset_name == "FC100":
        root_path = os.path.join(root_dir, "FC100")
        name = "FC100"
        dataset = Cifar100Dataset(root_path, name, split, transform=transform)
    else:
        assert False
    n_classes = dataset.n_classes
    print("Number of the class = %d" % n_classes)

    return dataset, n_classes



