# Building a Custom PyTorch Dataset for Oxford Flowers

Real-world datasets rarely come neatly packaged like MNIST or CIFAR. The Oxford Flowers dataset is a perfect example of messy data:

-   Images are stored in a flat folder: `image0001.jpg`,
    `image0002.jpg`, ...
-   Labels are stored separately in a MATLAB `.mat` file
-   Labels start at 1, but PyTorch expects 0-based labels
-   There's no built-in PyTorch dataset for it

To make this dataset usable for training, we need to build a custom PyTorch `Dataset` class.

A PyTorch `Dataset` only needs to answer three questions:

1.  `__init__` --- How do I set up the dataset?
2.  `__len__` --- How many samples are there?
3.  `__getitem__` --- Given an index `i`, what image and label
    should I return?
--

## Key Concepts

### ✔ Light setup (`__init__`)

-   Store image folder path
-   Load labels from `.mat`
-   Fix label indexing from 1--102 to 0--101
-   But do not load images here

### ✔ Lazy loading (`__getitem__`)

-   Load only the image you need
-   Convert it with PIL
-   Return image + label

### ✔ Simple length (`__len__`)

-   Just return number of samples (labels)


# Example Implementation: `OxfordFlowersDataset`

``` python
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as sio   # pip install scipy


class OxfordFlowersDataset(Dataset):
    def __init__(self, root_dir, labels_mat_file, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        mat = sio.loadmat(labels_mat_file)
        raw_labels = mat["labels"].squeeze()
        self.labels = raw_labels.astype("int64") - 1
        self.num_samples = len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        filename = f"image{idx + 1:04d}.jpg"
        img_path = self.root_dir / filename

        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
```

------------------------------------------------------------------------

# Using the Dataset with a DataLoader

``` python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = OxfordFlowersDataset(
    root_dir="path/to/jpg",
    labels_mat_file="path/to/labels.mat",
    transform=transform,
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

images, labels = next(iter(dataloader))

print("Batch image shape:", images.shape)
print("Batch labels:", labels[:10])
```
