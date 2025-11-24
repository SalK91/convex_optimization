# PyTorch Data Utilities: Transforms, Datasets, and DataLoaders

Efficient data handling is a core part of every machine learning
workflow. PyTorch provides a clean, modular data pipeline built around three utilities: Transforms, Datasets, and DataLoaders. Together, they make it easy to prepare data, apply preprocessing, and feed batches to your model during training.

## 1. Transforms

Transforms perform preprocessing operations on input data. They are typically used to convert raw samples (images, text, audio, etc.) into tensors and normalize them for training.

### Common Uses
-   Convert images to tensors\
-   Normalize pixel values\
-   Data augmentation (flip, crop, rotate)\
-   Compose multiple preprocessing steps

### Underlying Idea
Transforms are callable objects. A transform takes one sample as
input and returns a modified sample.

``` python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

### Practitioner Tips

-   Use `transforms.Compose()` to chain steps.
-   Apply data augmentation only on the training set.
-   Keep normalization consistent with the pretrained models you use.
 
## 2. Datasets

A Dataset is a wrapper around your data. It tells PyTorch how to
access one sample at a time. Every custom dataset must implement two
methods:

-   `__len__` → returns dataset size\
-   `__getitem__` → returns one sample at index *i*

### Example: Custom Image Dataset

``` python
from torch.utils.data import Dataset
from PIL import Image

class MyImages(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
```

### Practitioner Tips

-   Use `torchvision.datasets` for standard datasets (CIFAR, MNIST,
    ImageNet).
-   Load files inside `__getitem__`, not beforehand.
-   Keep transforms inside the dataset to ensure consistent
    preprocessing.


## 3. DataLoader

A DataLoader wraps a Dataset and helps you iterate through the data efficiently by:

-   batching samples\
-   shuffling\
-   loading data in parallel with worker processes

### Example

``` python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for images, labels in train_loader:
    pass
```

### Underlying Idea

The DataLoader retrieves indices from the Dataset, loads samples,
applies batching, and returns them in iterable form.

### Practitioner Tips

-   Set `shuffle=True` for training; `False` for evaluation.
-   Increase `num_workers` to speed up loading (based on CPU cores).
-   Use `pin_memory=True` when training on GPU.
 
## Putting It All Together

``` python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset = MyImages(filepaths, labels, transform=transform)

loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

This structure cleanly separates: - how data is processed
(Transforms) - how data is accessed (Dataset) - how data is
delivered to the model (DataLoader)

------------------------------------------------------------------------

## Conclusion

PyTorch's data utilities provide a flexible and powerful way to prepare
data for deep learning workflows. Understanding these components enables
practitioners to build efficient, scalable training pipelines.
