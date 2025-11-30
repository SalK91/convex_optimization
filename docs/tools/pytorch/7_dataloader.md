# PyTorch Data Splitting & DataLoader: Building a Reliable Training Pipeline

In the previous article, we transformed and prepared your images for model consumption. Now, it's time to tackle *two critical steps* in every machine learning workflow:

- Splitting  dataset into  training ,  validation , and  test  sets  
- Using  DataLoader  to efficiently serve your data in  batches 

These steps may feel like simple logistics, but they directly affect *model performance, fairness, and reliability*.


# Why Not Train on the Whole Dataset?

If more data is better, why don’t we train using  all  flower images?

Because training accuracy tells you only *how well your model memorizes*.  
It does  not  tell you how well it  generalizes  to new images.

That’s why we split the data:

| Split | Purpose |
|-------|---------|
|  Training set  | Used to teach the model |
|  Validation set  | Used during training to tune hyperparameters & detect overfitting |
|  Test set  | Used once at the very end to report final unbiased performance |

This split ensures your model learns  patterns , not  memorizes answers .


# Splitting the Dataset in PyTorch

Use `random_split()` to ensure a  random and balanced  distribution of flower types.

Example for Oxford Flowers (8,189 images):

```python
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)
```

This ensures:
- No rounding errors  
- All images are used exactly once  

> Your  original dataset remains unchanged  — you’re only creating *views*.


# DataLoader: Efficiently Serving Data in Batches

A `Dataset` gives access to  one sample at a time .  
A `DataLoader` groups data into  batches , making training much faster on GPU.

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=32, shuffle=False)
```


# Why Shuffle Only in Training?

| Split | shuffle? | Why? |
|--------|----------|------|
| Training | ✔ Yes | Mixes classes to avoid bias & forgetting |
| Validation | ❌ No | Does not affect evaluation |
| Test | ❌ No | Ensures deterministic, repeatable results |

Shuffling prevents your model from mistakenly learning:

> "Daisies always come first, then roses."

Instead, it learns image  patterns , not  order .


# Inspecting Batches

A single batch contains:

- A tensor of 32 images → `(32, 3, 224, 224)`
- A tensor of 32 labels → `(32,)`

Verify:

```python
images, labels = next(iter(train_loader))
print(images.shape)   # torch.Size([32, 3, 224, 224])
print(labels.shape)   # torch.Size([32])
```



# Batches, Remainders & Epochs Explained

If you have 5,732 training images and a batch size of 32:

```
5732 / 32 = 179 full batches + 1 partial batch (4 images)
```

So an  epoch  = 180 batches.

PyTorch still uses that last small batch — that’s  normal and expected .



# Common Mistake: Reloading Data in `__getitem__`

```python
def __getitem__(self, idx):
    df = pd.read_csv("labels.csv")  # ❌ Very expensive!
```

This reloads the whole file  thousands of times per epoch .

✔ Instead: Load once in `__init__`

```python
def __init__(self, csv_path):
    self.labels = pd.read_csv(csv_path)  # loaded once
```

This change alone can speed training up by  10× or more .


# If You Get CUDA Out of Memory Errors…

Try reducing batch size before anything else:

```python
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
```

Then gradually increase until you find the optimal value for your GPU.



# Complete Data Pipeline Example

```python
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

for images, labels in train_loader:
    print(images.shape, labels.shape)
    break
```

