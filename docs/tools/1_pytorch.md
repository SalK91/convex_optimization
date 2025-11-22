# PyTorch   

``` python
import torch
import numpy as np
```

------------------------------------------------------------------------

## 1. Creating Tensors

### `torch.zeros`

``` python
x_zeros = torch.zeros(2, 3)
```

### `torch.ones`

``` python
x_ones = torch.ones(2, 3)
```

### `torch.rand`

``` python
x_rand = torch.rand(2, 3)
```

------------------------------------------------------------------------

## 2. Tensor from a Python List

``` python
data = [[1,2,3],[4,5,6]]
x_from_list = torch.tensor(data)
```

------------------------------------------------------------------------

## 3. Tensor from NumPy Array

``` python
np_array = np.array([[1,2,3],[4,5,6]])
x_from_np = torch.from_numpy(np_array)
```

------------------------------------------------------------------------

## 4. Shape & Batch Dimension

``` python
images = torch.rand(32,3,64,64)
batch = images.shape[0]
sample = images.shape[1:]
```

------------------------------------------------------------------------

## 5. Dtype

``` python
torch.zeros(2,2,dtype=torch.float32)
```

------------------------------------------------------------------------

## 6. Reshaping

### Unsqueeze

``` python
x = torch.tensor([1,2,3,4])
x.unsqueeze(0)
```

### Squeeze

``` python
y = torch.rand(1,3,1,4)
y.squeeze()
```

------------------------------------------------------------------------

## 7. Slicing

``` python
x = torch.arange(12).reshape(3,4)
x[0]
x[:,1]
x[0:2,1:3]
```

------------------------------------------------------------------------

## 8. .item()

``` python
loss = torch.tensor(3.14)
loss.item()
```
