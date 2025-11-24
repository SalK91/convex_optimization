# Understanding Device Management in PyTorch: A Practical Guide

When working with tensors and neural networks in PyTorch, understanding device management is essential. Every tensor and model parameter lives on a *device*---either the CPU or an accelerator like a GPU. PyTorch does not automatically move your data or model between devices. If your tensors and model are not on the same device, your program may crash with errors such as:

> RuntimeError: Expected all tensors to be on the same device...

In this article, you'll learn how to properly manage devices, avoid common beginner errors, and build reliable training loops.
----

## Why Devices Matter

### CPU

-   Default device in PyTorch.
-   Handles general-purpose computations.
-   Runs operations sequentially.
-   Slower for large deep learning workloads.

### GPU (CUDA)

-   Parallel accelerator.
-   Can train models 10--15× faster than CPU.
-   Essential for scaling deep learning.
-   Must explicitly move data and models to this device.

## Checking for GPU Availability

PyTorch provides a simple API to verify whether a GPU is available:

``` python
torch.cuda.is_available()
```

A common pattern for selecting a device is:

``` python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

-   `"cuda"` → NVIDIA GPU.
-   `"cpu"` → fallback if no GPU\
-   Other options exist (e.g., `"mps"` for Apple Silicon), but CUDA is still the standard in most PyTorch workflows.
 
## Moving Models and Data to a Device

Once you choose a device, you must manually move:

1.  Your model
2.  Your input tensors
3.  Your target labels

### Move the model:

``` python
model = MyModel().to(device)
```

### Move the batch data inside the training loop:

``` python
inputs = inputs.to(device)
targets = targets.to(device)
```

### Check where something lives

``` python
tensor.device
next(model.parameters()).device
```
 
## Important: `.to()` Does NOT Modify in Place

A very common mistake is:

``` python
inputs.to(device)  # WRONG → result is discarded
```

You must assign it:

``` python
inputs = inputs.to(device)
```
 
## Training Loop With Proper Device Management

``` python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
optimizer = optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

for inputs, targets in dataloader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
```

## Avoiding GPU Memory Errors

Even when everything is set correctly, you might encounter:

>RuntimeError: CUDA out of memory

This occurs when your model or batch size requires more memory than the
GPU has.

### How to fix it

-   Lower your batch size first (the most common fix)
-   Reduce image resolution\
-   Use `torch.cuda.empty_cache()` if necessary\
-   Try gradient accumulation

For many systems: - Batch size 32--64 is a good starting point
 