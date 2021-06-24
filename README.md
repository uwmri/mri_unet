# MRI UNet

# Installation
This UNet requires PyTorch 1.8 or greater and can be installed using pip:
```
    pip install mri_unet 
```
Testing also requires torch-summary

# Usage
The model support 2D and 3D UNets with complex or real inputs. There are
also options to support complex kernels. For example, if I want a 3D UNet 
with complex kernels the code would be:

```python
    from mri_unet.unet import MRI_UNet
    model = MRI_UNet(in_channels=2, out_channels=2, complex_kernel=2, complex_input=True)
```

