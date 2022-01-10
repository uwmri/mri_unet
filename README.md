# MRI UNet

# Installation
This UNet requires PyTorch 1.10 or greater and can be installed using pip:
```
    pip install mri_unet 
```
Testing also requires torch-summary

# Usage
The model support 2D and 3D UNets with complex or real inputs. There are
also options to support complex kernels. For example, if I want a 3D UNet 
with complex kernels the code would be:

```python
    from mri_unet.unet import UNet
    
    # 2D, Complex Input, Real Kernels
    model = UNet(in_channels=2, out_channels=2, complex_input=True)

    # 3D, Complex Input, Real Kernels
    model = UNet(in_channels=2, out_channels=2, complex_input=True, ndims=3)
    
    # 2D, Complex Input, Complex Kernels
    model = UNet(in_channels=2, out_channels=2, complex_input=True, complex_kernel=True)
```

