# Image registration framework
Python/NumPy/SciPy implementation of a registration framework offering various combinations of similarity
measure (cost function) with optimization method.

The framework includes a collection of common transformation models which can be composed, combined and 
extended with new transformations, within the /transforms directory

The optimizers seek the transformation parameters which minimize the cost function.

# Distance measures
**Alpha AMD similarity measure**
- This combines intensity and spatial information. If you use this, please cite: https://doi.org/10.1109/TIP.2019.2899947
- The Alpha AMD similarity measure typically exhibits few local optima in the transformation parameter search space, compared to other commonly used measures, which makes the registration substantially more robust/less sensitive to the starting position, when used in a local search framework.

**Mutual Information**
- Uses scikit-learn implementation of mutual information function.

**Mean Squared Error**
- Average of pixel-wise intensity differences

**Discriminative Local Derivative Patterns**
- Implementation of Jiang et al (ref to follow)

# Optimizers
**Gradient Descent**
- Follow gradient calculated by distance measure, with configurable learning rate, stopping conditions etc

**Adam**
- Optimize using Adam algorithm, with configurable parameters

**Grid search**
- Evaluate distance metric over a grid of transform parameters

**Scipy**
- Based on minimize function from scipy.optimize package. Uses numerical estimation of gradient, does not require analytic gradient to be available

# Features
- The framework/measure supports images, represented by numpy arrays, with additional (anisotropic) voxel-sizes.
- 2D and 3D registration (ND for affine transformations)
- Completely python/numpy/scipy based codebase. No C/C++/... code, which should facilitate portability and understandability.

# Example
Try out the provided sample script:
python3 register\_framework\_example.py ./test\_images/reference\_example.png ./test\_images/floating\_example.png

# Further details
**register.py**
- Controls the registration, including the following options:
- Multi-start (add multiple initial transforms to start optimization from different points)
- Pyramid levels (configure smoothing and downsampling factors to guide registration from coarse to fine)
- Parameter scaling (specify scaling factors to ensure all transform parameters are equivalently scaled)
- Sampling factor (randomly choose a subset of pixels to improve speed, currently only implemented in Alpha AMD distance measure)
- Masking (ignore pixels outside area of interest)
- Pixel-weighting (adjust the importance of different pixels, currently only implemented in Alpha AMD)
- Choice of optimizer
- Choice of distance measure
- Image resampling

**optimizers**
- Each optimizer implements a common interface specified in ./optimizers/README.md
- Some optimizers (adam, gd) require that the distance measure returns gradient values

**distances**
- Distance measures are stored in ./distances/
- No common interface has currently been implemented. Registration class treats each one individually

# License
The registration framework is licensed under the permissive MIT license.

# Author/Copyright
Framework was originally written by (and copyright reserved for) Johan Ofverstedt. Extended by Jo Gay.
