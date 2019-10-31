# Image registration framework
This is a python/numpy implementation of an image registration framework, offering various combinations of similarity
measure (cost function) with optimization method.

The framework includes a collection of common transformation models which can be composed, combined and 
extended with new transformations, within the /transforms directory

The optimizers seek the transformation parameters which minimize the cost function.

## Available distance measures
**Alpha AMD similarity measure**
- This combines intensity and spatial information. If you use this, please cite: https://doi.org/10.1109/TIP.2019.2899947
- The Alpha AMD similarity measure typically exhibits few local optima in the transformation parameter search space, compared to other commonly used measures, which makes the registration substantially more robust/less sensitive to the starting position, when used in a local search framework.

**Mutual Information**
- Uses scikit-learn implementation of mutual information function.

**Mean Squared Error**
- Average of pixel-wise intensity differences

**Discriminative Local Derivative Patterns**
- Implementation of Jiang et al (ref to follow)

## Available optimizers
**Gradient Descent**
- Follow gradient calculated by distance measure, with configurable learning rate, stopping conditions etc

**Adam**
- Optimize using Adam algorithm, with configurable parameters

**Grid search**
- Evaluate distance metric over a grid of transform parameters

**Scipy**
- Based on minimize function from scipy.optimize package. Uses numerical estimation of gradient, does not require analytic gradient to be available

## Features
- The framework/measure supports images, represented by numpy arrays, with additional (anisotropic) voxel-sizes.
- 2D and 3D registration (ND for affine transformations)
- Completely python/numpy/scipy based codebase. No C/C++/... code, which should facilitate portability and understandability.

## Usage example
Try out the sample script provided:
- python3 register\_framework\_example.py ./test\_images/reference\_example.png ./test\_images/floating\_example.png

## Installation
Clone the project and then install the prerequisites e.g.

**pip**
- pip install -r requirements.txt

**conda**
- conda env create -f requirements.txt -n registerFramework

## Further details
This section contains more detailed information about the structure of the code and how to use the framework.

### 1. Controller (./register.py)
The Register object controls the registration, including the following options:
- Multi-start (add multiple initial transforms to start optimization from different points)
- Pyramid levels (configure smoothing and downsampling factors to guide registration from coarse to fine)
- Parameter scaling (specify scaling factors to ensure all transform parameters are equivalently scaled)
- Sampling factor (randomly choose a subset of pixels to improve speed, currently only implemented in Alpha AMD distance measure)
- Masking (ignore pixels outside area of interest)
- Pixel-weighting (adjust the importance of different pixels, currently only implemented in Alpha AMD)
- Choice of optimizer
- Choice of distance measure
- Image resampling
As shown in the example script, the Register object needs to be created, given the images to align as ndarrays, 
given at least one initial transform (starting point for the registration, probably identity transform), 
and (optionally, defaults are 'alphaAMD' and 'adam') given a model and optimization method to use.
The Register object should then be initialized (initialize(), sets up the pyramid levels etc) before 
calling the run() function to perform the registration process.

### 2. Optimizers (./optimizers/*.py)
Each optimizer implements a common interface dexcribed in ./optimizers/README.md

Some optimizers (adam, gd) require that the distance measure returns gradient values. Some distance measures do not do this.

Optimizers each have their own parameters such as number of iterations, gradient threshold, etc. These are selected
via the set_optimizer() method on the Register object. All examples below assume a 2D rigid registration,
which has 3 transform parameters: rotation, x-translation, y-translation.

#### gd: Gradient Descent (./optimizers/gd_optimizer.py)
- step_length: Scalar, usually between 0 and 1, default 1. The size of the step (often denoted gamma)
that will be taken in each iteration.
- end_step_length: Scalar, default None. If set, the step length reduces linearly between 
step_length and end_step_length
- gradient_magnitude_threshold: Scalar, default 0.0001. Stop optimization if the magnitude 
of the gradient is below this value
- param_scaling: List of scalars, default [1.0]*num_params. For each parameter in the transform,
provide a scaling factor in order to harmonize the scales of the parameters. For example, if you expect
a rotation of the order of 2 radians and a translation of the order of 100 px in each direction then 
supply [1./50,1.,1.] to treat one fiftieth of a radian rotation equivalently to one pixel translation
- iterations: Integer, default 5000. Continue optimization for this many iterations unless
gradient magnitude threshold is reached.

#### adam: Adaptive gradient descent (./optimziers/adam_optimizer.py)
An implementation of Kingma, D.P. and Ba, J., 2014. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_.
- step_length: Scalar, default 1. Initial learning rate (Adam's alpha parameter)
- end_step_length: Optional scalar, default None. If set, the step length reduces linearly between 
step_length and end_step_length
- beta1: Scalar, default 0.9. Adam's beta\_1 parameter, the exponential decay rate for 
the 1st moment estimates
- beta2: Scalar, default 0.999. Adam's beta\_2 parameter, the exponential decay rate for 
the 2nd moment estimates
- eps: Scalar, default 1e-8. A very small number to prevent division by zero.
- gradient_magnitude_threshold: Scalar, default 0.0001. Stop optimization if the magnitude 
of the gradient is below this value
- param_scaling: List of scalars, default [1.0]*num_params. For each parameter in the transform,
provide a scaling factor in order to harmonize the scales of the parameters. For example, if you expect
a rotation of the order of 2 radians and a translation of the order of 100 px in each direction then 
supply [1./50,1.,1.] to treat one fiftieth of a radian rotation equivalently to one pixel translation
- iterations: Integer, default 100. Continue optimization for this many iterations unless
gradient magnitude threshold is reached.

#### scipy: Wrapper for generic minimization function in scipy library (./optimizers/scipy_optimizer.py)
- method: String, default 'BFGS'. Which of the scipy optimization methods to use
- param_scaling: List of scalars, default [1.0]*num_params. For each parameter in the transform,
provide a scaling factor in order to harmonize the scales of the parameters. For example, if you expect
a rotation of the order of 2 radians and a translation of the order of 100 px in each direction then 
supply [1./50,1.,1.] to treat one fiftieth of a radian rotation equivalently to one pixel translation.
- minimizer_opts: Optional dictionary. Any options to be passed on to the scipy minimizer function
- reset_minimizer: Boolean, default False. Clear out any existing stored values in minimizer_opts before
updating with the new ones given.


#### gridsearch: Exhaustive search over a grid (./optimizers/grid_search_optimizer.py)
- bounds: List of lists. For each parameter applicable in the chosen type of transform, provide a list
giving the minimum and maximum values for that parameter (unscaled)
- steps: List of integers, or integer. Supply either the number of steps for each parameter, or a single
integer representing the number of steps for all parameters. Grid points will be spaced equally between
the lower bound and the upper bound, inclusive. E.g. if the bounds for a particular parameter are [-1.5, 1.5]
and steps=7, the grid points evaluated will be [-1.5, -1.0, -0.5, 0, 0.5, 1, 1.5].

### 3. Distance measures (./distances/*.py)
These are registration methods, i.e. methods for assessing the quality of a given registration.

All methods implement a function value_and_derivatives(transform) which returns a tuple 
(value, gradient) which measures the quality of the given transform, and the quality gradient in each of
the transform parameters. Some methods currently do not caluclate the gradient and return (value, None).

For initialization of the distance measure, no common interface has currently been implemented. 
Register object treats each method individually.

## License
The registration framework is licensed under the permissive MIT license.

## Author/Copyright
Framework was originally written by (and copyright reserved for) Johan Ofverstedt. Extended by Jo Gay.
