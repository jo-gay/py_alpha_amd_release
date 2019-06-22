#
# Py-Alpha-AMD Registration Framework
# Author: Johan Ofverstedt, Jo Gay
# Reference: Fast and Robust Symmetric Image Registration Based on Distances Combining Intensity and Spatial Information
#
# Copyright 2019 Johan Ofverstedt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#

#
# Registration framework base class. 
#

# Import Numpy/Scipy
import numpy as np
#import scipy.misc
from matplotlib import pyplot as plt
from PIL import Image

# Import optimizers
from optimizers import GradientDescentOptimizer
from optimizers import AdamOptimizer
from optimizers import SciPyOptimizer
from optimizers import GridSearchOptimizer

# Import distances required for alphaAMD
from distances import QuantizedImage
from distances import alpha_amd
from distances import symmetric_amd_distance

# Import other distances
from distances import MIDistance
from distances import SSDistance
from distances import dLDPDistance

# Import transforms and filters
import filters, transforms

available_opts = ['adam', 'sgd', 'scipy', 'gridsearch']
available_models = ['alphaamd', 'mi', 'ssd', 'dldp']

class Register:
    """Operate one of the available registration algorithms, using a pyramid scheme.
    
    Normal operation would be:
        instantiate
        
        follow these steps in any order
            add image data
            choose model
            choose optimizer
            add initial transform(s)
            add pyramid level(s)
        
        initialize
        run
        
    Functions intended only for internal use are prefixed _
    """

    def __init__(self, dim):
        """Initialize with default parameter set.
        
        Args:
            dim: the number of dimensions the image has. Many parts are only tested with 2D
        """
        self.dim = dim
        self.sampling_fraction = 1.0
        
        self.opt_opts = {'gradient_magnitude_threshold': 0.00001, \
                         'iterations': 1500, \
                         'step_lengths':np.array([[0.1, 1.0]])}
        self.opt_name = 'adam'
        self.model_opts = {}
        self.set_model('alphaamd')

        self.ref_im = None
        self.flo_im = None
        self.ref_mask = None
        self.flo_mask = None
        self.ref_weights = None
        self.flo_weights = None

        # Transforms
        self.initial_transforms = []
        self.transforms_param_scaling = []
        self.output_transforms = []
        self.values = []
        self.value_history = []

        # Resolution pyramid levels
        self.pyramid_factors = []
        self.pyramid_sigmas = []

        self.distances = []

        # Reporting/Output
        self.report_func = None
        self.report_freq = 25
        self.flags = []

    def add_initial_transform(self, transform, param_scaling=None):
        """Add a starting point for optimization.
        Args:
            transform: an object implementing the transform api
            param_scaling: a list of the same length as the number of parameters
            in transform. If supplied, transform parameters will be scaled by
            these factors to facilitate optimization where parameters have different
            inherent scales, such as translation vs scale.
        """
        if param_scaling is None:
            param_scaling = np.ones((transform.get_param_count(),))
        self.initial_transforms.append(transform)
        self.transforms_param_scaling.append(param_scaling)
    
    def add_initial_transforms(self, transformlist, param_scaling=None):
        """Add a number of starting points for optimization.

        Optimizer will start independently from each point specified, and the one
        which results in the best metric value will be the final optimum returned.
        """
        for i, t in enumerate(transformlist):
            if param_scaling is None:
                pscaling = np.ones((t.get_param_count(),))
            else:
                pscaling = param_scaling[i]
            self.add_initial_transform(t, pscaling)
    
    def clear_transforms(self):
        """Remove all stored transforms."""
        self.initial_transforms = []
        self.output_transforms = []
        self.transforms_param_scaling = []
        self.values = []
        self.value_history = []
    
    def get_output(self, index):
        """Return the output transform and metric value for the initial transform
        specified.
        
        Args:
            index: integer betwee 0 and number of initial transforms used
        """
        return self.output_transforms[index], self.values[index]

    def get_outputs(self):
        """Return all output transforms and corresponding values."""
        return self.output_transforms, self.values

    def get_value_at(self, transform, p_level=-1):
        """Return the value of the associated measure at the given transform, for
        pyramid level p_level."""
        v, _ = self.distances[p_level].value_and_derivatives(transform)
        return v
        
    def get_value_history(self, index, level):
        """Return the metric value for a given initial transform and pyramid level.
        
        Args:
            index: integer between 0 and number of initial transforms used
            level: integer between 0 and number of pyramid levels used
        """
        return self.value_history[index][level]

    def add_pyramid_level(self, factor, sigma):
        """Append a pyramid level to any that already exist.
        
        Args:
            factor: downsample rate
            sigma: gaussian blur parameter
        """
        self.pyramid_factors.append(factor)
        self.pyramid_sigmas.append(sigma)
    
    def add_pyramid_levels(self, factors, sigmas):
        """Append a number of pyramid levels to any that already exist.
        
        Args:
            factor: list of downsample rates
            sigma: list of gaussian blur parameters
        """
        for i in range(len(factors)):
            self.add_pyramid_level(factors[i], sigmas[i])

    def get_pyramid_level_count(self):
        """Return number of pyramid levels that have been set."""
        return len(self.pyramid_factors)

    def set_sampling_fraction(self, sampling_fraction):
        """Set the fraction of pixels to be sampled."""
        self.sampling_fraction = sampling_fraction
    
    def set_iterations(self, iterations):
        """Set the maximum number of iterations for the optimizer to use, where applicable."""
        self.opt_opts['iterations'] = iterations

    def set_step_lengths(self, step_lengths):
        """Set the step length to be used by the optimizer, where applicable."""
        self.opt_opts['step_lengths'] = np.array(step_lengths)#np.array([start_step_length, end_step_length])
    
    def set_model(self, model_name, **kwargs):
        """Set the model for the registration process. This determines which measure will be used
        to evaluate a given transform.
        
        Args:
            model_name: one of those listed in available_models
            **kwargs: Any other params required for the specific model.
        """
        assert(model_name.lower() in available_models), 'Model must be one of '+','.join(available_models)
        self.model_name = model_name.lower()
        self.model_opts = kwargs.copy()
        if self.model_name == 'alphaamd':
            self._make_dist_measure = self._make_alphaAMD_dist_measure
        elif self.model_name == 'mi':
            self._make_dist_measure = self._make_mi_dist_measure
        elif self.model_name == 'ssd':
            self._make_dist_measure = self._make_ssd_dist_measure
        elif self.model_name == 'dldp':
            self._make_dist_measure = self._make_dldp_dist_measure
        else:
            raise NotImplementedError(f'Sorry, model {model_name} has not been implemented')

    def set_optimizer(self, opt_name, **kwargs):
        """Set the optimizer for the registration process.
        
        Args:
            opt_name: one of those listed in available_opts
            **kwargs: any other parameters required for the optimizer. Any existing values will
            be cleared.
        """
        assert(opt_name.lower() in available_opts), 'Optimizer must be one of '+','.join(available_opts)
        self.opt_name = opt_name.lower()
        self.opt_opts = kwargs.copy()

    def set_reference_image(self, image, spacing = None):
        """Set the reference image."""
        self.ref_im = image
        if spacing is None:
            self.ref_spacing = np.ones(image.ndim)
        else:
            self.ref_spacing = spacing
        
    def set_floating_image(self, image, spacing = None):
        """Set the floating image."""
        self.flo_im = image
        if spacing is None:
            self.flo_spacing = np.ones(image.ndim)
        else:
            self.flo_spacing = spacing

    def set_reference_mask(self, mask):
        """Set a mask to show which parts of the reference image are valid."""
        self.ref_mask = mask

    def set_floating_mask(self, mask):
        """Set a mask to show which parts of the floating image are valid."""
        self.flo_mask = mask

    def set_reference_weights(self, weights):
        """Set pixel-wise weights for the reference image."""
        self.ref_weights = weights

    def set_floating_weights(self, weights):
        """Set pixel-wise weights for the floating image."""
        self.flo_weights = weights
        
    def set_image_data(self, ref_im, flo_im, ref_mask, flo_mask, ref_weights, flo_weights):
        """Shortcut to set all these parameters in one line."""
        self.set_reference_image(ref_im)
        self.set_floating_image(flo_im)
        self.set_reference_mask(ref_mask)
        self.set_floating_mask(flo_mask)
        self.set_reference_weights(ref_weights)
        self.set_floating_weights(flo_weights)

    def set_gradient_magnitude_threshold(self, t):
        """Set the threshold for the gradient for use with gradient descent optimizers."""
        self.opt_opts['gradient_magnitude_threshold'] = t

    def get_flags(self):
        return self.flags

    def set_report_freq(self, freq):
        self.report_freq = freq
    
    def set_report_func(self, func):
        self.report_func = func

    def _make_alphaAMD_dist_measure(self, ref_resampled, ref_mask_resampled, ref_weights, \
                          flo_resampled, flo_mask_resampled, flo_weights, pyramid_factor):

        alpha_levels = self.model_opts.get('alpha_levels', 7)
        ref_diag = np.sqrt(np.square(np.array(ref_resampled.shape)*self.ref_spacing).sum())
        flo_diag = np.sqrt(np.square(np.array(flo_resampled.shape)*self.flo_spacing).sum())

        q_ref = QuantizedImage(ref_resampled, alpha_levels, ref_weights, \
                               self.ref_spacing*pyramid_factor, remove_zero_weight_pnts = True)
        q_flo = QuantizedImage(flo_resampled, alpha_levels, flo_weights, \
                               self.flo_spacing*pyramid_factor, remove_zero_weight_pnts = True)

        tf_ref = alpha_amd.AlphaAMD(q_ref, alpha_levels, ref_diag, \
                                    self.ref_spacing*pyramid_factor, ref_mask_resampled, \
                                    ref_mask_resampled, interpolator_mode='linear', \
                                    dt_fun = None, mask_out_edges = True)
        tf_flo = alpha_amd.AlphaAMD(q_flo, alpha_levels, flo_diag, \
                                    self.flo_spacing*pyramid_factor, flo_mask_resampled, \
                                    flo_mask_resampled, interpolator_mode='linear', \
                                    dt_fun = None, mask_out_edges = True)

        symmetric_measure = self.model_opts.get('symmetric_measure', True)
        squared_measure = self.model_opts.get('squared_measure', False)

        sym_dist = symmetric_amd_distance.SymmetricAMDDistance(symmetric_measure=symmetric_measure, \
                                                               squared_measure=squared_measure)

        sym_dist.set_ref_image_source(q_ref)
        sym_dist.set_ref_image_target(tf_ref)

        sym_dist.set_flo_image_source(q_flo)
        sym_dist.set_flo_image_target(tf_flo)

        sym_dist.set_sampling_fraction(self.sampling_fraction)

        sym_dist.initialize()
        return sym_dist

    def _make_mi_dist_measure(self, ref_resampled, ref_mask_resampled, ref_weights, \
                          flo_resampled, flo_mask_resampled, flo_weights, pyramid_factor):

        # reducing levels to 32 improves running time by less than 20% but may improve stability
        mi_dist = MIDistance(self.model_opts.get('mutual_info_fn', 'norm'), self.model_opts.get('levels', 32)) 
        mi_dist.set_ref_image(ref_resampled, mask=ref_mask_resampled)
        mi_dist.set_flo_image(flo_resampled)

        mi_dist.initialize()
        return mi_dist
    
    def _make_ssd_dist_measure(self, ref_resampled, ref_mask_resampled, ref_weights, \
                          flo_resampled, flo_mask_resampled, flo_weights, pyramid_factor):
        dist = SSDistance()
        dist.set_ref_image(ref_resampled, mask=ref_mask_resampled)
        dist.set_flo_image(flo_resampled)

        dist.initialize()
        return dist

    def _make_dldp_dist_measure(self, ref_resampled, ref_mask_resampled, ref_weights, \
                          flo_resampled, flo_mask_resampled, flo_weights, pyramid_factor):
        dist = dLDPDistance(self.model_opts.get('derivative_mode', 'diff'))
        dist.set_ref_image(ref_resampled, mask=ref_mask_resampled)
        dist.set_flo_image(flo_resampled)

        dist.initialize()
        return dist
    
    def _make_dist_measure(self, *args, **kwargs):
        """Default function to be overwritten with one of the above during setup.
        """
        raise ValueError('Framework has not been initialized - please specify a model using set_model()')

    def initialize(self, pyramid_images_output_path=None):
        """Initialize the registration framework: must be called before run().
        
        Prepare pyramid scheme by creating and saving downsampled versions of the images 
        for each pyramid level. Set up a distance measure (separate instance for each pyramid
        level, with the corresponding version of the images).
        
        Args:
            pyramid_images_output_path: slash-terminated string specifying folder in which to
            save the downsampled images. Default None. If None, images are not saved. Only 
            applicable for 2D images
            
        Other running parameters are set in __init()__
        """
        if len(self.pyramid_factors) == 0:
            self.add_pyramid_level(1, 0.0)
        if len(self.initial_transforms) == 0:
            self.add_initial_transform(transforms.AffineTransform(self.dim))
        self.opt_opts['step_lengths'] = self.opt_opts.get('step_lengths',np.array([[0,1.0]]))
        while len(self.opt_opts['step_lengths']) < len(self.pyramid_factors):
            self.opt_opts['step_lengths'] = np.concatenate((self.opt_opts['step_lengths'], np.array([[0,1.0]])))
        
        ### Preprocessing

        pyramid_levels = len(self.pyramid_factors)

        for i in range(pyramid_levels):
            factor = self.pyramid_factors[i]

            ref_resampled = filters.downsample(filters.gaussian_filter(self.ref_im, self.pyramid_sigmas[i]), factor)
            flo_resampled = filters.downsample(filters.gaussian_filter(self.flo_im, self.pyramid_sigmas[i]), factor)
            
            ref_mask_resampled = filters.downsample(self.ref_mask, factor)
            flo_mask_resampled = filters.downsample(self.flo_mask, factor)

            ref_resampled = filters.normalize(ref_resampled, 0.0, ref_mask_resampled)
            flo_resampled = filters.normalize(flo_resampled, 0.0, flo_mask_resampled)

            if pyramid_images_output_path is not None and ref_resampled.ndim == 2:
                Image.fromarray(ref_resampled).convert('RGB').save(pyramid_images_output_path+'ref_resampled_%d.png'%(i+1))
                Image.fromarray(flo_resampled).convert('RGB').save(pyramid_images_output_path+'flo_resampled_%d.png'%(i+1))
#                scipy.misc.imsave('%sref_resampled_%d.png' % (pyramid_images_output_path, i+1), ref_resampled)
#                scipy.misc.imsave('%sflo_resampled_%d.png' % (pyramid_images_output_path, i+1), flo_resampled)
            
            if self.ref_weights is None:
                ref_weights = np.zeros(ref_resampled.shape)
                ref_weights[ref_mask_resampled] = 1.0
            else:
                ref_weights = filters.downsample(self.ref_weights, factor)
            if self.flo_weights is None:
                flo_weights = np.zeros(flo_resampled.shape)
                flo_weights[flo_mask_resampled] = 1.0
            else:
                flo_weights = filters.downsample(self.flo_weights, factor)

            if(False):
                #Display each image with its mask to check all is ok
                plt.subplot(121)
                plt.imshow(np.hstack((ref_resampled, ref_mask_resampled)), cmap='gray')
                plt.subplot(122)
                plt.imshow(np.hstack((flo_resampled, flo_mask_resampled)), cmap='gray')
                plt.show()
                
                
            dist_measure = self._make_dist_measure(ref_resampled, ref_mask_resampled, ref_weights, \
                                                  flo_resampled, flo_mask_resampled, flo_weights, factor)

            self.distances.append(dist_measure)

    def _initialize_optimizer(self, t_it, lvl_it, init_transform):
        """Instantiate and initialize optimizer depending on which one has been selected.
        
        Args:
            t_it: which initial transform are we working from (its index from 0 up)
            lvl_it: which pyramid level we are at (affects optimizer paramters)
            init_transform: where the optimizer should start from
        Returns:
            optimizer
        """
        assert(self.opt_name in available_opts), 'Optimizer has not been set'

        if self.opt_name == 'adam':
            opt = AdamOptimizer(self.distances[lvl_it], init_transform.copy())
            opt.set_gradient_magnitude_threshold(self.opt_opts.get('gradient_magnitude_threshold', 1e-6))
        elif self.opt_name == 'sgd':
            opt = GradientDescentOptimizer(self.distances[lvl_it], init_transform.copy())
            opt.set_gradient_magnitude_threshold(self.opt_opts.get('gradient_magnitude_threshold', 1e-6))
        elif self.opt_name == 'scipy':
            opt = SciPyOptimizer(self.distances[lvl_it], init_transform.copy(), method='L-BFGS-B')
            minim_opts = {'gtol': self.opt_opts.get('gradient_magnitude_threshold', 1e-9), \
                          'eps': self.opt_opts.get('epsilon', 0.1)}
#            #For lower resolutions (earlier levels in the pyramid) use smaller steps to avoid
#            #translating too far. Also use a larger gradient tolerance as we don't need high 
#            #accuracy before the final level
#            minim_opts = {'gtol': self.opt_opts['gradient_magnitude_threshold']*np.power(self.pyramid_factors[lvl_it], 2), \
#                          'eps': 0.02/self.pyramid_factors[lvl_it]}
#            minim_opts = {'xatol': 1e-1, \
#                          'ftol': 1e-6}
#            opt = SciPyOptimizer(self.distances[lvl_it], init_transform.copy(), method='Nelder-Mead')
            opt.set_minimizer_options(minim_opts)
        elif self.opt_name == 'gridsearch':
            opt = GridSearchOptimizer(self.distances[lvl_it], init_transform.copy(), \
                                      bounds=self.opt_opts.get('bounds', []), \
                                      steps=self.opt_opts.get('steps', 1),
                                      verbose=self.opt_opts.get('verbose', False))
        else:
            raise NotImplementedError(f'Sorry, optimizer {self.opt_name} has not been implemented')
        
        opt.set_report_freq(self.report_freq)
        
        if self.opt_opts['step_lengths'].ndim == 1:
            opt.set_step_length(*self.opt_opts['step_lengths'])
        else:
            opt.set_step_length(*self.opt_opts['step_lengths'][lvl_it, :])

        param_scaling = self.transforms_param_scaling[t_it]
        opt.set_scalings(param_scaling)
        
        if type(self.report_func) is list or type(self.report_func) is tuple:
            opt.set_report_callback(self.report_func[t_it])
        else:
            opt.set_report_callback(self.report_func)

        return opt

    def run(self, verbose=False):
        """Run the registration algorithm.
        
        For each initial transform, find the optimum transform for the first
        pyramid level. Use that transform as the starting point for the next
        pyramid level. Record the optimum transform from the last pyramid level
        along with the corresponding metric value. If using a generic optimizer
        then also record the success flag returned by the optimizer.
        """
        pyramid_level_count = len(self.pyramid_factors)
        transform_count = len(self.initial_transforms)

        for t_it in range(transform_count):
            init_transform = self.initial_transforms[t_it]

            self.value_history.append([])
            if self.opt_name == 'scipy':
                self.flags.append([])

            for lvl_it in range(pyramid_level_count):
                opt = self._initialize_optimizer(t_it, lvl_it, init_transform)
                
                itercount = self.opt_opts.get('iterations', 1000)
                if not isinstance(itercount, int):
                    assert(len(itercount) == pyramid_level_count), \
                            "Error in optimizer iterations, must be a single integer or " + \
                            "list of same length as pyramid levels"
                    itercount = self.opt_opts['iterations'][lvl_it]
        
                opt.optimize(itercount)

                if lvl_it + 1 == pyramid_level_count:
                    self.output_transforms.append(opt.get_transform())
                    self.values.append(opt.get_value())
                    self.initial_transforms[t_it] = opt.get_transform()
                else:
                    init_transform = opt.get_transform()

                self.value_history[-1].append(opt.get_value_history())
                if self.opt_name == 'scipy':
                    self.flags[-1].append(opt.success)
                if verbose:
                    print(f'Pyramid level {lvl_it} terminating at [' + \
                          ', '.join(['%.3f']*len(opt.get_transform().get_params()))%tuple(opt.get_transform().get_params()) \
                          + '] with value %.4f'%opt.get_value())
