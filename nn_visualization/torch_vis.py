import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from skimage.transform import resize
import nibabel as nib

############### General-purpose Gradients/Saliency #######################

def get_image_grad(model, image, true_class = 1, normalize = True):
    # Inputs:
    # - model: pytorch model; outputs length-2 vector, no softmax yet
    # - true_class: class for which to evaluate
    # - image: 4D torch image (torch.Tensor)
    # - normalize: whether or not to normalize the gradient
    # Outputs:
    # - Gradient of output wrt input image
    #   output dim: [1,1,x,y], (x,y) is shape of image
    # - NOTE: assumes model output is [[healthy score, AD score]]
    #         computes gradient wrt index 'true_class'
    
    # First set up image
    image.requires_grad = True
    image.grad = None # clear grad in case it has previous stored
    
    # run model, get predicted AD certainty 
    ad_pred = model(image)[:,true_class]
    ad_pred.backward()
    
    # get gradient
    g = image.grad
    
    # normalize if needed
    if normalize:
        g = g / torch.sqrt(torch.sum(g * g))
        
    return g



def get_saliency(model, image):
    # Inputs:
    # - model: pytorch model; outputs length-2 vector, no softmax yet
    # - image: 4D torch image (torch.Tensor)
    # Outputs:
    # - saliency map (abs. gradients) for the image
    #   2D numpy array, same size as original image
    
    # get (unnormalized) gradient
    grad = get_image_grad(model, image, normalize = False)
    
    # get absolute value & remove extra dimensions
    grad = torch.abs(grad)[0,0]
    
    # convert to numpy
    grad = grad.detach().numpy()
    
    return grad







################ Layer-wise Relevance Propagation ######################
# All useful code (everything other than wrapper) here taken from: https://github.com/moboehle/Pytorch-LRP

### Wrapper ###
def lrp_image(model, image, 
              rel_for_class = 0,
              beta = 0.25):
    # Wrapper function for easier use of LRP
    # Inputs:
    # - model: trained pytorch model
    # - image: brain image (4D)
    # - rel_for_class: class to compute for (0 = healthy, 1 = AD)
    # - beta: parameter for LRP; higher means more significant areas found, but more noise
    # Outputs:
    # - 
    
    # Set up LRP object
    lrp_model = InnvestigateModel(model,
                        beta = beta,
                        method = "b-rule")
    # run lrp, remove extra dims from result
    model_out, lrp_out = lrp_model.innvestigate(image, rel_for_class = rel_for_class)
    lrp_out = lrp_out[0, 0]
    
    # convert to numpy array
    lrp_out = lrp_out.detach().numpy()
    lrp_out = np.abs(lrp_out)
    return lrp_out




### Utils ###
def pprint(*args):
    out = [str(argument) + "\n" for argument in args]
    print(*out, "\n")


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))





### Helper Class/methods ###
def module_tracker(fwd_hook_func):
    """
    Wrapper for tracking the layers throughout the forward pass.
    Args:
        fwd_hook_func: Forward hook function to be wrapped.
    Returns:
        Wrapped method.
    """
    def hook_wrapper(relevance_propagator_instance, layer, *args):
        relevance_propagator_instance.module_list.append(layer)
        return fwd_hook_func(relevance_propagator_instance, layer, *args)

    return hook_wrapper


class RelevancePropagator:
    """
    Class for computing the relevance propagation and supplying
    the necessary forward hooks for all layers.
    """

    # All layers that do not require any specific forward hooks.
    # This is due to the fact that they are all one-to-one
    # mappings and hence no normalization is needed (each node only
    # influences exactly one other node -> relevance conservation
    # ensures that relevance is just inherited in a one-to-one manner, too).
    allowed_pass_layers = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                           torch.nn.BatchNorm3d,
                           torch.nn.ReLU, torch.nn.ELU, Flatten,
                           torch.nn.Dropout, torch.nn.Dropout2d,
                           torch.nn.Dropout3d,
                           torch.nn.Softmax,
                           torch.nn.LogSoftmax,
                           torch.nn.Sigmoid)
    # Implemented rules for relevance propagation.
    available_methods = ["e-rule", "b-rule"]

    def __init__(self, lrp_exponent, beta, method, epsilon, device):

        self.device = device
        self.layer = None
        self.p = lrp_exponent
        self.beta = beta
        self.eps = epsilon
        self.warned_log_softmax = False
        self.module_list = []
        if method not in self.available_methods:
            raise NotImplementedError("Only methods available are: " +
                                      str(self.available_methods))
        self.method = method

    def reset_module_list(self):
        """
        The module list is reset for every evaluation, in change the order or number
        of layers changes dynamically.
        Returns:
            None
        """
        self.module_list = []
        # Try to free memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def compute_propagated_relevance(self, layer, relevance):
        """
        This method computes the backward pass for the incoming relevance
        for the specified layer.
        Args:
            layer: Layer to be reverted.
            relevance: Incoming relevance from higher up in the network.
        Returns:
            The
        """

        if isinstance(layer,
                      (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
            return self.max_pool_nd_inverse(layer, relevance).detach()

        elif isinstance(layer,
                      (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            return self.conv_nd_inverse(layer, relevance).detach()

        elif isinstance(layer, torch.nn.LogSoftmax):
            # Only layer that does not conserve relevance. Mainly used
            # to make probability out of the log values. Should probably
            # be changed to pure passing and the user should make sure
            # the layer outputs are sensible (0 would be 100% class probability,
            # but no relevance could be passed on).
            if relevance.sum() < 0:
                relevance[relevance == 0] = -1e6
                relevance = relevance.exp()
                if not self.warned_log_softmax:
                    pprint("WARNING: LogSoftmax layer was "
                           "turned into probabilities.")
                    self.warned_log_softmax = True
            return relevance
        elif isinstance(layer, self.allowed_pass_layers):
            # The above layers are one-to-one mappings of input to
            # output nodes. All the relevance in the output will come
            # entirely from the input node. Given the conservation
            # of relevance, the input is as relevant as the output.
            return relevance

        elif isinstance(layer, torch.nn.Linear):
            return self.linear_inverse(layer, relevance).detach()
        else:
            raise NotImplementedError("The network contains layers that"
                                      " are currently not supported {0:s}".format(str(layer)))

    def get_layer_fwd_hook(self, layer):
        """
        Each layer might need to save very specific data during the forward
        pass in order to allow for relevance propagation in the backward
        pass. For example, for max_pooling, we need to store the
        indices of the max values. In convolutional layers, we need to calculate
        the normalizations, to ensure the overall amount of relevance is conserved.
        Args:
            layer: Layer instance for which forward hook is needed.
        Returns:
            Layer-specific forward hook.
        """

        if isinstance(layer,
                      (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
            return self.max_pool_nd_fwd_hook

        if isinstance(layer,
                      (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            return self.conv_nd_fwd_hook

        if isinstance(layer, self.allowed_pass_layers):
            return self.silent_pass  # No hook needed.

        if isinstance(layer, torch.nn.Linear):
            return self.linear_fwd_hook

        else:
            raise NotImplementedError("The network contains layers that"
                                      " are currently not supported {0:s}".format(str(layer)))

    @staticmethod
    def get_conv_method(conv_module):
        """
        Get dimension-specific convolution.
        The forward pass and inversion are made in a
        'dimensionality-agnostic' manner and are the same for
        all nd instances of the layer, except for the functional
        that needs to be used.
        Args:
            conv_module: instance of convolutional layer.
        Returns:
            The correct functional used in the convolutional layer.
        """

        conv_func_mapper = {
            torch.nn.Conv1d: F.conv1d,
            torch.nn.Conv2d: F.conv2d,
            torch.nn.Conv3d: F.conv3d
        }
        return conv_func_mapper[type(conv_module)]

    @staticmethod
    def get_inv_conv_method(conv_module):
        """
        Get dimension-specific convolution inversion layer.
        The forward pass and inversion are made in a
        'dimensionality-agnostic' manner and are the same for
        all nd instances of the layer, except for the functional
        that needs to be used.
        Args:
            conv_module: instance of convolutional layer.
        Returns:
            The correct functional used for inverting the convolutional layer.
        """

        conv_func_mapper = {
            torch.nn.Conv1d: F.conv_transpose1d,
            torch.nn.Conv2d: F.conv_transpose2d,
            torch.nn.Conv3d: F.conv_transpose3d
        }
        return conv_func_mapper[type(conv_module)]

    @module_tracker
    def silent_pass(self, m, in_tensor: torch.Tensor,
                    out_tensor: torch.Tensor):
        # Placeholder forward hook for layers that do not need
        # to store any specific data. Still useful for module tracking.
        pass

    @staticmethod
    def get_inv_max_pool_method(max_pool_instance):
        """
        Get dimension-specific max_pooling layer.
        The forward pass and inversion are made in a
        'dimensionality-agnostic' manner and are the same for
        all nd instances of the layer, except for the functional
        that needs to be used.
        Args:
            max_pool_instance: instance of max_pool layer.
        Returns:
            The correct functional used in the max_pooling layer.
        """

        conv_func_mapper = {
            torch.nn.MaxPool1d: F.max_unpool1d,
            torch.nn.MaxPool2d: F.max_unpool2d,
            torch.nn.MaxPool3d: F.max_unpool3d
        }
        return conv_func_mapper[type(max_pool_instance)]

    def linear_inverse(self, m, relevance_in):

        if self.method == "e-rule":
            m.in_tensor = m.in_tensor.pow(self.p)
            w = m.weight.pow(self.p)
            norm = F.linear(m.in_tensor, w, bias=None)

            norm = norm + torch.sign(norm) * self.eps
            relevance_in[norm == 0] = 0
            norm[norm == 0] = 1
            relevance_out = F.linear(relevance_in / norm,
                                     w.t(), bias=None)
            relevance_out *= m.in_tensor
            del m.in_tensor, norm, w, relevance_in
            return relevance_out

        if self.method == "b-rule":
            out_c, in_c = m.weight.size()
            w = m.weight.repeat((4, 1))
            # First and third channel repetition only contain the positive weights
            w[:out_c][w[:out_c] < 0] = 0
            w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
            # Second and fourth channel repetition with only the negative weights
            w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
            w[-out_c:][w[-out_c:] > 0] = 0

            # Repeat across channel dimension (pytorch always has channels first)
            m.in_tensor = m.in_tensor.repeat((1, 4))
            m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
            m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
            m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0

            # Normalize such that the sum of the individual importance values
            # of the input neurons divided by the norm
            # yields 1 for an output neuron j if divided by norm (v_ij in paper).
            # Norm layer just sums the importance values of the inputs
            # contributing to output j for each j. This will then serve as the normalization
            # such that the contributions of the neurons sum to 1 in order to
            # properly split up the relevance of j amongst its roots.

            norm_shape = m.out_shape
            norm_shape[1] *= 4
            norm = torch.zeros(norm_shape).to(self.device)

            for i in range(4):
                norm[:, out_c * i:(i + 1) * out_c] = F.linear(
                    m.in_tensor[:, in_c * i:(i + 1) * in_c], w[out_c * i:(i + 1) * out_c], bias=None)

            # Double number of output channels for positive and negative norm per
            # channel.
            norm_shape[1] = norm_shape[1] // 2
            new_norm = torch.zeros(norm_shape).to(self.device)
            new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
            new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
            norm = new_norm

            # Some 'rare' neurons only receive either
            # only positive or only negative inputs.
            # Conservation of relevance does not hold, if we also
            # rescale those neurons by (1+beta) or -beta.
            # Therefore, catch those first and scale norm by
            # the according value, such that it cancels in the fraction.

            # First, however, avoid NaNs.
            mask = norm == 0
            # Set the norm to anything non-zero, e.g. 1.
            # The actual inputs are zero at this point anyways, that
            # is why norm is zero in the first place.
            norm[mask] = 1
            # The norm in the b-rule has shape (N, 2*out_c, *spatial_dims).
            # The first out_c block corresponds to the positive norms,
            # the second out_c block corresponds to the negative norms.
            # We find the rare neurons by choosing those nodes per channel
            # in which either the positive norm ([:, :out_c]) is zero, or
            # the negative norm ([:, :out_c]) is zero.
            rare_neurons = (mask[:, :out_c] + mask[:, out_c:])

            # Also, catch new possibilities for norm == zero to avoid NaN..
            # The actual value of norm again does not really matter, since
            # the pre-factor will be zero in this case.

            norm[:, :out_c][rare_neurons] *= 1 if self.beta == -1 else 1 + self.beta
            norm[:, out_c:][rare_neurons] *= 1 if self.beta == 0 else -self.beta
            # Add stabilizer term to norm to avoid numerical instabilities.
            norm += self.eps * torch.sign(norm)
            input_relevance = relevance_in.squeeze(dim=-1).repeat(1, 4)
            input_relevance[:, :2*out_c] *= (1+self.beta)/norm[:, :out_c].repeat(1, 2)
            input_relevance[:, 2*out_c:] *= -self.beta/norm[:, out_c:].repeat(1, 2)
            inv_w = w.t()
            relevance_out = torch.zeros_like(m.in_tensor)
            for i in range(4):
                relevance_out[:, i*in_c:(i+1)*in_c] = F.linear(
                    input_relevance[:, i*out_c:(i+1)*out_c],
                    weight=inv_w[:, i*out_c:(i+1)*out_c], bias=None)

            relevance_out *= m.in_tensor

            relevance_out = sum([relevance_out[:, i*in_c:(i+1)*in_c] for i in range(4)])

            del input_relevance, norm, rare_neurons, \
                mask, new_norm, m.in_tensor, w, inv_w

            return relevance_out

    @module_tracker
    def linear_fwd_hook(self, m, in_tensor: torch.Tensor,
                        out_tensor: torch.Tensor):

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_shape", list(out_tensor.size()))
        return

    def max_pool_nd_inverse(self, layer_instance, relevance_in):

        # In case the output had been reshaped for a linear layer,
        # make sure the relevance is put into the same shape as before.
        relevance_in = relevance_in.view(layer_instance.out_shape)

        invert_pool = self.get_inv_max_pool_method(layer_instance)
        inverted = invert_pool(relevance_in, layer_instance.indices,
                               layer_instance.kernel_size, layer_instance.stride,
                               layer_instance.padding, output_size=layer_instance.in_shape)
        del layer_instance.indices

        return inverted

    @module_tracker
    def max_pool_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):
        # Ignore unused for pylint
        _ = self

        # Save the return indices value to make sure
        tmp_return_indices = bool(m.return_indices)
        m.return_indices = True
        _, indices = m.forward(in_tensor[0])
        m.return_indices = tmp_return_indices
        setattr(m, "indices", indices)
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_shape', in_tensor[0].size())

    def conv_nd_inverse(self, m, relevance_in):

        # In case the output had been reshaped for a linear layer,
        # make sure the relevance is put into the same shape as before.
        relevance_in = relevance_in.view(m.out_shape)

        # Get required values from layer
        inv_conv_nd = self.get_inv_conv_method(m)
        conv_nd = self.get_conv_method(m)

        if self.method == "e-rule":
            with torch.no_grad():
                m.in_tensor = m.in_tensor.pow(self.p).detach()
                w = m.weight.pow(self.p).detach()
                norm = conv_nd(m.in_tensor, weight=w, bias=None,
                               stride=m.stride, padding=m.padding,
                               groups=m.groups)

                norm = norm + torch.sign(norm) * self.eps
                relevance_in[norm == 0] = 0
                norm[norm == 0] = 1
                relevance_out = inv_conv_nd(relevance_in/norm,
                                            weight=w, bias=None,
                                            padding=m.padding, stride=m.stride,
                                            groups=m.groups)
                relevance_out *= m.in_tensor
                del m.in_tensor, norm, w
                return relevance_out

        if self.method == "b-rule":
            with torch.no_grad():
                w = m.weight

                out_c, in_c = m.out_channels, m.in_channels
                repeats = np.array(np.ones_like(w.size()).flatten(), dtype=int)
                repeats[0] *= 4
                w = w.repeat(tuple(repeats))
                # First and third channel repetition only contain the positive weights
                w[:out_c][w[:out_c] < 0] = 0
                w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
                # Second and fourth channel repetition with only the negative weights
                w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
                w[-out_c:][w[-out_c:] > 0] = 0
                repeats = np.array(np.ones_like(m.in_tensor.size()).flatten(), dtype=int)
                repeats[1] *= 4
                # Repeat across channel dimension (pytorch always has channels first)
                m.in_tensor = m.in_tensor.repeat(tuple(repeats))
                m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
                m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
                m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0
                groups = 4

                # Normalize such that the sum of the individual importance values
                # of the input neurons divided by the norm
                # yields 1 for an output neuron j if divided by norm (v_ij in paper).
                # Norm layer just sums the importance values of the inputs
                # contributing to output j for each j. This will then serve as the normalization
                # such that the contributions of the neurons sum to 1 in order to
                # properly split up the relevance of j amongst its roots.
                norm = conv_nd(m.in_tensor, weight=w, bias=None, stride=m.stride,
                               padding=m.padding, dilation=m.dilation, groups=groups * m.groups)
                # Double number of output channels for positive and negative norm per
                # channel. Using list with out_tensor.size() allows for ND generalization
                new_shape = m.out_shape
                new_shape[1] *= 2
                new_norm = torch.zeros(new_shape).to(self.device)
                new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
                new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
                norm = new_norm
                # Some 'rare' neurons only receive either
                # only positive or only negative inputs.
                # Conservation of relevance does not hold, if we also
                # rescale those neurons by (1+beta) or -beta.
                # Therefore, catch those first and scale norm by
                # the according value, such that it cancels in the fraction.

                # First, however, avoid NaNs.
                mask = norm == 0
                # Set the norm to anything non-zero, e.g. 1.
                # The actual inputs are zero at this point anyways, that
                # is why norm is zero in the first place.
                norm[mask] = 1
                # The norm in the b-rule has shape (N, 2*out_c, *spatial_dims).
                # The first out_c block corresponds to the positive norms,
                # the second out_c block corresponds to the negative norms.
                # We find the rare neurons by choosing those nodes per channel
                # in which either the positive norm ([:, :out_c]) is zero, or
                # the negative norm ([:, :out_c]) is zero.
                rare_neurons = (mask[:, :out_c] + mask[:, out_c:])

                # Also, catch new possibilities for norm == zero to avoid NaN..
                # The actual value of norm again does not really matter, since
                # the pre-factor will be zero in this case.

                norm[:, :out_c][rare_neurons] *= 1 if self.beta == -1 else 1 + self.beta
                norm[:, out_c:][rare_neurons] *= 1 if self.beta == 0 else -self.beta
                # Add stabilizer term to norm to avoid numerical instabilities.
                norm += self.eps * torch.sign(norm)
                spatial_dims = [1] * len(relevance_in.size()[2:])

                input_relevance = relevance_in.repeat(1, 4, *spatial_dims)
                input_relevance[:, :2*out_c] *= (1+self.beta)/norm[:, :out_c].repeat(1, 2, *spatial_dims)
                input_relevance[:, 2*out_c:] *= -self.beta/norm[:, out_c:].repeat(1, 2, *spatial_dims)
                # Each of the positive / negative entries needs its own
                # convolution. TODO: Can this be done in groups, too?

                relevance_out = torch.zeros_like(m.in_tensor)
                # Weird code to make up for loss of size due to stride
                tmp_result = result = None
                for i in range(4):
                    tmp_result = inv_conv_nd(
                        input_relevance[:, i*out_c:(i+1)*out_c],
                        weight=w[i*out_c:(i+1)*out_c],
                        bias=None, padding=m.padding, stride=m.stride,
                        groups=m.groups)
                    result = torch.zeros_like(relevance_out[:, i*in_c:(i+1)*in_c])
                    tmp_size = tmp_result.size()
                    slice_list = [slice(0, l) for l in tmp_size]
                    result[slice_list] += tmp_result
                    relevance_out[:, i*in_c:(i+1)*in_c] = result
                relevance_out *= m.in_tensor

                sum_weights = torch.zeros([in_c, in_c * 4, *spatial_dims]).to(self.device)
                for i in range(m.in_channels):
                    sum_weights[i, i::in_c] = 1
                relevance_out = conv_nd(relevance_out, weight=sum_weights, bias=None)

                del sum_weights, m.in_tensor, result, mask, rare_neurons, norm, \
                    new_norm, input_relevance, tmp_result, w

                return relevance_out

    @module_tracker
    def conv_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', list(out_tensor.size()))
        return


### Primary LRP Module ### 
class InnvestigateModel(torch.nn.Module):
    """
    ATTENTION:
        Currently, innvestigating a network only works if all
        layers that have to be inverted are specified explicitly
        and registered as a module. If., for example,
        only the functional max_poolnd is used, the inversion will not work.
    """

    def __init__(self, the_model, lrp_exponent=1, beta=.5, epsilon=1e-6,
                 method="e-rule"):
        """
        Model wrapper for pytorch models to 'innvestigate' them
        with layer-wise relevance propagation (LRP) as introduced by Bach et. al
        (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).
        Given a class level probability produced by the model under consideration,
        the LRP algorithm attributes this probability to the nodes in each layer.
        This allows for visualizing the relevance of input pixels on the resulting
        class probability.
        Args:
            the_model: Pytorch model, e.g. a pytorch.nn.Sequential consisting of
                        different layers. Not all layers are supported yet.
            lrp_exponent: Exponent for rescaling the importance values per node
                            in a layer when using the e-rule method.
            beta: Beta value allows for placing more (large beta) emphasis on
                    nodes that positively contribute to the activation of a given node
                    in the subsequent layer. Low beta value allows for placing more emphasis
                    on inhibitory neurons in a layer. Only relevant for method 'b-rule'.
            epsilon: Stabilizing term to avoid numerical instabilities if the norm (denominator
                    for distributing the relevance) is close to zero.
            method: Different rules for the LRP algorithm, b-rule allows for placing
                    more or less focus on positive / negative contributions, whereas
                    the e-rule treats them equally. For more information,
                    see the paper linked above.
        """
        super(InnvestigateModel, self).__init__()
        self.model = the_model
        self.device = torch.device("cpu", 0)
        self.prediction = None
        self.r_values_per_layer = None
        self.only_max_score = None
        # Initialize the 'Relevance Propagator' with the chosen rule.
        # This will be used to back-propagate the relevance values
        # through the layers in the innvestigate method.
        self.inverter = RelevancePropagator(lrp_exponent=lrp_exponent,
                                            beta=beta, method=method, epsilon=epsilon,
                                            device=self.device)

        # Parsing the individual model layers
        self.register_hooks(self.model)
        if method == "b-rule" and float(beta) in (-1., 0):
            which = "positive" if beta == -1 else "negative"
            which_opp = "negative" if beta == -1 else "positive"
            # print("WARNING: With the chosen beta value, "
            #       "only " + which + " contributions "
            #       "will be taken into account.\nHence, "
            #       "if in any layer only " + which_opp +
            #       " contributions exist, the "
            #       "overall relevance will not be conserved.\n")

    def cuda(self, device=None):
        self.device = torch.device("cuda", device)
        self.inverter.device = self.device
        return super(InnvestigateModel, self).cuda(device)

    def cpu(self):
        self.device = torch.device("cpu", 0)
        self.inverter.device = self.device
        return super(InnvestigateModel, self).cpu()

    def register_hooks(self, parent_module):
        """
        Recursively unrolls a model and registers the required
        hooks to save all the necessary values for LRP in the forward pass.
        Args:
            parent_module: Model to unroll and register hooks for.
        Returns:
            None
        """
        for mod in parent_module.children():
            if list(mod.children()):
                self.register_hooks(mod)
                continue
            mod.register_forward_hook(
                self.inverter.get_layer_fwd_hook(mod))
            if isinstance(mod, torch.nn.ReLU):
                mod.register_backward_hook(
                    self.relu_hook_function
                )

    @staticmethod
    def relu_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero.
        """
        return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, in_tensor):
        """
        The innvestigate wrapper returns the same prediction as the
        original model, but wraps the model call method in the evaluate
        method to save the last prediction.
        Args:
            in_tensor: Model input to pass through the pytorch model.
        Returns:
            Model output.
        """
        return self.evaluate(in_tensor)

    def evaluate(self, in_tensor):
        """
        Evaluates the model on a new input. The registered forward hooks will
        save all the data that is necessary to compute the relevance per neuron per layer.
        Args:
            in_tensor: New input for which to predict an output.
        Returns:
            Model prediction
        """
        # Reset module list. In case the structure changes dynamically,
        # the module list is tracked for every forward pass.
        self.inverter.reset_module_list()
        self.prediction = self.model(in_tensor)
        return self.prediction

    def get_r_values_per_layer(self):
        if self.r_values_per_layer is None:
            pprint("No relevances have been calculated yet, returning None in"
                   " get_r_values_per_layer.")
        return self.r_values_per_layer

    def innvestigate(self, in_tensor=None, rel_for_class=None):
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.
        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                self.evaluate(in_tensor)

            # If no class index is specified, analyze for class
            # with highest prediction.
            if rel_for_class is None:
                # Default behaviour is innvestigating the output
                # on an arg-max-basis, if no class is specified.
                org_shape = self.prediction.size()
                # Make sure shape is just a 1D vector per batch example.
                self.prediction = self.prediction.view(org_shape[0], -1)
                max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                relevance_tensor = only_max_score.view(org_shape)
                self.prediction.view(org_shape)

            else:
                org_shape = self.prediction.size()
                self.prediction = self.prediction.view(org_shape[0], -1)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[:, rel_for_class] += self.prediction[:, rel_for_class]
                relevance_tensor = only_max_score.view(org_shape)
                self.prediction.view(org_shape)

            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            for layer in rev_model:
                # Compute layer specific backwards-propagation of relevance values
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                r_values_per_layer.append(relevance.cpu())

            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return self.prediction, r_values_per_layer[-1]

    def forward(self, in_tensor):
        return self.model.forward(in_tensor)

    def extra_repr(self):
        r"""Set the extra representation of the module
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return self.model.extra_repr()





############# CNN Model/Architecture ####################
class ClassificationModel2D(nn.Module):

    def __init__(self, dropout=0.4, dropout2=0.4):
        nn.Module.__init__(self)
        self.Conv_1 = nn.Conv2d(1, 8, 3)
        self.Conv_1_bn = nn.BatchNorm2d(8)
        self.Conv_1_mp = nn.MaxPool2d(2)
        self.Conv_2 = nn.Conv2d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm2d(16)
        self.Conv_2_mp = nn.MaxPool2d(3)
        self.Conv_3 = nn.Conv2d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm2d(32)
        self.Conv_3_mp = nn.MaxPool2d(2)
        self.Conv_4 = nn.Conv2d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm2d(64)
        self.Conv_4_mp = nn.MaxPool2d(3)
        self.dense_1 = nn.Linear(576, 128)
        self.dense_2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        x = self.dense_2(x)
        return x