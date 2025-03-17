'''layers.py
Neural network layers (e.g. Dense, Conv2D, etc.) implemented with the low-level TensorFlow API.

'''
import tensorflow as tf
from math import pi

class Layer:
    '''Parent class for all specific neural network layers (e.g. Dense, Conv2D). Implements all functionality shared in
    common across different layers (e.g. net_in, net_act).
    '''
    def __init__(self, layer_name, activation, prev_layer_or_block, do_batch_norm=False, batch_norm_momentum=0.99,
                 do_layer_norm=False):
        '''Neural network layer constructor. You should not generally make Layers objects, rather you should instantiate
        objects of the subclasses (e.g. Dense, Conv2D).

        Parameters:
        -----------
        layer_name: str.
            Human-readable name for a layer (Dense_0, Conv2D_1, etc.). Used for debugging and printing summary of net.
        activation: str.
            Name of activation function to apply within the layer (e.g. 'relu', 'linear').
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        do_batch_norm. bool:
            Whether to do batch normalization in the layer.
        do_batch_norm. float:
            The batch normalization momentum hyperparamter.
        do_layer_norm. bool:
            Whether to do layer normalization in the layer.

        '''
        self.layer_name = layer_name
        self.act_fun_name = activation
        self.prev_layer_or_block = prev_layer_or_block
        self.do_batch_norm = do_batch_norm
        self.batch_norm_momentum = batch_norm_momentum

        self.wts = None
        self.b = None
        self.output_shape = None

        # We need to make this tf.Variable so this boolean gets added to the static graph when net compiled. Otherwise,
        # bool cannot be updated during training when using @tf.function
        self.is_training = tf.Variable(False, trainable=False)

        # The following relates to features you will implement later in the semester. Ignore for now.
        self.bn_gain = None
        self.bn_bias = None
        self.bn_mean = None
        self.bn_stdev = None
        self.ln_gain = None
        self.ln_bias = None


    def get_name(self):
        '''Returns the human-readable string name of the current layer.'''
        return self.layer_name

    def get_act_fun_name(self):
        '''Returns the activation function string name used in the current layer.'''
        return self.act_fun_name

    def get_prev_layer_or_block(self):
        '''Returns a reference to the Layer object that represents the layer below the current one.'''
        return self.prev_layer_or_block

    def get_wts(self):
        '''Returns the weights of the current layer'''
        return self.wts

    def get_b(self):
        '''Returns the bias of the current layer'''
        return self.b

    def has_wts(self):
        '''Does the current layer store weights? By default, we assume it does not (i.e. always return False).'''
        return False

    def get_mode(self):
        '''Returns whether the Layer is in a training state.

        HINT: Check out the instance variables above...
        '''
        return self.is_training

    def set_mode(self, is_training):
        '''Informs the layer whether the neural network is currently training. Used in Dropout and some other layer
        types.

        Parameters:
        -----------
        is_training: bool.
            True if the network is currently training, False otherwise.

       
        '''
        self.is_training.assign(is_training)


    def init_params(self, input_shape):
        '''Initializes the Layer's parameters (wts + bias), if it has any.

        '''
        pass

    def compute_net_input(self, x):
        '''Computes the net_in on the input tensor `x`.

        '''
        pass

    def compute_net_activation(self, net_in):
        '''Computes the appropriate activation based on the `net_in` values passed in.


        Parameters:
        -----------
        net_in: tf.constant. tf.float32s. shape=(B, ...)
            The net input computed in the current layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, ...).
            The activation computed on the current mini-batch.
        '''

        if self.act_fun_name == "relu":
            net_act = tf.nn.relu(net_in)
        elif self.act_fun_name == "linear":
            net_act = net_in
        elif self.act_fun_name == "softmax":
            net_act = tf.nn.softmax(net_in, axis=-1)
        else:
            raise ValueError(f'Unknown activation function {self.act_fun_name}')
        return net_act
        
    def __call__(self, x):
        '''Do a forward pass thru the layer with mini-batch `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, ...)
            The input mini-batch computed in the current layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, ...).
            The activation computed on the current mini-batch.

               '''
        # print(x)
        # print("wts")
        # print(self.wts)
        # print('b')
        # print(self.b)
        # if isinstance(self, Dropout):
        #     net_in = x
        
        # net_in = x @ self.wts + self.b
        if self.wts is None and self.b is None and self.has_wts:
            self.init_params(x.shape)
        net_in = self.compute_net_input(x)
        if self.output_shape is None:
            self.output_shape = list(net_in.shape)
        if self.do_batch_norm:
            net_in = self.compute_batch_norm(net_in)
        net_act = self.compute_net_activation(net_in)
        if self.output_shape is None:
            self.output_shape = list(net_act.shape)
        return net_act
        

    def get_params(self):
        '''Gets a list of all the parameters learned by the layer (wts, bias, etc.).

        '''
        params = []

        if self.wts is not None:
            params.append(self.wts)
        if self.b is not None and self.b.trainable:
            params.append(self.b)
        # The following relates to features you will implement later in the semester. Running code should not
        # affect anything you are implementing now.
        if self.bn_gain is not None:
            params.append(self.bn_gain)
        if self.bn_bias is not None:
            params.append(self.bn_bias)
        if self.ln_gain is not None:
            params.append(self.ln_gain)
        if self.ln_bias is not None:
            params.append(self.ln_bias)

        return params

    def get_kaiming_gain(self):
        '''Returns the Kaiming gain that is appropriate for the current layer's activation function.


        Returns:
        --------
        float.
            The Kaiming gain.
        '''
        if self.act_fun_name == "relu":
            return 2.0
        else: 
            return 1.0

    def is_doing_batchnorm(self):
        '''Returns whether the current layer is using batch normalization.


        Returns:
        --------
        bool.
            True if the layer has batch normalization turned on, False otherwise.
        '''
        return self.do_batch_norm

    def init_batchnorm_params(self):
        '''Initializes the trainable and non-trainable parameters used in batch normalization. This includes the
        batch norm gain and bias, as well as the moving average mean and standard deviation.

       '''
        if not self.do_batch_norm:
            return
        if self.bn_mean is not None:
            return
        if self.output_shape is None:
            raise ValueError("Output shape is not set. Run a forward pass first.")

        bacth_norm_shape = [1] * (len(self.output_shape) - 1) + [self.output_shape[-1]] # [1, 1, 1, H] or [1, H]
        self.bn_gain = tf.Variable(tf.ones(bacth_norm_shape), trainable = True) # gain = 1
        self.bn_bias = tf.Variable(tf.zeros(bacth_norm_shape), trainable = True) # bias = 0
        self.bn_mean = tf.Variable(tf.zeros(bacth_norm_shape), trainable = False) # mean = 0
        self.bn_stdev = tf.Variable(tf.ones(bacth_norm_shape), trainable = False) # stdev = 1
        self.b = tf.Variable(0.0, trainable = False) #turn bias off


    def compute_batch_norm(self, net_in, eps=0.001):
        '''Computes the batch normalization based on on the net input `net_in`.

        '''

    def is_doing_layernorm(self):
        '''Check if layer normalization is enabled. True if layer normalization is enabled, False otherwise.

        '''
        pass

    def init_layernorm_params(self, x):
        '''Initializes the parameters for layer normalization if layer normalization is enabled.


        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, ...).
            Input tensor to be normalized.

               '''
        if not self.do_layer_norm:
            return

    def compute_layer_norm(self, x, eps=0.001):
        '''Computes layer normalization for the input tensor. Layer normalization normalizes the activations of the
        neurons in a layer for each data point independently, rather than across the batch.


        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, M).
            Input tensor to be normalized.
        eps: float.
            A small constant added to the standard deviation to prevent division by zero. Default is 0.001.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, M).
            The normalized tensor with the same shape as the input tensor.
        '''
        pass

    def gelu(self, net_in):
        '''Applies the Gaussian Error Linear Unit (GELU) activation function.


        Parameters:
        -----------
        net_in: tf.constant. tf.float32s. shape=(B, M)
            The net input to which the activation function should be applied.

        Returns:
        --------
        tf.constant. shape=(B, M)
            Output tensor after applying the GELU activation function.
        '''
        pass


class Dense(Layer):
    '''Neural network layer that uses Dense net input.'''
    def __init__(self, name, units, activation='relu', wt_scale=1e-3, prev_layer_or_block=None,
                 wt_init='normal', do_batch_norm=False, do_layer_norm=False):
        '''Dense layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Dense_0). Used for debugging and printing summary of net.
        units: int.
            Number of units in the layer (H).
        activation: str.
            Name of activation function to apply within the layer (e.g. 'relu', 'linear').
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        do_batch_norm. bool:
            Whether to do batch normalization in the layer.
        do_layer_norm. bool:
            Whether to do layer normalization in the layer.

              '''
        super().__init__(name, activation, prev_layer_or_block,
                         do_batch_norm=do_batch_norm,
                         do_layer_norm=do_layer_norm)
        self.units = units
        self.wt_scale = wt_scale
        self.wt_init = wt_init

    def has_wts(self):
        '''Returns whether the Dense layer has weights. This is always true so always return... :)'''
        return True

    def init_params(self, input_shape):
        '''Initializes the Dense layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list.
            The anticipated shape of mini-batches of input that the layer will process. For most of the semester,
            this list will look: (B, M).

               '''
        inputs = input_shape[-1]
        weight_shape = [inputs, self.units]
        
        if self.wt_init == 'normal':
            w_init = tf.random.normal(weight_shape, stddev=self.wt_scale)
            b_init = tf.random.normal([self.units], stddev=self.wt_scale)
        elif self.wt_init == 'he':
            # He: stddev = sqrt(2/inputs)
            gain = self.get_kaiming_gain()
            stddev = (gain / inputs) ** 0.5
            w_init = tf.random.normal(weight_shape, stddev=stddev)
            b_init = tf.zeros([self.units])  
        else:
            raise ValueError(f"Unknown weight initialization method: {self.wt_init}")
        
        self.wts = tf.Variable(w_init, name=f"{self.layer_name}_wts")
        self.b = tf.Variable(b_init, name=f"{self.layer_name}_b")

    def compute_net_input(self, x):
        '''Computes the net input for the current Dense layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, M).
            Input from the layer beneath in the network.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, H).
            The net_in.

               '''
        if self.wts is None:
            self.init_params(x.shape)
        
        net_in = x @ self.wts
        if self.b is not None:
            net_in = net_in + self.b
        return net_in

    def compute_batch_norm(self, net_in, eps=0.001):
        '''Computes the batch normalization in a manner that is appropriate for Dense layers.


        Parameters:
        -----------
        net_in: tf.constant. tf.float32s. shape=(B, H).
            The net input computed on the current mini-batch.
        eps: float.
            A small "fudge factor" to prevent division by 0 when standardizing the net_in.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, H).
            The net_in, standardized according to the batch normalization algorithm.
        '''
        if self.bn_mean is None:
            self.init_batchnorm_params()
        curr_mean = tf.reduce_mean(net_in, axis=0)
        curr_stdev = tf.math.reduce_std(net_in, axis=0)
        # if self.is_training.numpy():
        if self.is_training:
            net_in_hat = (net_in - curr_mean) / (curr_stdev + eps)
            self.bn_mean.assign(self.batch_norm_momentum * self.bn_mean + (1 - self.batch_norm_momentum) * curr_mean)
            self.bn_stdev.assign(self.batch_norm_momentum * self.bn_stdev + (1 - self.batch_norm_momentum) * curr_stdev)
        else:
            net_in_hat = (net_in - self.bn_mean) / (self.bn_stdev + eps)
        net_in_hat = self.bn_gain * net_in_hat + self.bn_bias
        return net_in_hat

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Dense layer output({self.layer_name}) shape: {self.output_shape}'

class Dropout(Layer):
    '''A dropout layer that nixes/zeros out a proportion of the net input signals.'''
    def __init__(self, name, rate, prev_layer_or_block=None):
        '''Dropout layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        rate: float.
            Proportion (between 0.0 and 1.0.) of net_in signals to drop/nix within each mini-batch.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.

               '''
        # if prev_layer_or_block is None: 
        #     prev_layer_or_block = self.get_prev_layer_or_block()
        super().__init__(layer_name= name, activation='linear', prev_layer_or_block=prev_layer_or_block)
        self.rate = rate 

    def compute_net_input(self, x):
        '''Computes the net input for the current Dropout layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, ...).
            Input from the layer beneath in the network. This could be 2D (e.g. (B, H)) if the preceding layer is Dense
            or another number of dimensions (e.g. 4D (B, Iy, Ix, K) for Conv2D).

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, ...), same shape as the input `x`.
            The net_in.

              '''
        return tf.nn.dropout(x, rate=self.rate) if self.is_training else x #if not training, return just x

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Dropout layer output({self.layer_name}) shape: {self.output_shape}'



class Flatten(Layer):
    '''A flatten layer that flattens the non-batch dimensions of the input signal.'''
    def __init__(self, name, prev_layer_or_block=None):
        '''Flatten layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.

               '''
        super().__init__(layer_name= name, activation='linear', prev_layer_or_block=prev_layer_or_block)

    def compute_net_input(self, x):
        '''Computes the net input for the current Flatten layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, ...).
            Input from the layer beneath in the network. Usually the input will come from Conv2D or MaxPool2D layers
            in which case the shape of `x` is 4D: (B, Iy, Ix, K).

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, F),
            The net_in. Here `F` is the number of units once the non-batch dimensions of the input signal `x` are
            flattened out.

               '''

        B = x.shape[0]
        F = tf.reduce_prod(x.shape[1:])
        net_in = tf.reshape(x, [B, F])
        return net_in

    def __str__(self):
  
        return f'Flatten layer output({self.layer_name}) shape: {self.output_shape}'


class MaxPool2D(Layer):
    '''A 2D maxpooling layer.'''
    def __init__(self, name, pool_size=(2, 2), strides=1, prev_layer_or_block=None, padding='VALID'):
        '''MaxPool2D layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        pool_size. tuple. len(pool_size)=2.
            The horizontal and vertical size of the pooling window.
            These will always be the same. For example: (2, 2), (3, 3), etc.
        strides. int.
            The horizontal AND vertical stride of the max pooling operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        padding: str.
            Whether or not to pad the input signal before performing max-pooling in TensorFlow str format.
            Supported options: 'VALID', 'SAME'

        '''
        super().__init__(layer_name= name, activation='linear', prev_layer_or_block=prev_layer_or_block)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def compute_net_input(self, x):
        '''Computes the net input for the current MaxPool2D layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K1).
            Input from the layer beneath in the network. Should be 4D (e.g. from a Conv2D or MaxPool2D layer).
            K1 refers to the number of units/filters in the PREVIOUS layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K2).
            The net_in. K2 refers to the number of units/filters in the CURRENT layer.

              '''

        ksize = [1, self.pool_size[0], self.pool_size[1], 1]
        # Similarly, create a 4-element list for the strides.
        strides = [1, self.strides, self.strides, 1]
        net_in = tf.nn.max_pool2d(x, ksize=ksize, strides=strides, padding=self.padding)
        return net_in
        
    def __str__(self):
        '''
        '''
        return f'MaxPool2D layer output({self.layer_name}) shape: {self.output_shape}'


class Conv2D(Layer):
    '''A 2D convolutional layer'''
    def __init__(self, name, units, kernel_size=(1, 1), strides=1, activation='relu', wt_scale=1e-3,
                 prev_layer_or_block=None, wt_init='normal', do_batch_norm=False):
        '''Conv2D layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        units: ints.
            Number of convolutional filters/units (K).
        kernel_size: tuple. len(kernel_size)=2.
            The horizontal and vertical extent (pixels) of the convolutional filters.
            These will always be the same. For example: (2, 2), (3, 3), etc.
        strides. int.
            The horizontal AND vertical stride of the convolution operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        activation: str.
            Name of the activation function to apply in the layer.
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        do_batch_norm. bool:
            Whether to do batch normalization in this layer.
      '''
        super().__init__(layer_name= name, activation=activation,
                         prev_layer_or_block=prev_layer_or_block,
                         do_batch_norm=do_batch_norm)
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.wt_scale = wt_scale
        self.wt_init = wt_init  
        self.padding = "SAME"   

    def has_wts(self):
        '''Returns whether the Conv2D layer has weights. This is always true so always return... :)'''
        return True

    def init_params(self, input_shape):
        '''Initializes the Conv2D layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list. len(input_shape)=4.
            The anticipated shape of mini-batches of input that the layer will process: (B, Iy, Ix, K1).
            K1 is the number of units/filters in the previous layer.


        '''
        inputs = input_shape[-1]
        filter_height, filter_width = self.kernel_size
        weight_shape = [filter_height, filter_width, inputs, self.units]
        
        if self.wt_init == 'normal':
            w_init = tf.random.normal(weight_shape, stddev=self.wt_scale)
            b_init = tf.random.normal([self.units], stddev=self.wt_scale)
        elif self.wt_init == 'he':
            # stddev = (2.0 / inputs) ** 0.5
            gain = self.get_kaiming_gain()
            stddev = (gain / (filter_height * filter_width * inputs)) ** 0.5
            w_init = tf.random.normal(weight_shape, stddev=stddev)

            b_init = tf.zeros([self.units]) 
        else:
            raise ValueError(f"Unknown weight initialization method: {self.wt_init}")
        
        self.wts = tf.Variable(w_init, name=f"{self.layer_name}_wts")
        self.b = tf.Variable(b_init, name=f"{self.layer_name}_b")
        

    def compute_net_input(self, x):
        '''Computes the net input for the current Conv2D layer. Uses SAME boundary conditions.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K1).
            Input from the layer beneath in the network. K1 is the number of units in the previous layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K2).
            The net_in. K2 is the number of units in the current layer.

        
        '''
        if self.wts is None:
            self.init_params(x.shape)
        
        conv = tf.nn.conv2d(x, filters=self.wts, strides=[1, self.strides, self.strides, 1], padding=self.padding)
        net_in = conv + self.b
        return net_in

    def compute_batch_norm(self, net_in, eps=0.001):
        '''Computes the batch normalization in a manner that is appropriate for Conv2D layers.


        Parameters:
        -----------
        net_in: tf.constant. tf.float32s. shape=(B, Iy, Ix, K).
            The net input computed on the current mini-batch.
        eps: float.
            A small "fudge factor" to prevent division by 0 when standardizing the net_in.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K).
            The net_in, standardized according to the batch normalization algorithm.

              '''
        if self.bn_mean is None:
            self.init_batchnorm_params()
        curr_mean = tf.reduce_mean(net_in, axis=[0, 1, 2], keepdims = True)
        curr_stdev = tf.math.reduce_std(net_in, axis=[0, 1, 2], keepdims = True)
        # if self.is_training.numpy():
        if self.is_training:
            net_in_hat = (net_in - curr_mean) / (curr_stdev + eps)
            self.bn_mean.assign(self.batch_norm_momentum * self.bn_mean + (1 - self.batch_norm_momentum) * curr_mean)
            self.bn_stdev.assign(self.batch_norm_momentum * self.bn_stdev + (1 - self.batch_norm_momentum) * curr_stdev)
        else:
            net_in_hat = (net_in - self.bn_mean) / (self.bn_stdev + eps)
        net_in_hat = self.bn_gain * net_in_hat + self.bn_bias
        return net_in_hat

    def __str__(self):

        return f'Conv2D layer output({self.layer_name}) shape: {self.output_shape}'
