'''vgg_nets.py
The family of VGG neural networks implemented using the CS444 deep learning library

'''
import network
from layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from block import VGGConvBlock, VGGDenseBlock


class VGG4(network.DeepNetwork):
    '''The VGG4 neural network, which has the following architecture:

    Conv2D → Conv2D → MaxPool2D → Flatten → Dense → Dropout → Dense

       '''
    def __init__(self, C, input_feats_shape, filters=64, dense_units=128, reg=0, wt_scale=1e-3, wt_init='normal'):
        '''VGG4 network constructor

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: int.
            Number of filters in each convolutional layer (the same in all layers).
        dense_units: int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
                    '''
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
        self.filters = filters
        self.dense_units = dense_units
        self.wt_scale = wt_scale
        self.wt_init = wt_init
        self.C = C

        self.layer1 = Conv2D('conv1', filters, kernel_size = (3,3), strides = 1, activation = 'relu', wt_scale=wt_scale, prev_layer_or_block=None,
                        wt_init = wt_init, do_batch_norm = False)
        self.layer2 = Conv2D('conv2', filters, kernel_size = (3,3), strides = 1, activation = 'relu', wt_scale = wt_scale, prev_layer_or_block=self.layer1,
                             wt_init = wt_init, do_batch_norm = False)
        self.layer3 = MaxPool2D('pool1', pool_size=(2, 2), strides=2, prev_layer_or_block=self.layer2, padding='VALID')
        self.layer4 = Flatten('flatten1', prev_layer_or_block=self.layer3)
        self.layer5 = Dense("dense1", units = dense_units, activation='relu', wt_scale=wt_scale, prev_layer_or_block=self.layer4,
                 wt_init=wt_init, do_batch_norm=False, do_layer_norm=False)
        self.layer6 = Dropout('dropout1', rate = 0.5, prev_layer_or_block=self.layer5)
        self.output_layer = Dense("denseout", units = C, activation='softmax', wt_scale=wt_scale, prev_layer_or_block=self.layer6,
                 wt_init=wt_init, do_batch_norm=False, do_layer_norm=False) 
        
    def __call__(self, x):
        '''Forward pass through the VGG4 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.output_layer(x)
        return x

class VGG6(network.DeepNetwork):
    '''The VGG6 neural network, which has the following architecture:

    Conv2D → Conv2D → MaxPool2D → Conv2D → Conv2D → MaxPool2D → Flatten → Dense → Dropout → Dense

    
    '''
    def __init__(self, C, input_feats_shape, filters=(64, 128), dense_units=(256,), reg=0, wt_scale=1e-3,
                 wt_init='normal'):
        '''The VGG6 constructor.

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: tuple of ints.
            Number of filters in each convolutional layer of a block.
            The same for conv layers WITHIN a block, different for conv layers BETWEEN blocks.
        dense_units: tuple of int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            
        '''
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
        self.filters = filters
        self.dense_units = dense_units
        self.wt_scale = wt_scale
        self.wt_init = wt_init
        self.C = C

        self.block1 = VGGConvBlock('block1', filters[0], wt_scale=wt_scale, wt_init=wt_init, prev_layer_or_block=None)
        self.block2 = VGGConvBlock('block2', filters[1], wt_scale=wt_scale, wt_init=wt_init, prev_layer_or_block=self.block1)
        self.layer3 = Flatten('flatten1', prev_layer_or_block=self.block2)
        self.block4 = VGGDenseBlock('block3', dense_units, wt_scale=wt_scale, wt_init=wt_init, prev_layer_or_block=self.layer3)

        self.output_layer = Dense("denseout", units = C, activation='softmax', wt_scale=wt_scale, prev_layer_or_block=self.block4,
                    wt_init=wt_init, do_batch_norm=False, do_layer_norm=False)
        
    def __call__(self, x):
        '''Forward pass through the VGG6 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.layer3(x)
        x = self.block4(x)
        x = self.output_layer(x)
        return x


class VGG8(network.DeepNetwork):
    '''The VGG8 neural network, which has the following architecture:

    Conv2D → Conv2D → MaxPool2D → Conv2D → Conv2D → MaxPool2D → Conv2D → Conv2D → MaxPool2D → Flatten → Dense → Dropout → Dense

    
    '''
    def __init__(self, C, input_feats_shape, filters=(64, 128, 256), dense_units=(512,), reg=0, wt_scale=1e-3,
                 wt_init='he', conv_dropout=False, conv_dropout_rates=(0.1, 0.2, 0.3)):
        '''The VGG8 constructor.

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: tuple of ints.
            Number of filters in each convolutional layer of a block.
            The same for conv layers WITHIN a block, different for conv layers BETWEEN blocks.
        dense_units: tuple of int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer (if using normal wt init method).
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        conv_dropout: bool.
            Do we place a dropout layer in each conv block?
        conv_dropout_rates: tuple of floats. len(conv_dropout_rates)=num_conv_blocks
            The dropout rate to use in each conv block. Only has an effect if `conv_dropout` is True.

        
        '''
        self.C = C
        self.input_feats_shape = input_feats_shape
        self.filters = filters
        self.dense_units = dense_units
        self.reg = reg
        self.wt_scale = wt_scale
        self.wt_init = wt_init
        self.conv_dropout = conv_dropout
        self.conv_dropout_rates = conv_dropout_rates

        self.block1 = VGGConvBlock('block1', filters[0], wt_scale=wt_scale, wt_init=wt_init, prev_layer_or_block=None)
        self.block2 = VGGConvBlock('block2', filters[1], wt_scale=wt_scale, wt_init=wt_init, prev_layer_or_block=self.block1)
        self.block3 = VGGConvBlock('block3', filters[2], wt_scale=wt_scale, wt_init=wt_init, prev_layer_or_block=self.block2)
        self.layer4 = Flatten('flatten1', prev_layer_or_block=self.block3)
        self.block5 = VGGDenseBlock('block5', dense_units, wt_scale=wt_scale, wt_init=wt_init, prev_layer_or_block=self.layer4)

        self.output_layer = Dense("denseout", units = C, activation='softmax', wt_scale=wt_scale, prev_layer_or_block=self.block5,
                    wt_init=wt_init, do_batch_norm=False, do_layer_norm=False)
    

    def __call__(self, x):
        '''Forward pass through the VGG8 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.layer4(x)
        x = self.block5(x)
        x = self.output_layer(x)
        return x

class VGG15(network.DeepNetwork):
    '''The VGG15 neural network, which has the following architecture:

    Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → Conv2D → MaxPool2D →
    Flatten →
    Dense → Dropout →
    Dense

  
    '''
    def __init__(self, C, input_feats_shape, filters=(64, 128, 256, 512, 512), dense_units=(512,), reg=0.6,
                 wt_scale=1e-3, wt_init='he', conv_dropout=False, conv_dropout_rates=(0.1, 0.2, 0.3, 0.3, 0.3)):
        '''The VGG15 constructor.

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: tuple of ints.
            Number of filters in each convolutional layer of a block.
            The same for conv layers WITHIN a block, different for conv layers BETWEEN blocks.
        dense_units: tuple of int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer (if using normal wt init method).
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        conv_dropout: bool.
            Do we place a dropout layer in each conv block?
        conv_dropout_rates: tuple of floats. len(conv_dropout_rates)=num_conv_blocks
            The dropout rate to use in each conv block. Only has an effect if `conv_dropout` is True.

        
        '''
        self.C = C
        self.input_feats_shape = input_feats_shape
        self.filters = filters
        self.dense_units = dense_units
        self.reg = reg
        self.wt_scale = wt_scale
        self.wt_init = wt_init
        self.conv_dropout = conv_dropout
        self.conv_dropout_rates = conv_dropout_rates

        self.block1 = VGGConvBlock('block1', filters[0], wt_scale=wt_scale, num_conv_layers=2, wt_init=wt_init, prev_layer_or_block=None)
        self.block2 = VGGConvBlock('block2', filters[1], wt_scale=wt_scale, num_conv_layers=2, wt_init=wt_init, prev_layer_or_block=self.block1)
        self.block3 = VGGConvBlock('block3', filters[2], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, prev_layer_or_block=self.block2)
        self.block4 = VGGConvBlock('block4', filters[3], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, prev_layer_or_block=self.block3)
        self.block5 = VGGConvBlock('block5', filters[4], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, prev_layer_or_block=self.block4)
        self.layer4 = Flatten('flatten1', prev_layer_or_block=self.block5)
        self.block6 = VGGDenseBlock('block6', dense_units, wt_scale=wt_scale, wt_init=wt_init, prev_layer_or_block=self.layer4, dropout=True)

        self.output_layer = Dense("denseout", units = C, activation='softmax', wt_scale=wt_scale, prev_layer_or_block=self.block6,
                    wt_init=wt_init, do_batch_norm=False, do_layer_norm=False)
    


    def __call__(self, x):
        '''Forward pass through the VGG15 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.layer4(x)
        x = self.block6(x)
        x = self.output_layer(x)
        return x


class VGG4Plus(network.DeepNetwork):
    '''The VGG4 network with batch normalization added to all Conv2D layers and all non-output Dense layers.'''
    def __init__(self, C, input_feats_shape, filters=64, dense_units=128, reg=0, wt_scale=1e-3, wt_init='he'):
        '''VGG4Plus network constructor

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: int.
            Number of filters in each convolutional layer (the same in all layers).
        dense_units: int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        '''
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
        self.filters = filters
        self.dense_units = dense_units
        self.wt_scale = wt_scale
        self.wt_init = wt_init
        self.C = C
        self.layer1 = Conv2D('conv1', filters, kernel_size = (3,3), strides = 1, activation = 'relu', wt_scale=wt_scale, prev_layer_or_block=None,
                        wt_init = wt_init, do_batch_norm = True)
        self.layer2 = Conv2D('conv2', filters, kernel_size = (3,3), strides = 1, activation = 'relu', wt_scale = wt_scale, prev_layer_or_block=self.layer1,
                             wt_init = wt_init, do_batch_norm = True)
        self.layer3 = MaxPool2D('pool1', pool_size=(2, 2), strides=2, prev_layer_or_block=self.layer2, padding='VALID')
        self.layer4 = Flatten('flatten1', prev_layer_or_block=self.layer3)
        self.layer5 = Dense("dense1", units = dense_units, activation='relu', wt_scale=wt_scale, prev_layer_or_block=self.layer4,
                 wt_init=wt_init, do_batch_norm=True, do_layer_norm=False)
        self.layer6 = Dropout('dropout1', rate = 0.5, prev_layer_or_block=self.layer5)
        self.output_layer = Dense("denseout", units = C, activation='softmax', wt_scale=wt_scale, prev_layer_or_block=self.layer6,
                 wt_init=wt_init, do_batch_norm=False, do_layer_norm=False) 
                
    def __call__(self, x):
        '''Forward pass through the VGG15 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.output_layer(x)
        return x


class VGG15Plus(network.DeepNetwork):
    '''The VGG15Plus network is the VGG15 network with batch normalization added to all Conv2D layers and all
    non-output Dense layers.
    '''
    def __init__(self, C, input_feats_shape, filters=(64, 128, 256, 512, 512), dense_units=(512,), reg=0.6,
                  wt_scale=1e-3, wt_init='he', conv_dropout=False, conv_dropout_rates=(0.1, 0.2, 0.3, 0.3, 0.3)):
        '''The VGG15Plus constructor.

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: tuple of ints.
            Number of filters in each convolutional layer of a block.
            The same for conv layers WITHIN a block, different for conv layers BETWEEN blocks.
        dense_units: tuple of int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer (if using normal wt init method).
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        conv_dropout: bool.
            Do we place a dropout layer in each conv block?
        conv_dropout_rates: tuple of floats. len(conv_dropout_rates)=num_conv_blocks
            The dropout rate to use in each conv block. Only has an effect if `conv_dropout` is True.
        '''
        self.C = C
        self.input_feats_shape = input_feats_shape
        self.filters = filters
        self.dense_units = dense_units
        self.reg = reg
        self.wt_scale = wt_scale
        self.wt_init = wt_init
        self.conv_dropout = conv_dropout
        self.conv_dropout_rates = conv_dropout_rates

        self.block1 = VGGConvBlock('block1', filters[0], wt_scale=wt_scale, num_conv_layers=2, wt_init=wt_init, do_batch_norm= True, prev_layer_or_block=None)
        self.block2 = VGGConvBlock('block2', filters[1], wt_scale=wt_scale, num_conv_layers=2, wt_init=wt_init, do_batch_norm= True, prev_layer_or_block=self.block1)
        self.block3 = VGGConvBlock('block3', filters[2], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, do_batch_norm= True, prev_layer_or_block=self.block2)
        self.block4 = VGGConvBlock('block4', filters[3], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, do_batch_norm= True, prev_layer_or_block=self.block3)
        self.block5 = VGGConvBlock('block5', filters[4], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, do_batch_norm=True, prev_layer_or_block=self.block4)
        self.layer4 = Flatten('flatten1', prev_layer_or_block=self.block5)
        self.block6 = VGGDenseBlock('block6', dense_units, wt_scale=wt_scale, wt_init=wt_init, do_batch_norm= True, prev_layer_or_block=self.layer4, dropout=True)

        self.output_layer = Dense("denseout", units = C, activation='softmax', wt_scale=wt_scale, prev_layer_or_block=self.block6,
                    wt_init=wt_init, do_batch_norm=False, do_layer_norm=False)
    

 
    def __call__(self, x):
        '''Forward pass through the VGG15 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.layer4(x)
        x = self.block6(x)
        x = self.output_layer(x)
        return x



class VGG15PlusPlus(network.DeepNetwork):
    '''The VGG15PlusPlus network is the VGG15 network with:
    1. Batch normalization added to all Conv2D layers and all non-output Dense layers.
    2. Dropout added to all conv blocks.
    '''
    def __init__(self, C, input_feats_shape, filters=(64, 128, 256, 512, 512), dense_units=(512,), reg=0.6,
                 wt_scale=1e-3, wt_init='he', conv_dropout=True, conv_dropout_rates=(0.3, 0.4, 0.4, 0.4, 0.4)):
        self.C = C
        self.input_feats_shape = input_feats_shape
        self.filters = filters
        self.dense_units = dense_units
        self.reg = reg
        self.wt_scale = wt_scale
        self.wt_init = wt_init
        self.conv_dropout = conv_dropout
        self.conv_dropout_rates = conv_dropout_rates

        self.block1 = VGGConvBlock('block1', filters[0], wt_scale=wt_scale, num_conv_layers=2, wt_init=wt_init, do_batch_norm= True, dropout= True, prev_layer_or_block=None)
        self.block2 = VGGConvBlock('block2', filters[1], wt_scale=wt_scale, num_conv_layers=2, wt_init=wt_init, do_batch_norm= True, dropout= True, prev_layer_or_block=self.block1)
        self.block3 = VGGConvBlock('block3', filters[2], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, do_batch_norm= True, dropout= True, prev_layer_or_block=self.block2)
        self.block4 = VGGConvBlock('block4', filters[3], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, do_batch_norm= True, dropout= True, prev_layer_or_block=self.block3)
        self.block5 = VGGConvBlock('block5', filters[4], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, do_batch_norm=True, dropout= True, prev_layer_or_block=self.block4)
        self.layer4 = Flatten('flatten1', prev_layer_or_block=self.block5)
        self.block6 = VGGDenseBlock('block6', dense_units, wt_scale=wt_scale, wt_init=wt_init, do_batch_norm= True, prev_layer_or_block=self.layer4, dropout=True)

        self.output_layer = Dense("denseout", units = C, activation='softmax', wt_scale=wt_scale, prev_layer_or_block=self.block6,
                    wt_init=wt_init, do_batch_norm=False, do_layer_norm=False)
    

    def __call__(self, x):
        '''Forward pass through the VGG15 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.layer4(x)
        x = self.block6(x)
        x = self.output_layer(x)
        return x

class VGG16(network.DeepNetwork):
    '''The VGG16 neural network, which has the following architecture:

    Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → Conv2D → MaxPool2D →
    Conv2D → Conv2D → Conv2D → MaxPool2D →
    Flatten →
    Dense → Dropout →
    Dense → Dropout →
    Dense

    '''
    def __init__(self, C, input_feats_shape, filters=(64, 128, 256, 512, 512), dense_units=(512,), reg=0.6,
                 wt_scale=1e-3, wt_init='he', conv_dropout=False, conv_dropout_rates=(0.1, 0.2, 0.3, 0.3, 0.3)):
        '''The VGG16 constructor.

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: tuple of ints.
            Number of filters in each convolutional layer of a block.
            The same for conv layers WITHIN a block, different for conv layers BETWEEN blocks.
        dense_units: tuple of int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer (if using normal wt init method).
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        conv_dropout: bool.
            Do we place a dropout layer in each conv block?
        conv_dropout_rates: tuple of floats. len(conv_dropout_rates)=num_conv_blocks
            The dropout rate to use in each conv block. Only has an effect if `conv_dropout` is True.

        '''
        self.C = C
        self.input_feats_shape = input_feats_shape
        self.filters = filters
        self.dense_units = dense_units
        self.reg = reg
        self.wt_scale = wt_scale
        self.wt_init = wt_init
        self.conv_dropout = conv_dropout
        self.conv_dropout_rates = conv_dropout_rates
        

        self.block1 = VGGConvBlock('block1', filters[0], wt_scale=wt_scale, num_conv_layers=2, wt_init=wt_init, prev_layer_or_block=None, do_batch_norm= True)
        self.block2 = VGGConvBlock('block2', filters[1], wt_scale=wt_scale, num_conv_layers=2, wt_init=wt_init, prev_layer_or_block=self.block1, do_batch_norm= True)
        self.block3 = VGGConvBlock('block3', filters[2], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, prev_layer_or_block=self.block2, do_batch_norm= True)
        self.block4 = VGGConvBlock('block4', filters[3], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, prev_layer_or_block=self.block3, do_batch_norm= True)
        self.block5 = VGGConvBlock('block5', filters[4], wt_scale=wt_scale, num_conv_layers=3, wt_init=wt_init, prev_layer_or_block=self.block4, do_batch_norm= True)
        self.layer4 = Flatten('flatten1', prev_layer_or_block=self.block5)
        self.block6 = VGGDenseBlock('block6', dense_units, wt_scale=wt_scale, wt_init=wt_init, prev_layer_or_block=self.layer4, dropout=True, do_batch_norm= True)
        self.block7 = VGGDenseBlock('block7', dense_units, wt_scale=wt_scale, wt_init=wt_init, prev_layer_or_block=self.block6, dropout=True, do_batch_norm= True)

        self.output_layer = Dense("denseout", units = C, activation='softmax', wt_scale=wt_scale, prev_layer_or_block=self.block7,
                    wt_init=wt_init)

    def __call__(self, x):
        '''Forward pass through the VGG15 network with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.layer4(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.output_layer(x)
        return x


class AlexNet(network.DeepNetwork):
    '''
    Conv2D → MaxPool2D → Conv2D → MaxPool2D → Conv2D → Conv2D → Conv2D → MaxPool2D →
    Flatten → Dense → Dropout → Dense → Dropout → Dense

    '''
    def __init__(self, C, input_feats_shape, filters=(96, 256, 384, 384, 256), dense_units=(4096, 4096), reg=0,
                 wt_scale=1e-3, wt_init='normal'):
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
        self.C = C
        self.filters = filters
        self.dense_units = dense_units
        self.wt_scale = wt_scale
        self.wt_init = wt_init

        self.layer1 = Conv2D('conv1', filters[0], kernel_size=(11, 11), strides=4, activation='relu', 
                             wt_scale=wt_scale, prev_layer_or_block=None, wt_init=wt_init, do_batch_norm=False)
        self.layer2 = MaxPool2D('pool1', pool_size=(3, 3), strides=2, prev_layer_or_block=self.layer1, padding='VALID')

        self.layer3 = Conv2D('conv2', filters[1], kernel_size=(5, 5), strides=1, activation='relu', 
                             wt_scale=wt_scale, prev_layer_or_block=self.layer2, wt_init=wt_init, do_batch_norm=False)
        self.layer4 = MaxPool2D('pool2', pool_size=(3, 3), strides=2, prev_layer_or_block=self.layer3, padding='VALID')

        self.layer5 = Conv2D('conv3', filters[2], kernel_size=(3, 3), strides=1, activation='relu', 
                             wt_scale=wt_scale, prev_layer_or_block=self.layer4, wt_init=wt_init, do_batch_norm=False)
        self.layer6 = Conv2D('conv4', filters[3], kernel_size=(3, 3), strides=1, activation='relu', 
                             wt_scale=wt_scale, prev_layer_or_block=self.layer5, wt_init=wt_init, do_batch_norm=False)
        self.layer7 = Conv2D('conv5', filters[4], kernel_size=(3, 3), strides=1, activation='relu', 
                             wt_scale=wt_scale, prev_layer_or_block=self.layer6, wt_init=wt_init, do_batch_norm=False)
        self.layer8 = MaxPool2D('pool3', pool_size=(3, 3), strides=2, prev_layer_or_block=self.layer7, padding='VALID')

        self.layer9 = Flatten('flatten1', prev_layer_or_block=self.layer8)
        self.layer10 = Dense('dense1', units=dense_units[0], activation='relu', wt_scale=wt_scale, 
                             prev_layer_or_block=self.layer9, wt_init=wt_init, do_batch_norm=False, do_layer_norm=False)
        self.layer11 = Dropout('dropout1', rate=0.5, prev_layer_or_block=self.layer10)

        self.layer12 = Dense('dense2', units=dense_units[1], activation='relu', wt_scale=wt_scale, 
                             prev_layer_or_block=self.layer11, wt_init=wt_init, do_batch_norm=False, do_layer_norm=False)
        self.layer13 = Dropout('dropout2', rate=0.5, prev_layer_or_block=self.layer12)

        self.output_layer = Dense('denseout', units=C, activation='softmax', wt_scale=wt_scale, 
                                  prev_layer_or_block=self.layer13, wt_init=wt_init, do_batch_norm=False, do_layer_norm=False)

    def __call__(self, x):
        '''Forward pass through AlexNet.

        Parameters:
        -----------
        x: tf.constant. shape=(B, Iy, Ix, n_chans).
            Input data samples.

        Returns:
        --------
        tf.constant. shape=(B, C).
            Output activations.
        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.output_layer(x)
        return x
