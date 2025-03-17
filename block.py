'''block.py
Defines the parent Block class and VGG blocks
'''
from layers import Conv2D, MaxPool2D, Dropout, Dense


class Block:
    '''The `Block` parent class and specifies functionality shared by all blocks. All blocks inherit from this class.'''
    def __init__(self, blockname, prev_layer_or_block):
        '''Block constructor.

        Parameters:
        -----------
        blockname: str.
            Human-readable name for a block (VGGConvBlock_0, VGGConvBlock_1, etc.). Used for debugging and printing
            summary of net.
        prev_layer_or_block: Layer or Block object.
            Reference to the Layer or Block object that is beneath the current Layer object. `None` if there is no
            preceding layer or block.
            Examples VGG6: VGGConvBlock_0 → VGGConvBlock_1 → Flatten → VGGDenseBlock_0 → Dense
                The VGGConvBlock_1 block object has `prev_layer_or_block=VGGConvBlock_1` block.
                The VGGDenseBlock_0 block object has `prev_layer_or_block=Flatten` layer.

           '''
        self.blockname = blockname
        self.prev_layer_or_block = prev_layer_or_block
        self.layers = []

    def get_prev_layer_or_block(self):
        '''Returns a reference to the Layer object that represents the layer/block below the current one.

        '''
        return self.prev_layer_or_block

    def get_layer_names(self):
        '''Returns a list of human-readable string names of the layers that belong to this block.

        '''
        names = []
        for layer in self.layers:
            names.append(layer.get_name())
        return names

    def get_params(self):
        '''Returns a list of trainable parameters spread out across all layers that belong to this block.

        '''
        all_params = []

        for layer in self.layers:
            params = layer.get_params()
            all_params.extend(params)

        return all_params

    def get_wts(self):
        '''Returns a list of trainable weights (no biases/other) spread out across all layers that belong to this block.

        '''
        all_wts = []

        for layer in self.layers:
            wts = layer.get_wts()

            if wts is not None:
                all_wts.append(wts)

        return all_wts

    def get_mode(self):
        '''Gets the mode of the block (i.e. training, not training). Since this is always the same in all layers,
        we use the first layer in the block as a proxy for all of them.

        '''
        return self.layers[0].get_mode()

    def set_mode(self, is_training):
        '''Sets the mode of every layer in the block to the bool value `is_training`.

        '''

        for layer in self.layers:
            layer.set_mode(is_training)

    def init_batchnorm_params(self):
        '''Initializes the batch norm parameters in every layer in the block (only should have an effect on them if they
        are configured to perform batch normalization).

        '''
        for layer in self.layers:
            layer.init_batchnorm_params()

    def __str__(self):
        '''The toString method that gets a str representation of the layers belonging to the current block. These layers
        are indented for clarity.

        '''
        string = self.blockname + ':'
        for layer in reversed(self.layers):
            string += '\n\t' + layer.__str__()
        return string

class VGGConvBlock(Block):
    '''A convolutional block in the VGG family of neural networks. It is composed of the following sequence of layers:

    Conv2D → Conv2D → MaxPool2D


    '''
    def __init__(self, blockname, units, prev_layer_or_block, num_conv_layers=2, pool_size=(2, 2), wt_scale=1e-3,
                 dropout=False, dropout_rate=0.1, wt_init='normal', do_batch_norm=False):
        '''VGGConvBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for a block (VGGConvBlock_0, VGGConvBlock_1, etc.). Used for debugging and printing
            summary of net.
        units: int:
            Number of units (i.e. filters) to use in each convolutional layer.
        num_conv_layers: int.
            Number of 2D conv layers to place in sequence within the block. By default this is 2.
        pool_size. tuple. len(pool_size)=2.
            The horizontal and vertical size of the pooling window.
            These will always be the same. For example: (2, 2), (3, 3), etc.
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        dropout: bool.
            Whether to place a dropout layer after the 2D maxpooling layer in the block.
        dropout_rate: float.
            If using a dropout layer, the dropout rate of that layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        do_batch_norm. bool:
            Whether to do batch normalization in appropriate layers.
        '''
        super().__init__(blockname, prev_layer_or_block=prev_layer_or_block)
        self.units = units
        self.num_conv_layers = num_conv_layers
        self.pool_size = pool_size
        self.wt_scale = wt_scale
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.wt_init = wt_init
        self.do_batch_norm = do_batch_norm
        
        for i in range(num_conv_layers):
            if i == 0:
                prev_layer_or_block = prev_layer_or_block
            else:
                prev_layer_or_block = self.layers[i-1]

            conv_layer = Conv2D(blockname + '/conv_' + str(i), units, kernel_size = (3,3), strides = 1, activation = 'relu', wt_scale = wt_scale, prev_layer_or_block = prev_layer_or_block, wt_init = wt_init, do_batch_norm = do_batch_norm)
            self.layers.append(conv_layer)
    
        maxpool_layer = MaxPool2D(blockname + '/pool', pool_size = pool_size, strides = 2, prev_layer_or_block = self.layers[-1], padding = 'VALID')
        self.layers.append(maxpool_layer)

    def __call__(self, x):
        '''Forward pass through the block the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K1).
            Data samples. K1 is the number of channels/units in the PREV layer or block.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K2).
            Activations produced by the output layer to the data.
            K2 is the number of channels/units in the CURR layer or block.
            NOTE: Iy and Ix represent the spatial dims. The actual spatial dims will likely decrease in the block.

               '''
        
        for layer in self.layers:
            x = layer(x)
        return x


class VGGDenseBlock(Block):
    '''A dense block in the VGG family of neural networks. It is composed of the following sequence of layers:

    Dense → Dropout

    We leave the option of placing multiple Dense (and optionally Dropout) layers in a sequence. For example, both the
    following could happen:

    Dense → Dropout → Dense → Dropout
    Dense → Dense
    '''
    def __init__(self, blockname, units, prev_layer_or_block, num_dense_blocks=1, wt_scale=1e-3, dropout=True,
                 dropout_rate=0.5, wt_init='normal', do_batch_norm=False):
        '''VGGDenseBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for a block (VGGDenseBlock_0, etc.). Used for debugging and printing summary of net.
        units: tuple of ints:
            Number of units to use in each dense layer.
            For example: units[0] would be the number for the 1st dense layer, units[1] would be the number for the 2nd,
            etc.
        num_dense_blocks: int.
            Number of sequences of Dense (and optionally Dropout) layers in a sequence (see examples above).
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        dropout: bool.
            Whether to place a dropout layer after each Dense layer in the block.
        dropout_rate: float.
            If using a dropout layer, the dropout rate of that layer. The same in all Dropout layers.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        do_batch_norm. bool:
            Whether to do batch normalization in appropriate layers.

        '''
        super().__init__(blockname, prev_layer_or_block=prev_layer_or_block)
        self.units = units
        self.num_dense_blocks = num_dense_blocks
        self.wt_scale = wt_scale
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.wt_init = wt_init
        self.do_batch_norm = do_batch_norm

        for i in range(num_dense_blocks):
            if i == 0:
                prev_layer_or_block = prev_layer_or_block
            else:
                prev_layer_or_block = self.layers[i-1]

            dense_layer = Dense(blockname + '/dense_' + str(i), units = units[i], activation = 'relu', wt_scale = wt_scale, prev_layer_or_block = prev_layer_or_block, wt_init = wt_init, do_batch_norm = do_batch_norm)
            self.layers.append(dense_layer)

            if dropout:
                dropout_layer = Dropout(blockname + '/dropout_' + str(i), rate = dropout_rate, prev_layer_or_block = self.layers[-1])
                self.layers.append(dropout_layer)
    


    def __call__(self, x):
        '''Forward pass through the block the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy*Ix*K).
            Net act signal from Flatten layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, H).
            Activations produced by the output Dense layer to the data.

             '''
        for layer in self.layers:
            x = layer(x)
        return x
