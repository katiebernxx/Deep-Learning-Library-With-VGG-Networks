'''network.py
Deep neural network core functionality implemented with the low-level TensorFlow API.

'''
import time
import numpy as np
import tensorflow as tf

from tf_util import arange_index

class DeepNetwork:
    '''The DeepNetwork class is the parent class for specific networks (e.g. VGG).
    '''
    def __init__(self, input_feats_shape, reg=0):
        '''DeepNetwork constructor.

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        reg: float.
            The regularization strength.

        '''
        # Keep these instance vars:
        self.optimizer_name = None
        self.loss_name = None
        self.output_layer = None
        self.all_net_params = []
        self.input_feats_shape = input_feats_shape
        self.reg = reg

    def compile(self, loss='cross_entropy', optimizer='adam', lr=1e-3, beta_1=0.9, print_summary=True):
        '''Compiles the neural network to prepare for training.

        This involves performing the following tasks:
        1. Storing instance vars for the loss function and optimizer that will be used when training.
        2. Initializing the optimizer.
        3. Doing a "pilot run" forward pass with a single fake data sample that has the same shape as those that will be
        used when training. This will trigger each weight layer's lazy initialization to initialize weights, biases, and
        any other parameters.
\        5. Get references to all the trainable parameters (e.g. wts, biases) from all network layers. This list will be
        used during backpropogation to efficiently update all the network parameters.

        Parameters:
        -----------
        loss: str.
            Loss function to use during training.
        optimizer: str.
            Optimizer to use to train trainable parameters in the network. Initially supported options: 'adam'.
        lr: float.
            Learning rate used by the optimizer during training.
        beta_1: float.
            Hyperparameter in Adam and AdamW optimizers that controls the accumulation of gradients across successive
            parameter updates (in moving average).
        print_summary: bool.
            Whether to print a summary of the network architecture and shapes of activations in each layer.

              '''
        self.loss_name = loss
        self.optimizer_name = optimizer

        # Initialize optimizer
        #TODO: Fill this section in
        if optimizer == 'adam':
            self.opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)
        elif optimizer == 'adamw':
            self.opt = tf.keras.optimizers.AdamW(learning_rate=lr, beta_1=beta_1, weight_decay=self.reg)
        else:
            raise ValueError(f'Unknown optimizer {optimizer}')


        # Do 'fake' forward pass through net to create wts/bias
        x_fake = self.get_one_fake_input()
        self(x_fake)

        # Initialize batch norm vars
        self.init_batchnorm_params()

        # Print network arch
        if print_summary:
            self.summary()

        # Get reference to all net params
        self.all_net_params = self.get_all_params()

    def get_one_fake_input(self):
        '''Generates a fake mini-batch of one sample to forward through the network when it is compiled to trigger
        lazy initialization to instantiate the weights and biases in each layer.

        '''
        return tf.zeros(shape=(1, *self.input_feats_shape))

    def summary(self):
        '''Traverses the network backward from output layer to print a summary of each layer's name and shape.

        '''
        print(75*'-')
        layer = self.output_layer
        while layer is not None:
            print(layer)
            layer = layer.get_prev_layer_or_block()
        print(75*'-')

    def set_layer_training_mode(self, is_training):
        '''Sets the training mode in each network layer.

        Parameters:
        -----------
        is_training: bool.
            True if the network is currently in training mode, False otherwise.

             '''
        layer = self.output_layer  # start from output layer
        while layer is not None:
            layer.set_mode(is_training) 
            layer = layer.prev_layer_or_block  

    def init_batchnorm_params(self):
        '''Initializes batch norm related parameters in all layers that are using batch normalization.

        '''
        pass

    def get_all_params(self, wts_only=False):
        '''Traverses the network backward from the output layer to compile a list of all trainable network paramters.


        Parameters:
        -----------
        wts_only: bool.
            Do we only collect a list of only weights (i.e. no biases or other parameters).

        Returns:
        --------
        Python list.
            List of all trainable parameters across all network layers.
        '''
        all_net_params = []

        layer = self.output_layer
        while layer is not None:
            if wts_only:
                params = layer.get_wts()

                if params is None:
                    params = []
                if not isinstance(params, list):
                    params = [params]
            else:
                params = layer.get_params()

            all_net_params.extend(params)
            layer = layer.get_prev_layer_or_block()
        return all_net_params

    def accuracy(self, y_true, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct.

        Parameters:
        -----------
        y_true: tf.constant. shape=(B,).
            int-coded true classes.
        y_pred: tf.constant. shape=(B,).
            int-coded predicted classes by the network.

        Returns:
        -----------
        float.
            The accuracy in range [0, 1]

        '''
        correct_predictions = tf.where(tf.equal(y_true, y_pred), 1.0, 0.0)  # 1 for right, 0 for wrong
        accuracy = tf.reduce_mean(correct_predictions)  # calc mean
        return accuracy

    def predict(self, x, output_layer_net_act=None):
        '''Predicts the class of each data sample in `x` using the passed in `output_layer_net_act`.
        If `output_layer_net_act` is not passed in, the method should compute it in order to perform the prediction.

        Parameters:
        -----------
        x: tf.constant. shape=(B, ...). Data samples
        output_layer_net_act: tf.constant. shape=(B, C) or None. Network activation.

        Returns:
        -----------
        tf.constant. tf.ints32. shape=(B,).
            int-coded predicted class for each sample in the mini-batch.
        '''
        if output_layer_net_act is None:
            output_layer_net_act = self(x)
    
        predictions = tf.argmax(output_layer_net_act, axis=1, output_type=tf.int32)
        return predictions
        

    def loss(self, out_net_act, y, eps=1e-16):
        '''Computes the loss for the current minibatch based on the output layer activations `out_net_act` and int-coded
        class labels `y`.

        Parameters:
        -----------
        output_layer_net_act: tf.constant. shape=(B, C) or None.
            Net activation in the output layer for the current mini-batch.
        y: tf.constant. shape=(B,). tf.int32s.
            int-coded true classes for the current mini-batch.

        Returns:
        -----------
        float.
            The loss.

        '''
        if self.loss_name.lower() != 'cross_entropy':
            raise ValueError(f'Unknown loss function {self.loss_name}')

        # else:
            # raise ValueError(f'Unknown loss function {self.loss_name}')

        # extract probs for the correct classes using arange_index
        correct_class_probs = arange_index(out_net_act, y)  # Equivalent to out_net_act[range(N), y]
        
        # calc the log loss
        log_probs = tf.math.log(correct_class_probs + eps)

        # calc mean loss
        loss = -tf.reduce_mean(log_probs)

        # Keep the following code
        # Handles the regularization for Adam
        if self.optimizer_name.lower() == 'adam':
            all_net_wts = self.get_all_params(wts_only=True)
            reg_term = self.reg*0.5*tf.reduce_sum([tf.reduce_sum(wts**2) for wts in all_net_wts])
            loss = loss + reg_term

        return loss

    def update_params(self, tape, loss):
        '''Do backpropogation: have the optimizer update the network parameters recorded on `tape` based on the
        gradients computed of `loss` with respect to each of the parameters. The variable `self.all_net_params`
        represents a 1D list of references to ALL trainable parameters in every layer of the network

        Parameters:
        -----------
        tape: tf.GradientTape.
            Gradient tape object on which all the gradients have been recorded for the most recent forward pass.
        loss: tf.Variable. float.
            The loss computed over the current mini-batch.
        '''
        grads = tape.gradient(loss, self.all_net_params)
        self.opt.apply_gradients(zip(grads, self.all_net_params))
        
    # @tf.function(jit_compile=True)
    @tf.function
    def train_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during training. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Updating the network parameters using backprop (via update_params method).

        Parameters:
        -----------
        x_batch: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.constant. tf.ints32. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The loss.

        '''
        with tf.GradientTape() as tape:
            out_net_act = self(x_batch)
            loss_value = self.loss(out_net_act, y_batch)
        
        grads = tape.gradient(loss_value, self.all_net_params)
        self.opt.apply_gradients(zip(grads, self.all_net_params))
        
        return loss_value

    # @tf.function(jit_compile=True)
    @tf.function
    def test_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during test/validation time. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Obtaining the predicted classes for the mini-batch samples.
        4. Compute the accuracy of the predictions.

        Parameters:
        -----------
        x_batch: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.constant. tf.ints32. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.

        '''
        # forward pass
        y_pred = self(x_batch)
    
        loss_value = self.loss(y_pred, y_batch)
        
        pred_classes = tf.argmax(y_pred, axis=1, output_type=tf.int32)
        
        acc = self.accuracy(y_batch, pred_classes)
        
        return acc, loss_value

    def fit(self, x, y, x_val=None, y_val=None, batch_size=128, max_epochs=10000, val_every=1, verbose=True,
            patience=999, lr_patience=999, lr_decay_factor=0.5, lr_max_decays=12):
        '''Trains the neural network on the training samples `x` (and associated int-coded labels `y`).

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(N, Iy, Ix, n_chans).
            The data samples.
        y: tf.constant. tf.int32s. shape=(N,).
            int-coded class labels
        x_val: tf.constant. tf.float32. shape=(N_val, Iy, Ix, n_chans).
            Validation set samples.
        y_val: tf.constant. tf.float32. shape=(N_val,).
            int-coded validation set class labels.
        batch_size: int.
            Number of samples to include in each mini-batch.
        max_epochs: int.
            Network should train no more than this many epochs.
            Why it is not just called `epochs` will be revealed in Week 2.
        val_every: int.
            How often (in epoches) to compute validation set accuracy and loss.
        verbose: bool.
            If `False`, there should be no print outs during training. Messages indicating start and end of training are
            fine.
        patience: int.
            Number of most recent computations of the validation set loss to consider when deciding whether to stop
            training early (before `max_epochs` is reached).
        lr_patience: int.
            Number of most recent computations of the validation set loss to consider when deciding whether to decay the
            optimizer learning rate.
        lr_decay_factor: float.
            A value between 0.0. and 1.0 that represents the proportion of the current learning rate that the learning
            rate should be set to. For example, 0.7 would mean the learning rate is set to 70% its previous value.
        lr_max_decays: int.
            Number of times we allow the lr to decay during training.

        Returns:
        -----------
        train_loss_hist: Python list of floats. len=num_epochs.
            Training loss computed on each training mini-batch and averaged across all mini-batchs in one epoch.
        val_loss_hist: Python list of floats. len=num_epochs/val_freq.
            Loss computed on the validation set every time it is checked (`val_every`).
        val_acc_hist: Python list of floats. len=num_epochs/val_freq.
            Accuracy computed on the validation every time it is checked  (`val_every`).
        e: int.
            The number of training epochs used

        '''
        self.set_layer_training_mode(is_training=True)
        N = x.shape[0]
        
        train_loss_hist = []
        val_loss_hist = []
        val_acc_hist = []

        val_loss_window = []  # track recent validation losses for early stopping
        lr_loss_window = []  # track recent validation losses for learning rate decay
        lr_decays = 0  # counter for number of times LR has been decayed
    
        rng = np.random.default_rng(0) #np.random.RandomState(42)
        
        for epoch in range(max_epochs):
            start_time = time.time()
            num_steps = int(np.ceil(N / batch_size))
            epoch_loss = 0.0
            
            for _ in range(num_steps):
                batch_indices = rng.choice(N, size=batch_size, replace=True)
                x_batch = tf.gather(x, batch_indices)
                y_batch = tf.gather(y, batch_indices)
                
                loss_value = self.train_step(x_batch, y_batch)
                epoch_loss += loss_value
            
            epoch_loss = epoch_loss / num_steps
            train_loss_hist.append(epoch_loss.numpy())
            
            if (x_val is not None) and (y_val is not None) and (epoch % val_every == 0):
                self.set_layer_training_mode(is_training=False)
                val_acc, val_loss = self.evaluate(x_val, y_val)
                val_loss_hist.append(val_loss.numpy())
                val_acc_hist.append(val_acc.numpy())

                #early stopping
                val_loss_window, stop = self.early_stopping(val_loss_window, val_loss.numpy(), patience)
                if stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

                # early stopping for learning rate decay
                # lr_loss_window.append(val_loss.numpy())
                lr_loss_window, decay_lr = self.early_stopping(lr_loss_window, val_loss.numpy(), lr_patience)
                if decay_lr and lr_decays < lr_max_decays:
                    self.update_lr(lr_decay_factor)
                    lr_loss_window = []
                    lr_decays += 1

                # if len(lr_loss_window) > lr_patience:
                #     lr_loss_window.pop(0)
                #     if lr_loss_window[-1] > lr_loss_window[0] and lr_decays < lr_max_decays:
                #         self.update_lr(lr_decay_factor)
                #         lr_decays += 1
                #         print(f"Learning rate decayed {lr_decays}/{lr_max_decays} times.")
            
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"Epoch {epoch+1}/{max_epochs}: train_loss = {epoch_loss:.4f}, "
                        f"val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}, time = {elapsed:.2f}s")
                self.set_layer_training_mode(is_training=True)
            else:
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"Epoch {epoch+1}/{max_epochs}: train_loss = {epoch_loss:.4f}, time = {elapsed:.2f}s")
        
        e = epoch
        print(f'Finished training after {epoch} epochs!')
        return train_loss_hist, val_loss_hist, val_acc_hist, e

    def evaluate(self, x, y, batch_sz=64):
        '''Evaluates the accuracy and loss on the data `x` and labels `y`. Breaks the dataset into mini-batches for you
        for efficiency.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(N, Iy, Ix, n_chans).
            The complete dataset or one of its splits (train/val/test/dev).
        y: tf.constant. tf.ints32. shape=(N,).
            int-coded labels of samples in the complete dataset or one of its splits (train/val/test/dev).
        batch_sz: int.
            The batch size used to process the provided dataset. Larger numbers will generally execute faster, but
            all samples (and activations they create in the net) in the batch need to be maintained in memory at a time,
            which can result in crashes/strange behavior due to running out of memory.
            The default batch size should work fine throughout the semester and its unlikely you will need to change it.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.
        '''
        # Set the mode in all layers to the non-training mode
        self.set_layer_training_mode(is_training=False)

        # Make sure the mini-batch size isn't larger than the number of available samples
        N = len(x)
        if batch_sz > N:
            batch_sz = N

        num_batches = N // batch_sz

        # Make sure the mini-batch size is positive...
        if num_batches < 1:
            num_batches = 1

        # Process the dataset in mini-batches by the network, evaluating and avging the acc and loss across batches.
        loss = acc = 0
        for b in range(num_batches):
            curr_x = x[b*batch_sz:(b+1)*batch_sz]
            curr_y = y[b*batch_sz:(b+1)*batch_sz]

            curr_acc, curr_loss = self.test_step(curr_x, curr_y)
            acc += curr_acc
            loss += curr_loss
        acc /= num_batches
        loss /= num_batches

        return acc, loss

    def early_stopping(self, recent_val_losses, curr_val_loss, patience):
        '''Helper method used during training to determine whether training should stop before the maximum number of
        training epochs is reached based on the most recent loss values computed on the validation set
        (`recent_val_losses`) the validation loss on the current epoch (`curr_val_loss`) and `patience`.

       
        Parameters:
        -----------
        recent_val_losses: Python list of floats. len between 0 and `patience` (inclusive).
            Recently computed losses on the validation set.
        curr_val_loss: float
            The loss computed on the validation set on the current training epoch.
        patience: int.
            The patience: how many recent loss values computed on the validation set we should consider when deciding
            whether to stop training early.

        Returns:
        -----------
        recent_val_losses: Python list of floats. len between 1 and `patience` (inclusive).
            The list of recent validation loss values passsed into this method updated to include the current validation
            loss.
        stop. bool.
            Should we stop training based on the recent validation loss values and the patience value?

             '''
        # print(recent_val_losses)
        if len(recent_val_losses) < patience:
            recent_val_losses = np.append(recent_val_losses, curr_val_loss)
            # print('Not at capacity yet...', recent_val_losses)
            return recent_val_losses, False

        # Add curr loss to our history
        recent_val_losses = np.append(recent_val_losses, curr_val_loss)
        # Get rid of prev oldest
        recent_val_losses = recent_val_losses[1:]
        # See what the new oldest is
        oldest_loss = recent_val_losses[0]
        # See what the new most recent patience-1 losses are
        recent_losses = recent_val_losses[1:]

        if oldest_loss < np.min(recent_losses):
            # print('Stopping now', recent_val_losses)
            return recent_val_losses, True
        else:
            # print('Continuing on...', recent_val_losses)
            return recent_val_losses, False

        # stop = False
        # append curr val loss to list
        # recent_val_losses.append(curr_val_loss)
        
        # print (recent_val_losses)
        
        # list should not exceed patience limit
        # if len(recent_val_losses) > patience:
        #     recent_val_losses.pop(0)  # remove oldest loss value

        

        # print (recent_val_losses)

        # early stopping condition: check if we have `patience` values recorded
        # if len(recent_val_losses) == patience:
        #     oldest_loss = recent_val_losses[0]  # first element (oldest)
        #     recent_losses = recent_val_losses[1:]  # remaining elements (newer losses)
            
        #     # if oldest loss is smaller than all more recent losses, stop training
        #     if oldest_loss < min(recent_losses):
        #         return recent_val_losses, True
        
        # return recent_val_losses, False
        # if recent_val_losses.size == patience:
        #     oldest_loss = recent_val_losses[0]
        #     recent_losses = recent_val_losses[1:]

        #     if oldest_loss < np.min(recent_losses):
        #         return recent_val_losses, True

        # if recent_val_losses.size > patience:
        #     recent_val_losses = recent_val_losses[1:]

        # return recent_val_losses, False
    


    def update_lr(self, lr_decay_rate):
        '''Adjusts the learning rate used by the optimizer to be a proportion `lr_decay_rate` of the current learning
        rate.

        (Week 3)

        Paramters:
        ----------
        lr_decay_rate: float.
            A value between 0.0. and 1.0 that represents the proportion of the current learning rate that the learning
            rate should be set to. For example, 0.7 would mean the learning rate is set to 70% its previous value.

        
        '''
        current_lr = self.opt.learning_rate.numpy()
        print('Current lr =', current_lr)

        # Update the learning rate
        # self.opt.learning_rate = current_lr * lr_decay_rate
        self.opt.learning_rate.assign(current_lr * lr_decay_rate)

        updated_lr = self.opt.learning_rate.numpy()
        print('Updated lr =', updated_lr)

        # print('Current lr=', self.opt.learning_rate.numpy(), end=' ')
        # print('Updated lr=', self.opt.learning_rate.numpy())
