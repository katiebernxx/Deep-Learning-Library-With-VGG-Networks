'''datasets.py
Loads and preprocesses datasets for use in neural networks.

'''
import tensorflow as tf
import numpy as np


def load_dataset(name):
    '''Uses TensorFlow Keras to load and return  the dataset with string nickname `name`.

    Parameters:
    -----------
    name: str.
        Name of the dataset that should be loaded. Support options in Project 1: 'cifar10', 'mnist'.

    Returns:
    --------
    x: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        The training set (preliminary).
    y: tf.constant. tf.int32s.
        The training set int-coded labels (preliminary).
    x_test: tf.constant. tf.float32s.
        The test set.
    y_test: tf.constant. tf.int32s.
        The test set int-coded labels.
    classnames: Python list. strs. len(classnames)=num_unique_classes.
        The human-readable string names of the classes in the dataset. If there are 10 classes, len(classnames)=10.

    Summary of preprocessing steps:
    -------------------------------
    1. Uses tf.keras.datasets to load the specified dataset training set and test set.
    2. Loads the class names from the .txt file downloaded from the project website with the same name as the dataset
        (e.g. cifar10.txt).
    3. Features: Converted from UINT8 to tf.float32 and normalized so that a 255 pixel value gets mapped to 1.0 and a
        0 pixel value gets mapped to 0.0.
    4. Labels: Converted to tf.int32 and flattened into a tensor of shape (N,).

    '''
     # load dataset based on name
    if name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        classnames_file = 'cifar10.txt'
    elif name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        classnames_file = 'mnist.txt'
        x_train = tf.expand_dims(x_train, axis=-1)
        x_test = tf.expand_dims(x_test, axis=-1)
    else:
        raise ValueError(f"Dataset '{name}' not supported. Choose 'cifar10' or 'mnist'.")

    # load class names from txt file
    with open(classnames_file, 'r') as f:
        classnames = [line.strip() for line in f.readlines()]

    # convert to float32 and normalize
    x_train = tf.cast(x_train, tf.float32) / 255.0
    x_test = tf.cast(x_test, tf.float32) / 255.0

    # flatten labels to shape (N,)
    y_train = tf.cast(y_train.flatten(), tf.int32)
    y_test = tf.cast(y_test.flatten(), tf.int32)

    # return
    return x_train, y_train, x_test, y_test, classnames


def standardize(x_train, x_test, eps=1e-10):
    '''Standardizes the image features using the global RGB triplet method.

    Parameters:
    -----------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features (preliminary).
    x_test: tf.constant. tf.float32s. shape=(N_test, I_y, I_x, n_chans).
        Test set features.

    Returns:
    --------
    tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Standardized training set features (preliminary).
    tf.constant. tf.float32s. shape=(N_test, I_y, I_x, n_chans).
        Standardized test set features (preliminary).
    '''

    #need to standardize across full img and for all imgs
    rgb_means = tf.reduce_mean(x_train, axis=[0,1,2], keepdims = True) # getting the mean for each color channel
    rdb_stds = tf.math.reduce_std(x_train, axis=[0,1,2], keepdims = True) #stds for all color channels
    x_train_standardized = (x_train - rgb_means) / (rdb_stds + eps)
    x_test_standardized = (x_test - rgb_means) / (rdb_stds + eps)
        
    return x_train_standardized, x_test_standardized


def train_val_split(x_train, y_train, val_prop=0.1):
    '''Subdivides the preliminary training set into disjoint/non-overlapping training set and validation sets.
    The val set is taken from the end of the preliminary training set.

    Parameters:
    -----------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features (preliminary).
    y_train: tf.constant. tf.int32s. shape=(N_train,).
        Training set class labels (preliminary).
    val_prop: float.
        The proportion of preliminary training samples to reserve for the validation set. If the proportion does not
        evenly subdivide the initial N, the number of validation set samples should be rounded to the nearest int.

    Returns:
    --------
    tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features.
    tf.constant. tf.int32s. shape=(N_train,).
        Training set labels.
    tf.constant. tf.float32s. shape=(N_val, I_y, I_x, n_chans).
        Validation set features.
    tf.constant. tf.int32s. shape=(N_val,).
        Validation set labels.
    '''
    # calc num of validation samples
    val_size = int(len(x_train) * val_prop + 0.5)  # round to nearest int
    
    # split data
    x_val = x_train[-val_size:]  
    y_val = y_train[-val_size:]
    
    x_train_new = x_train[:-val_size]  # remaining samples for training
    y_train_new = y_train[:-val_size]
    
    return x_train_new, y_train_new, x_val, y_val


def get_dataset(name, standardize_ds=True, val_prop=0.1):
    '''Automates the process of loading the requested dataset `name`, standardizing it (optional), and create the val
    set.

    Parameters:
    -----------
    name: str.
        Name of the dataset that should be loaded. Support options in Project 1: 'cifar10', 'mnist'.
    standardize_ds: bool.
        Should we standardize the dataset?
    val_prop: float.
        The proportion of preliminary training samples to reserve for the validation set. If the proportion does not
        evenly subdivide the initial N, the number of validation set samples should be rounded to the nearest int.

    Returns:
    --------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        The training set.
    y_train: tf.constant. tf.int32s.
        The training set int-coded labels.
    x_val: tf.constant. tf.float32s. shape=(N_val, I_y, I_x, n_chans).
        Validation set features.
    y_val: tf.constant. tf.int32s. shape=(N_val,).
        Validation set labels.
    x_test: tf.constant. tf.float32s.
        The test set.
    y_test: tf.constant. tf.int32s.
        The test set int-coded labels.
    classnames: Python list. strs. len(classnames)=num_unique_classes.
        The human-readable string names of the classes in the dataset. If there are 10 classes, len(classnames)=10.
    '''
    # load dataset using load_dataset
    x_train, y_train, x_test, y_test, classnames = load_dataset(name)
    
    # standardize dataset if required
    if standardize_ds:
        x_train, x_test = standardize(x_train, x_test)

    # create val set using train_val_split
    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, val_prop=val_prop)

    # return
    return x_train, y_train, x_val, y_val, x_test, y_test, classnames

