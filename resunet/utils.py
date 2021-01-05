#/usr/bin/env python3

import os
import json
import argparse
import tensorflow.keras as keras
import numpy as np
import scipy.ndimage
import sklearn.model_selection as model_selection

from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

def load_model(fname, weights):
    """
    Loads keras model from saved files.

    Parameters
    ----------
    fname : str
        JSON file name containing saved model architecture.
    weights : str
        h5 file name containing saved model weights.

    Returns
    -------
    model :
        Previously trained model.
    """

    with open(str(fname), 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(str(weights))

    return model

def load_data(arrname, cmbmaps):
    """
    Loads data from a '.npz' file into a dictionary.

    The data is loaded into a dictionary with the array names as the keys and the
    arrays as the values.

    Parameters
    ----------
    arrname : str
        '.npz' file with saved arrays that are named.
    cmbmaps :
        List of CMB maps to retrieve from arrname.

        If used with the ResUNet-CMB data pipeline, format must be '{map letter}_{len or unl}'.
        In the case of phi or kappa: '{tru or rec}_{phi or kappa}'.
        Example list format: ['q_unl', 'e_len', 'unl_e']

        If used for general purposes then cmbmaps is a list of the array names, as strings, in the
        '.npz' file that will be retrieved.

    Returns
    -------
    datadic :
        Dictionary with data from arrname. Key values of saved dictionary are the names
        from cmbmaps list which are the names of the saved arrays. The values in the
        dictionary are the retrieved arrays.
    """

    datadic = {}
    with np.load(arrname) as data:
        for i in cmbmaps:
            datadic[i] = data[i]
    return datadic

def make_predictions(model, i_maps_n, batch_size, fname):
    """
    Makes predictions with fully trained keras model.

    Parameters
    ----------
    model :
        Trained keras model object.
    i_maps_n :
        Dictionary with named inputs (from keras tensor initial definition) and corresponding
        maps to be used for predictions.
    batch_size : int
        Number of predictions to be made in each iteration. Normally set to be the same
        as the batch size used in training.
    fname : str
        File name for predictions to be saved as.

    Returns
    -------
    '.npz' file with the array of predictions is saved to the working directory.

    If used with ResUNet-CMB,
    'kappa_pred': Kappa predictions.
    'unle_pred' : Primordial E predictions.
    'tau_pred' : Tau predictions.

    If used with a different model, name convention for predictions is
    '{final network layer name}_pred'.
    """

    predarr = model.predict(i_maps_n, batch_size=batch_size, verbose=1)
    datadic = {key.name.split('/')[0] + '_pred': predarr[i] for i, key in enumerate(model.outputs)}
    np.savez(fname, **datadic)

def preprocess(params, unproc, submean=False, imagesize=128):
    """
    Function to standardize image data before being used with the deep learning
    network.

    Performs a standardizing procedure on image data that has already been split
    into training, validation, and test sets using the resunet.utils.split_dataset function.
    Performs the procedure:
    X_\mathrm{Processed} = frac{( X - \overline{X}_{Training})}{\sigma_{Training}}.

    Where X is a map from any map type. For more information on the process refer to
    "Reconstructing Patchy Reinoization with Deep Learning."

    Parameters
    ----------
    params :
        Class object resunet.utils.Params. Params is a container for the variables
        defined in the configuration file.
    unproc :
        Dictionary of unprocessed data. Has a format where key is the name of map type and
        the value is the array of all maps of that key type. This is usually the output of
        the resunet.utils.split_dataset function.
    submean : boolean
        If true, the mean of the maps, across the set of each map type, will be calculated
        and subtracted before finding the standard deviation.
    imagesize : int
        Pixel image size of all the maps.

    Returns
    -------
    map_stds.npz :
        Saves standard deviations calculated to a '.npz' file in the working directory.
    map_means.npz :
        Saves means calculated to a '.npz' file in the working directory. This is only done
        if submean is True.
    unproc :
        Dictionary of processed, standardized, CMB maps ready for use with the deep learning
        network.
    """

    stdvals = {}
    meanvals = {}
    for i in [params.feat_used, params.target_used]:
        for key, value in i.items():
            # both values will be calculated from the training data set
            if submean:
                meanvals[value+"_mean"] = np.mean(unproc[value+"_train"])
                stdvals[value+"_std"] = np.std(unproc[value+"_train"] - meanvals[value+"_mean"])
            else:
                stdvals[value+"_std"] = np.std(unproc[value+"_train"])

    np.savez(
        os.path.join(params.output_dir, "map_stds"), **stdvals)
    np.savez(
        os.path.join(params.output_dir, "map_means"), **meanvals)

    # standardizing data
    for i in [params.feat_used, params.target_used]:
        for key, value in i.items():
            unproc[value+"_train"] = (unproc[value+"_train"] - meanvals[value+"_mean"]) / stdvals[value+"_std"]
            unproc[value+"_val"] = (unproc[value+"_val"] - meanvals[value+"_mean"]) / stdvals[value+"_std"]
            unproc[value+"_test"] = (unproc[value+"_test"] - meanvals[value+"_mean"]) / stdvals[value+"_std"]

    for key, value in unproc.items():
        unproc[key] = np.reshape(unproc[key], (value.shape[0], imagesize, imagesize, 1))

    return unproc


def split_dataset(datadic, datanames, tr=0.8, va=0.1, te=0.1, rseed=43):
    """
    Splits and shuffles data sets into a training, validation, and test set.
    Function based on code by Jorge Barrios from
    https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn 

    Parameters
    ----------
    datadic :
        Dictionary of data to split. The key is the map type while the value is
        an array of all the maps (images). This is usually the output from the
        resunet.utils.load_data function.
        Example format:
            datadic = {"q_len" : array_of_qlen_maps}
    datanames :
        List of the keys, as strings, of the datadic dictionary. In the example
        given in datadic description, this would be ["q_len"].
    tr : float
        Training split percentage as a decimal.
    va : float
        Validation split percentage as a decimal.
    te : float
        Test split percentage as a decimal.
    rseed : int
        NumPy random seed used when shuffling data while splitting.

    Returns
    -------
    iodata :
        Dictionary of split and shuffled data. For each item in the dataname list,
        three new key, value pairs are created and saved. One pair for each of the training,
        test, and validation set,
        Example keys:
            'q_len_train'
            'q_len_val'
            'q_len_test'
    """

    val_remain = va / (1 - te)

    iodata = {}
    for i in datanames:
        te_and_re = model_selection.train_test_split(
                datadic[i], test_size=te, random_state=rseed)
        tr_and_val = model_selection.train_test_split(
                te_and_re[0], test_size=val_remain, random_state=rseed+1)
        iodata[i+'_test'] = te_and_re[1]
        iodata[i+'_val'] = tr_and_val[1]
        iodata[i+'_train'] = tr_and_val[0]

    return iodata

def argpsetup():
    parser = argparse.ArgumentParser("Trains a ResUNet model for CMB distortion field predictions.")
    parser.add_argument('config', help="JSON configuration file path. Must be a string.")
    args = parser.parse_args()

    return args

class Params:
    """
    Class for storing network parameters from configuration JSON file.
    """
    def __init__(self, jfilepath):
        self.get_params(jfilepath)

    def get_params(self, jfilepath):
        with open(jfilepath, "r") as f:
            params = json.load(f)
        for key in params:
            setattr(self, key, params[key])
