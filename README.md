# ResUNet-CMB
This repository contains the implementation of the ResUNet-CMB convolutional neural network from the papers [Reconstructing Patchy Reionization with Deep Learning](https://arxiv.org/abs/2101.01214) and [Reconstructing Cosmic Polarization Rotation with ResUNet-CMB](https://arxiv.org/abs/2109.09715). The ResUNet-CMB network is designed for the simultaneous reconstruction of multiple fields that distort the primary CMB fluctuations such as lensing convergence, patchy reionization, and cosmic birefringence from input CMB polarization maps Q and U.

This repository contains code for the production of the CMB simulations used for training and making predictions (data pipeline) and the code used for training the network.

## Requirements
To run the data pipeline requires:
* Python>=2.7
* [orphics](https://github.com/msyriac/orphics)
* [pixell](https://github.com/simonsobs/pixell)

detailed requirements are located in datapipe_requirements.txt. Reproducing the exact pipeline used for the papers requires a modified version of [orphics](https://github.com/EEmGuzman/orphics) and [pixell](https://github.com/EEmGuzman/pixell).

To train the network requires:
* Python>=3.7
* TensorFlow>=2.0

detailed requirements are located in tensorflowtraining_requirements.txt.

## Architecture
![](/images/ResUNet-CMB_arch.png?raw=true)
The ResUNet-CMB architecture is a modified version of the ResUNet from the paper [DeepCMB: Lensing Reconstruction of the Cosmic Microwave Background with Deep Neural Networks](https://arxiv.org/abs/1810.01483).

The ResUNet-CMB network has around 5 million trainable parameters. Numbers below the convolutional layers represent the number of filters for that layer. No residual connections are shown. The first residual connection takes the concatenated input tensor and adds it element-wise to the output of the second convolution block. The next residual connection takes the input to the second convolution block and adds it to the output of the fourth. The pattern continues until the final two convolution blocks where no residual connection is used. All residual connections contain a batch normalization layer.

## Data Pipeline
To recreate the data used in the paper, the [modified forked version](https://github.com/EEmGuzman/orphics) of orphics is required. The default values for the patchy reionization spectrum listed in datapipe.py are the values used for the paper. CAMB spectra files are copied from [quicklens](https://github.com/dhanson/quicklens). The pipeline can be run in a Python 2 environment. 

To run the pipleline use

        python datapipeline.py
while in the datapipe directory. A 'map_sets32.npz' file with all the data will be saved to the working directory along with information about the applied window. If you do not wish to use the data pipeline provided, you must provide a '.npz' file with the arrays of maps you want to use to train the network.

The naming convention used for the simulated CMB maps can be found in the datapipeline.py file. It is important to note two subtleties:
- The '_obs' addition to each map name indicates it is the observed map. It is a map that has been rotated by cosmic birefringence, modulated by patchy reionization and then lensed. Noise and a taper were also applied.
- The '_prim' addition to each map name indicates it is the primordial map with a taper applied.

## Training
### Based on 3-output ResUNet-CMB model from [the patchy reionization deep learning paper](https://arxiv.org/abs/2101.01214)
The main.py file executes the preprocessing of the data, training of the network, and making and saving predictions.

A configuration file with various hyperparameters and settings is used to train the network. Within the configuration file there are three dictionaries set that determine which maps from the map_sets32.npz file get used as the input features and which are the labels:
- __feat_used__ determines how the input maps get assigned. The keys are the names of the input Keras tensors. The ResUNet-CMB network has two input Keras tensors that are named "qlen" and "ulen". You may change these names in the resunet.py file. The values assigned (key:value) in the dictionary are the names of the arrays saved in the provided '.npz' file.
- __target_used__ determines how the labels get assigned to the network output. The keys are the names of the final layers of each output of the network. For ResUNet-CMB the three outputs are named "kappa", "unle", and "tau". The values assigned (key:value) in the dictionary are the names of the arrays in the '.npz' file that should be used for each output.
- __trace_used__ is a list of array names from the '.npz' file that will be preprocessed and saved along with the feat_used and target_used maps but are not used during training.

Similar to what is stated in the data pipeline section, here 'obs' refers to the observed maps. The 'prim' refers to the primordial or truth maps.

Another parameter of interest in the configuration file is the "model" one. If you want to define and train another network, you can create a new model in resunet.py. Afterwards, to train your custom network, write its name as the value for the "model" parameter.

The hyperparameter values used for training the networks in the paper are the default values in the config.json file. It is important to remember to set the data paths before training.

After setting the parameters in the configuration file use

    python main.py "config.json"

to train the network. The config_4output.json file is the configuration file used for training the 4-output network from [Reconstructing Cosmic Polarization Rotation with ResUNet-CMB](https://arxiv.org/abs/2109.09715).

## Evaluation and Results
### This section is based on [Reconstructing Patchy Reionization with Deep Learning](https://arxiv.org/abs/2101.01214)
The network takes modulated and lensed Q and U CMB polarization maps (128 x 128 pixels) as input and produces three outputs:
- An estimate of the patchy reionization field. This is a map of the optical depth variation on the sky.
- Lensing convergence field
- Primordial E-mode CMB polarization map.

Example ResUNet-CMB predictions from fully trained networks for two noise levels compared to the true maps are shown below. Residual maps are calculated as the true map minus the prediction.

![](/images/ResUNet-CMB_map_results.png?raw=true)

In this repository we include the fully trained ResUNet-CMB noiseless network that produced some of the results seen above. An example of how to load the pre-trained network and make predictions to achieve similar results is provided in a [Jupyter Notebook](https://github.com/EEmGuzman/resunet-cmb/blob/master/trained_networks/example_evaluate_results.ipynb).

It is important to note the pre-trained network included was trained with an extra metric not currently present in the main file, the coefficient of determination.

## TODO
- Add code used for patchy reionization quadratic estimator [symlens](https://github.com/simonsobs/symlens) implementation
- Update data pipeline to ensure compatibility with Python 3. Create a single environment that works for both the data pipeline and the network training.
- Convert orphics lensing quadratic estimator in data pipeline to an implementation with symlens.
- Modify README with more information about the 4-output network.
