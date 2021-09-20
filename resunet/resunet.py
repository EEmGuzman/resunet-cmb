#!/usr/bin/env python3

import numpy as np
import tensorflow.keras as keras

from tensorflow.keras.layers import Input

## Basic blocks ##

def skip_connection(x, xskip):
    """
    A long skip connection that concatenates output from an encoding layer to a
    layer in the decoder along the channel dimension.

    Parameters
    ----------
    x :
        Input tensor from the decoder phase.
    xskip :
        Tensor from the encoder phase.
    """

    con = keras.layers.Concatenate(axis=3)([x, xskip])
    return con

def convolution_block(x, filters, kernel_size, padding, strides, act, dropval, name=None, first_block=False, upsample=False):
    """
    Fundamental convolution block.

    This block has three possible configurations.
    1) conv -> activation -> batchnorm
    2) dropout -> conv -> activation -> batchnorm
    3) upsample -> dropout -> conv -> activation -> batchnorm

    This convolution block is used in both the encoder and decoder of the network.
    Upsampling in the decoder is done with a dedicated layer. Downsampling is done
    with a convolutional layer.

    Parameters
    ----------
    x :
        Input tensor.
    filters : int
        Number of activation maps in the convolutiional layer.
    kernel_size : int
        Dimension of filter in convolutional layer. Only a single integer is accepted.
    padding : str
        Convolutional layer padding.
    strides : int
        Stride number for convolutional layer.
    act :
        Activation function
    dropval :
        Drop rate for droupout layer.
    name : str
        Name that will be applied to only the batch normalization layer.
    first_block : boolean
        If True, dropout layer will be omitted from the block.
    upsample : boolean
        If True, a 2D upsampling layer will be placed before the dropout layer.
        This enables configuration (3).

    Returns
    -------
    con :
        Output tensor.
    """

    if first_block:
        con = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
    elif upsample:
        con = keras.layers.UpSampling2D(strides)(x)
        con = keras.layers.Dropout(dropval)(con)
        con = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=1)(con)
    else:
        con = keras.layers.Dropout(dropval)(x)
        con = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(con)
    con = keras.layers.Activation(act)(con)
    con = keras.layers.BatchNormalization(name=name)(con)
    return con

def residual_block(x, filters, kernel_size, padding, strides, act, dropval, name=[None, None], first_block=False, upsample=[False, False, False], filters_changed=False, mid_skip=None):
    """
    Parameters
    ----------
    x :
        Input tensor.
    filters : list
        A list with two elements. The items in the list, filters[0] and filters[1],
        are the number of filters in the first and second convolution block,
        respectively.
    kernel_size : int
        Dimension of filter in convolutional layer. Only a single integer is accepted.
        The kernel_size provided is applied to both convolutional blocks.
    padding : str
        Convolutional layer padding. Given string is applied to both convolution
        blocks.
    strides : list
        A list with three elements. Two items in the list, strides[0] and strides[1],
        are the stride values for the first and second convolutional block, respectively.
        The third item, stride[2], is the stride that will be applied to the convolutional
        layer or upsampling layer in the residual connection.
    act : list
        A list with two elements. The items in the list, act[0] and act[1], are the
        activation functions for the first and second convolutional block, respectively.
    dropval :
        Drop rate for droupout layer. Given value is applied to both convolution
        blocks.
    name : list
        A list with two elements. The items in the list, name[0] and name[1], are the
        names that will be applied to only the batch normalization layer of the first
        and second convolution block, respectively.
    first_block : boolean
        If True, dropout layer will be omitted from the first convolution block.
    upsample : list
        A list with three elements. Two items in the list, upsample[0] and upsample[1],
        indicate if an 2D upsampling layer should be added before the dropout layer in
        the first and second convolution block, respectively. The third item, upsample[2],
        indicates if a 2D upsampling layer should be used. All elements in the list
        must be boolean.
    filters_changed : boolean
        Set to True if the input to the residual block and the output have a different
        number of channels (filters). If True, a convolutional layer, if not already present
        due to automatic triggers, will be placed in the residual connection to fix the channel
        dimension.
    mid_skip :
        A keras tensor. This keras tensor will be concatenated with the output of the
        first convolution block.

    Returns
    -------
    con0 :
        Output tensor of the first convolution block.
    con1 :
        Output tensor of the second convolution block. This is before the residual
        connection is applied.
    resoutput :
        Output tensor from running the entire residual block. Residual connection is applied.
    """

    con0 = convolution_block(x, filters[0], kernel_size, padding, strides[0], act[0], dropval, name=name[0], first_block=first_block, upsample=upsample[0])
    if mid_skip is not None:
        con0 = skip_connection(con0, mid_skip)
    con1 = convolution_block(con0, filters[1], kernel_size, padding, strides[1], act[1], dropval, name=name[1], upsample=upsample[1])

    if strides[2] != 1 and upsample[2] == True:
        x = keras.layers.UpSampling2D(strides[2])(x)
        if filters_changed == True:
            x = keras.layers.Conv2D(filters[1], kernel_size=1, padding=padding, strides=1)(x)
    elif (strides[2] != 1 and upsample[2] == False):
        x = keras.layers.Conv2D(filters[1], kernel_size=1, padding=padding, strides=strides[2])(x)
    elif filters_changed == True:
        x = keras.layers.Conv2D(filters[1], kernel_size=1, padding=padding, strides=1)(x)
    rescon = keras.layers.BatchNormalization()(x)

    resoutput = keras.layers.Add()([rescon, con1])
    return con0, con1, resoutput

def ResUNet_CMB(params):
    """
    ResUNet-CMB Network

    Network used in "Reconstructing Patchy Reionization with Deep Learning."

    Parameters
    ----------
    params:
        params is a container for the variables defined in the configuration file.
        An instance of class resunet.utils.Params.

    Returns
    -------
    model :
        model object.
    """

    input_img1 = Input(shape=(params.imagesize, params.imagesize, 1), dtype=np.float32, name="qlen")
    input_img2 = Input(shape=(params.imagesize, params.imagesize, 1), dtype=np.float32, name="ulen")

    # encoder
    enc_0 = keras.layers.Concatenate(axis=3)([input_img1, input_img2])
    enc_1 = residual_block(enc_0, [64,64], 5, "same", [1,1,1], ["selu","selu"], params.dropval, first_block=True, filters_changed=True)[2]
    enc_2 = residual_block(enc_1, [64,128], 5, "same", [1,2,2], ["selu","selu"], params.dropval, filters_changed=True)
    enc_3 = residual_block(enc_2[2], [128,128], 5, "same", [1,1,1], ["selu","selu"], params.dropval)[2]

    # bridge between encoder and decoder
    enc_dec = residual_block(enc_3, [256,128], 5, "same", [2,2,1], ["selu","selu"], params.dropval, upsample=[False, True, False])[2]

    lskip1 = skip_connection(enc_dec, enc_3)
    dec_1 = residual_block(lskip1, [128,128], 5, "same", [1,1,1], ["selu","selu"], params.dropval, filters_changed=True)[2]
    dec_2 = residual_block(dec_1, [64, 64], 5, "same", [2,1,2], ["selu","selu"], params.dropval, upsample=[True, False, True], mid_skip=enc_2[0], filters_changed=True)[2]

    # Block all branches use for final residual connection
    dec_3 = convolution_block(dec_2, 64, 5, "same", 1, "selu", params.dropval)

    # kappa branch
    kappa_1 = convolution_block(dec_3, 64, 5, "same", 1, "selu", params.dropval)
    kappa_res_1 = keras.layers.BatchNormalization()(dec_2)
    kappa_1 = keras.layers.Add()([kappa_res_1, kappa_1])
    kappa_2 = convolution_block(kappa_1, 64, 5, "same", 1, "selu", params.dropval)
    kappa_3 = convolution_block(kappa_2, 1, 5, "same", 1, "linear", params.dropval, name="kappa")

    # primordial E branch
    unle_1 = convolution_block(dec_3, 64, 5, "same", 1, "selu", params.dropval)
    unle_res_1 = keras.layers.BatchNormalization()(dec_2)
    unle_1 = keras.layers.Add()([unle_res_1, unle_1])
    unle_2 = convolution_block(unle_1, 64, 5, "same", 1, "selu", params.dropval)
    unle_3 = convolution_block(unle_2, 1, 5, "same", 1, "linear", params.dropval, name="unle")

    # tau branch
    tau_1 = convolution_block(dec_3, 64, 5, "same", 1, "selu", params.dropval)
    tau_res_1 = keras.layers.BatchNormalization()(dec_2)
    tau_1 = keras.layers.Add()([tau_res_1, tau_1])
    tau_2 = convolution_block(tau_1, 64, 5, "same", 1, "selu", params.dropval)
    tau_3 = convolution_block(tau_2, 1, 5, "same", 1, "linear", params.dropval, name="tau")

    model = keras.models.Model(inputs=[input_img1, input_img2], outputs=[tau_3, unle_3, kappa_3])
    return model
    
def ResUNet_CMB_4out(params):
    """
    ResUNet-CMB 4-output Network

    Network used in "Reconstructing Cosmic Polarization Rotation with ResUNet-CMB"

    Parameters
    ----------
    params:
        params is a container for the variables defined in the configuration file.
        An instance of class resunet.utils.Params.

    Returns
    -------
    model :
        model object.
    """

    input_img1 = Input(shape=(params.imagesize, params.imagesize, 1), dtype=np.float32, name="qlen")
    input_img2 = Input(shape=(params.imagesize, params.imagesize, 1), dtype=np.float32, name="ulen")

    # encoder
    enc_0 = keras.layers.Concatenate(axis=3)([input_img1, input_img2])
    enc_1 = residual_block(enc_0, [64,64], 5, "same", [1,1,1], ["selu","selu"], params.dropval, first_block=True, filters_changed=True)[2]
    enc_2 = residual_block(enc_1, [64,128], 5, "same", [1,2,2], ["selu","selu"], params.dropval, filters_changed=True)
    enc_3 = residual_block(enc_2[2], [128,128], 5, "same", [1,1,1], ["selu","selu"], params.dropval)[2]

    # bridge between encoder and decoder
    enc_dec = residual_block(enc_3, [256,128], 5, "same", [2,2,1], ["selu","selu"], params.dropval, upsample=[False, True, False])[2]

    lskip1 = skip_connection(enc_dec, enc_3)
    dec_1 = residual_block(lskip1, [128,128], 5, "same", [1,1,1], ["selu","selu"], params.dropval, filters_changed=True)[2]
    dec_2 = residual_block(dec_1, [64, 64], 5, "same", [2,1,2], ["selu","selu"], params.dropval, upsample=[True, False, True], mid_skip=enc_2[0], filters_changed=True)[2]

    # Block all branches use for final residual connection
    dec_3 = convolution_block(dec_2, 64, 5, "same", 1, "selu", params.dropval)

    # kappa branch
    kappa_1 = convolution_block(dec_3, 64, 5, "same", 1, "selu", params.dropval)
    kappa_res_1 = keras.layers.BatchNormalization()(dec_2)
    kappa_1 = keras.layers.Add()([kappa_res_1, kappa_1])
    kappa_2 = convolution_block(kappa_1, 64, 5, "same", 1, "selu", params.dropval)
    kappa_3 = convolution_block(kappa_2, 1, 5, "same", 1, "linear", params.dropval, name="kappa")

    # primordial E branch
    unle_1 = convolution_block(dec_3, 64, 5, "same", 1, "selu", params.dropval)
    unle_res_1 = keras.layers.BatchNormalization()(dec_2)
    unle_1 = keras.layers.Add()([unle_res_1, unle_1])
    unle_2 = convolution_block(unle_1, 64, 5, "same", 1, "selu", params.dropval)
    unle_3 = convolution_block(unle_2, 1, 5, "same", 1, "linear", params.dropval, name="unle")

    # tau branch
    tau_1 = convolution_block(dec_3, 64, 5, "same", 1, "selu", params.dropval)
    tau_res_1 = keras.layers.BatchNormalization()(dec_2)
    tau_1 = keras.layers.Add()([tau_res_1, tau_1])
    tau_2 = convolution_block(tau_1, 64, 5, "same", 1, "selu", params.dropval)
    tau_3 = convolution_block(tau_2, 1, 5, "same", 1, "linear", params.dropval, name="tau")

    # alpha branch
    cbf_1 = convolution_block(dec_3, 64, 5, "same", 1, "selu", params.dropval)
    cbf_res_1 = keras.layers.BatchNormalization()(dec_2)
    cbf_1 = keras.layers.Add()([cbf_res_1, cbf_1])
    cbf_2 = convolution_block(cbf_1, 64, 5, "same", 1, "selu", params.dropval)
    cbf_3 = convolution_block(cbf_2, 1, 5, "same", 1, "linear", params.dropval, name="cbf")

    model = keras.models.Model(inputs=[input_img1, input_img2], outputs=[tau_3, unle_3, kappa_3, cbf_3])
    return model

