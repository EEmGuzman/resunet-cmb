#!/usr/bin/env python3

import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import resunet.resunet as rsn
import resunet.utils as rutils

from tensorflow.keras.models import model_from_json

## Set up ##
# getting parameters from JSON file and creating output directory structure
args = rutils.argpsetup()
params = rutils.Params(args.config)
os.makedirs(params.output_dir, exist_ok=True)

complete_list = list(params.feat_used.values()) + list(params.target_used.values()) + params.trace_used
datad = rutils.load_data(params.data_path, complete_list)

unproc = rutils.split_dataset(
    datad, complete_list, tr=params.training_split, va=params.val_split, te=params.test_split)
iodata = rutils.preprocess(params, unproc, submean=True, imagesize=params.imagesize)

# ensure correct tf session
K.clear_session()
K.set_floatx('float32')

if params.allocate_vram_an == True:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

## building model ##
model = getattr(rsn, params.model_name)(params)
optimz = getattr(keras.optimizers, params.optimizer)(lr=params.learning_rate)

# creating callbacks for training
halt = keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=params.haltpatience, restore_best_weights=True)
halvelr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=params.lrfactordelta, patience=params.lrpatience)
csv_logger = keras.callbacks.CSVLogger(
    os.path.join(params.output_dir, 'training.log'), separator=',', append=False)

callbacks = [halt, halvelr, csv_logger]

if params.tensboard_log == True:
    logdir=os.path.join(params.output_dir, 'logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    callbacks = [halt, halvelr, csv_logger, tensorboard_callback]

model.compile(optimizer=optimz, loss=params.loss)

model.summary()
keras.utils.plot_model(
    model, to_file=os.path.join(params.output_dir, 'model_arch.png'), show_layer_names=True, show_shapes=True)

## training model ##
x_train = {}
y_train = {}
x_val = {}
y_val = {}
x_test = {}
for key, value in params.feat_used.items():
    x_train[key] = iodata[value + "_train"]
    x_val[key] = iodata[value + "_val"]
    x_test[key] = iodata[value + "_test"]
for key, value in params.target_used.items():
    y_train[key] = iodata[value + "_train"]
    y_val[key] = iodata[value + "_val"]

history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=params.batch_size,
        epochs=params.epochs,
        callbacks=callbacks,
        verbose=1)

model.save(
    os.path.join(params.output_dir, 'complete_model_valloss-{0:.4f}.h5').format(min(history.history["val_loss"])))

model.save_weights(
    os.path.join(params.output_dir, 'model_weights_valloss-{0:.4f}.h5').format(min(history.history['val_loss'])))

with open(os.path.join(params.output_dir, 'model_architecture.json'), 'w') as f:
    f.write(model.to_json())

# Predictions saved as 2 arrays: kappa_pred and unle_pred
print('Making predictions on test data...')
rutils.make_predictions(model, x_test, params.batch_size, os.path.join(params.output_dir, "predictions"))

# getting all test data into a single dictionary
testdic = {}
for key, value in iodata.items():
    if key.endswith('test'):
        testdic[key] = value

np.savez(
    os.path.join(params.output_dir, "test_data"), **testdic)
