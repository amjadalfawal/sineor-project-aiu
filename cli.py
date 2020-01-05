from preprocessing import process_text_based_data
import torch
import torch.utils.data
import h5py
import torch.autograd as autograd
import torch.optim as optim
import matplotlib
import numpy as np
import time

import matplotlib.pyplot as plt
from drawnow import drawnow
from models import ExampleModel
import argparse
from helper import contruct_data_loader_from_disk, set_protien_experiments_id, write_out, test_eval_model, save_model_on_disk_torch_version, draw_plot, logs
print("--------------------------------------------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------- protin senior -------------------------------------------------")
process_text_based_data()
print("--------------------------------------------------------------------------------------------------------------------------------")






#flags const variables 
live_plot = False
eval_interval = 10
learning_rate = 0.001
minimum_updates = 100
minibatch_size = 640
 
use_gpu = True
if torch.cuda.is_available():
    write_out("CUDA is available, using GPU")
    use_gpu = True


if not live_plot:
    print("Live plot deactivated, see output folder for plot.")
    matplotlib.use('Agg')

training_file = "data/preprocessed/sample.hdf5"
validation_file = "data/preprocessed/sample.hdf5"
testing_file = "data/preprocessed/sample.hdf5"


def train_model(data_set_identifier, train_file, val_file, learning_rate, minibatch_size):
    set_protien_experiments_id(data_set_identifier, learning_rate, minibatch_size)

    train_loader = contruct_data_loader_from_disk(train_file, minibatch_size)
    validation_loader = contruct_data_loader_from_disk(val_file, minibatch_size)
    validation_dataset_size = validation_loader.dataset.__len__()

    model = ExampleModel(9, "ONEHOT", minibatch_size, use_gpu=use_gpu) 

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # plot settings
    if live_plot:
        plt.ion()
    fig = plt.figure()
    sample_num = list()
    train_loss_values = list()
    validation_loss_values = list()

    best_model_loss = 1.1
    best_model_minibatch_time = None
    best_model_path = None
    stopping_condition_met = False
    minibatches_proccesed = 0

    while not stopping_condition_met:
        optimizer.zero_grad()
        model.zero_grad()
        loss_tracker = np.zeros(0)
        for minibatch_id, training_minibatch in enumerate(train_loader, 0):
            minibatches_proccesed += 1
            primary_sequence, tertiary_positions, mask = training_minibatch
            start_compute_loss = time.time()
            loss = model.neg_log_likelihood(primary_sequence, tertiary_positions)
            write_out("Train loss:", float(loss))
            start_compute_grad = time.time()
            loss.backward()
            loss_tracker = np.append(loss_tracker, float(loss))
            end = time.time()
            write_out("Loss time:", start_compute_grad-start_compute_loss, "Grad time:", end-start_compute_grad)
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

            # for every eval_interval samples, plot performance on the validation set
            if minibatches_proccesed % eval_interval == 0:

                train_loss = loss_tracker.mean()
                loss_tracker = np.zeros(0)
                validation_loss, data_total = test_eval_model(validation_loader, model)
                if validation_loss < best_model_loss:
                    best_model_loss = validation_loss
                    best_model_minibatch_time = minibatches_proccesed
                    best_model_path = save_model_on_disk_torch_version(model)

                write_out("Validation loss:", validation_loss, "Train loss:", train_loss)
                write_out("Best model so far (label loss): ", validation_loss, "at time", best_model_minibatch_time)
                write_out("Best model stored at " + best_model_path)
                write_out("Minibatches processed:",minibatches_proccesed)
                sample_num.append(minibatches_proccesed)
                train_loss_values.append(train_loss)
                validation_loss_values.append(validation_loss)
                if live_plot:
                    drawnow(draw_plot(fig, plt, validation_dataset_size, sample_num, train_loss_values, validation_loss_values))

                if minibatches_proccesed > minimum_updates and minibatches_proccesed > best_model_minibatch_time * 2:
                    stopping_condition_met = True
                    break
    logs(best_model_loss)
    return best_model_path

start = time.time()
train_model_path = train_model("TRAIN", training_file, validation_file, learning_rate, minibatch_size)
end = time.time()
print( "Time elapsed: "  , end - start)

print(train_model_path)