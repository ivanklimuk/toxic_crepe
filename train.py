import utils
from constants import *
import torch
import numpy as np
from model.crepe import Crepe, metrics, confusion_matrix, roc
from model.dataloader import DataLoader, data_iterator
from evaluate import evaluate
import logging
import os
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')


def train(model, optimizer, loss_function, data_iterator, metrics, num_steps, save_summary_steps=10):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_function: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        num_steps: (int) number of batches to train on, each of size batch_size
        save_summary_steps: (int) the amount of steps after which to save the summary
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_average = utils.RunningAverage()

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        # fetch the next training batch
        train_batch, labels_batch = next(data_iterator)

        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_function(output_batch, labels_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if (i + 1) % save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](labels_batch, np.round(output_batch)) for metric in metrics}
            summary_batch['loss'] = loss.data
            summ.append(summary_batch)

        # update the average loss
        loss_average.update(loss.data)
        t.set_postfix(loss='{:05.3f}'.format(loss_average()))

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " - ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_data, val_data, epochs, batch_size, optimizer, loss_function, metrics, model_path, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validaion data with keys 'data' and 'labels'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_function: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        model_path: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    '''
    if restore_file is not None:
        restore_path = os.path.join(model_path, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    '''

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, epochs))

        # Compute number of batches in one epoch (one full pass over the training set)
        num_steps = (len(train_data['data']) + 1) // batch_size
        train_data_iterator = data_iterator(data=train_data['data'], labels=train_data['labels'], batch_size=batch_size, shuffle=True)
        train(model, optimizer, loss_function, train_data_iterator, metrics, num_steps)

        # Calculate the train confusion matrix and ROC AUC score
        roc_confusion(model=model, data=train_data, batch_size=batch_size)

        # Evaluate for one epoch on validation set
        num_steps = (len(val_data['data']) + 1) // batch_size
        val_data_iterator = data_iterator(data=val_data['data'], labels=val_data['labels'], batch_size=batch_size, shuffle=False)
        val_metrics = evaluate(model, loss_function, val_data_iterator, metrics, num_steps)

        # Calculate the validation confusion matrix and ROC AUC score
        roc_confusion(model=model, data=val_data, mode='Val', batch_size=batch_size)

        val_acc = val_metrics['f1']
        is_best = val_acc > best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_path)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best F1 score!")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_path, EXPERIMENT_PREFIX + "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_path, EXPERIMENT_PREFIX + "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


def roc_confusion(model, data, mode='Train', batch_size=BATCH_SIZE):
    batch_outputs = [model(torch.from_numpy(data['data'][i: i + batch_size]).float()).detach().numpy()
                                                                for i in range(0, len(data['data']), batch_size)]
    '''batch_outputs = []
    for i in range(0, len(data['data']), batch_size):
        prediction = model(torch.from_numpy(data['data'][i: i + batch_size]).float())
        batch_outputs.append(prediction.detach().numpy())'''
    output = np.vstack(batch_outputs)
    confusion = confusion_matrix(data['labels'], output)
    roc_auc_score = roc(data['labels'], output)
    logging.info('- {} ROC AUC score: {}'.format(mode, roc_auc_score))
    logging.info('{} confusion matrix:'.format(mode))
    logging.info(str(confusion[0]))
    logging.info(str(confusion[1]))


if __name__ == '__main__':
    # use GPU if available
    cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(128)
    if cuda:
        torch.cuda.manual_seed(128)

    # Set the logger
    utils.set_logger(os.path.join(MODEL_PATH, EXPERIMENT_PREFIX + 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data
    data_loader = DataLoader(DATA_PATH,
                             labels_path=LABELS_PATH,
                             categorical=CATEGORICAL,
                             max_length=MAX_LENGTH,
                             header=HEADER,
                             ascii=ASCII,
                             russian=RUSSIAN,
                             digits=DIGITS,
                             punctuation=PUNCTUATION,
                             lower=LOWER)
    train_data, val_data, train_labels, val_labels = data_loader.load_data(split=True)
    train_data = {'data': train_data, 'labels': train_labels}
    val_data = {'data': val_data, 'labels': val_labels}

    logging.info("... done.")

    # Define the model and optimizer
    if cuda:
        model = Crepe.cuda()  # not finished yet
    else:
        model = Crepe(vocabulary_size=data_loader.vocabulary_size,
                      channels=CHANNELS,
                      kernel_sizes=KERNEL_SIZES,
                      pooling_sizes=POOLING_SIZES,
                      linear_size=LINEAR_SIZE,
                      dropout=DROPOUT,
                      output_size=OUTPUT_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logging.info("Model compiled: {} parameters total.".format(sum([len(layer.view(-1)) for layer in model.parameters()])))

    # define the loss function
    loss_function = torch.nn.BCELoss()

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(EPOCHS))
    train_and_evaluate(model,
                       train_data,
                       val_data,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       optimizer=optimizer,
                       loss_function=loss_function,
                       metrics=metrics,
                       model_path=MODEL_PATH,
                       restore_file=None)
