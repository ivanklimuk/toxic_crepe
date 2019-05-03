# Toxic CREPE
A pytorch implementation for the CREPE text classification model (https://arxiv.org/abs/1509.01626).

It is trained to predict the toxicity of a given piece of text.
___
### Data
The train data has to be stored as a .csv file in the `./data` folder and contain 2 columns - the labels and the raw texts themselves.

**Importnat note:** the current implementation is a binary classifier, which means it has only one single sigmoid output and is being trained with the Binary Cross Entropy loss, hence the labels should be integer values (either 0 or 1) and don't need any additional encoding.
___
### Variables

#### Model and data parameters:
- `MAX_LENGTH` - defines the maximal length of each input text (longer texts are truncated, shorter texts are padded with zeros)

- `CHANNELS` - the number of filters (same on each hidden convolutional layer)

- `KERNEL_SIZES` - the list of kernel sizes for each convolutional layer

- `POOLING_SIZE` - the size of each pooling operation in the model

- `LINEAR_SIZE` - the size of the hidden fully connected layers (same for all of them)

- `DROPOUT` - the dropout value

- `OUTPUT_SIZE` - the number of the outputs (should be equeal to the number of classes in case it's a multilabel classification problem)

- `EPOCHS` - the number of epochs

- `BATCH_SIZE` - the batch size

- `LEARNING_RATE`- the learning rate

#### Other parameters
- `MODEL_PATH` - the path for all model-related files

- `DATA_PATH` - the training dataset path (folder + filename)

- `EXPERIMENT_PREFIX` - a prefix for the experiment - all the corresponding files will have this prefix before their names

- `RUS` - a hardcoded cyrillic alphabet string

- `BEST_MODEL_PATH` - the filename of the best model (folder + filename)
___
### Train and outputs

To train the model you need to define all the desired parameters in the `constants.py` file (or in a `dev.env` file in the future)
After that you can run:
```
python train.py
```
The training process will procude multiple files in the model directory:
- A train log file called `train.log`

- Two models with the `.pth.tar` extension: the best model and last model

- Two json files with the evaluation metrics for the best and the last models