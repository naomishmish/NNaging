# CNNs Aging Project 
Deep learning models show remarkable results on computer vision tasks by optimizing the network parameters, such as network depth, layers connections, weights initialization, and network hyperparameters.
A fascinating observation is that biological systems fail differently than man-made machines as they age. In this work, we will study how these Deep networks perform as a function of age-associated failure. We will explore the effect of aging on different state-of-the-art deep networks and 2D vision datasets.


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [CCNs Aging Project](#cnns-aging-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
		* [Run test with pruning](#run-test-with-pruning)
		* [Run multiply test](#run-multiply-test)
		* [Run all configurations](#run-all-configurations)
		* [Analyze](#analyze)
			* [Plot metrics](#plot-metrics)
			* [Plot failure rate](#plot-failure-rate)
			* [Plot fit to Gompertz and Weibull](#plot-fit-to-gompertz-and-weibull)
    * [Using Multiple GPU](#using-multiple-gpu)
	* [Customization](#customization)
		* [Custom CLI options](#custom-cli-options)
		* [Data Loader](#data-loader)
		* [Model](#model)
		* [Loss](#loss)
		* [metrics](#metrics)
		* [Additional logging](#additional-logging)
		* [Testing](#testing)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Features
* 4 CNN models - VGG16, InceptionV3, ResNet50, EfficientNetB7
* 3 dataloader - Cifar10, Cifar100, ImageNet
* `.json` config files support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* 3 python scripts for ploting the experiments results:
  * `plot_all.py` ploting the different metrics for every configuration.
  * `plot_acc.py` ploting the failure rate for every configuration.
  * `plot_fit.py` ploting the fit to Gompertz and Weibull function.

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model and option for prune the models
  │
  ├── config.json - holds example configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── plot_all.py - ploting the different metrics for every configuration
  ├── plot_acc.py - ploting the failure rate for every configuration
  ├── plot_fit.py - ploting the fit to Gompertz and Weibull function
  |
  ├── our_prune.py - implementation of LastN pruning
  |
  ├── std.py - calculate the std of the different experiments
  ├── func.py -  contain different functions for analyzing the data
  |
  ├── avg.sh - run each test number of times and calculate the avarage of the different metrics
  ├── run.sh - call the avg script for each one of the configurations.
  |
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py - contain the four models: VGG16, InceptionV3, ResNet50 and EfficientNetB7
  │   ├── metric.py - contain four metrics: accuracy, top 5 accuracy, TPR and TNR
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  ├── utils/ - small utility functions
  │   ├── util.py
  │   └── ...
  |
  ├── config_fiels/ - contain all the configuration fiels with the tuned hyper-parameters
  │   └── ...
  │
  └── prune_result_final/ - contain al the csv fiels from the avg script
     └── ...
  ```

## Usage
The command `python train.py -c config_files/config.json` run the training of the chosen configuration.

### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "VGG16_Cifar10",        // training session name
  "n_gpu": 1,                   // number of GPUs to use for training.
  
  "arch": {
    "type": "VGG16Model",       // name of model architecture to train
    "args": {

    }                
  },
  "data_loader": {
    "type": "Cifar10DataLoader",         // selecting data loader
    "args":{
      "data_dir": "data/",             // dataset path
      "batch_size": 64,                // batch size
      "shuffle": true,                 // shuffle training data before splitting
      "validation_split": 0.1          // size of validation dataset. float(portion) or int(number of samples)
      "num_workers": 2,                // number of cpu processes to be used for data loading
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "cross_entropy",                  // loss
  "metrics": [
    "accuracy", "top_k_acc"            // list of metrics to evaluate
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10	                 // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}
```

All of the configuration files are under config_files folder.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```
### Run test with pruning
In order to run the test with pruning use the command:

  ```
  python test.py -r path/to/checkpoint -p <percent> -t <pruning type>
  ```
 * When percent is in decimal, for example: percent = 0.2 for remove 20% of the weights
 * When pruning type can be one of the three options: random / l1 / lastN

### Run multiply test
In order to get the average of several runnings, run the avg.sh script. It will generate csv file with the average metrics and their std.
* csv format:
  ```
  pruning_percent avg_loss avg_accuracy avg_top5 avg_TPR avg_TNR std_accuracy std_top5 std_TPR avg_TNR
         |           |        |            |        |       |       |            |        |       |  
	     |           |        |            |        |       |       |            |        |       |
	     |           |        |            |        |       |       |            |        |       |
	     |           |        |            |        |       |       |            |        |       |   
	     |           |        |            |        |       |       |            |        |       |   
	     |           |        |            |        |       |       |            |        |       |   
	     |           |        |            |        |       |       |            |        |       |   
  ```

* command line:
  ```
  ./avg.sh <model name> path/to/checkpoint <prune type>
  ```
  for example:
  * model name: VGG16_cifar10
  * prune type: random / l1 / lastN

### Run all configurations
To run all the experiments, use the script run.sh. If you want to run just part of the configurations, you can comment their appearance in the script.

### Analyze
For analyzing the data, there are three scripts. they read the data from the csv fiels generated by the avg.sh script.

#### Plot metrics
Ploting the different metrics for every configuration. Has no parameters.

#### Plot failure rate
Ploting the failure rate for every configuration. Has no parameters.
 
#### Plot fit to Gompertz and Weibull
Ploting the fit to Gompertz and Weibull function. Has no parameters.


## Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config_files/config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config_files/config.py
  ```

## Customization

### Project initialization
Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file. 

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target` 
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.


### Data Loader
containing three kinds of dataloaders: Cifar10, Cifar100 and ImageNet.
The EfficientNetB7 model need to get the Images in a different initialization so there is a copy of the dataloaders for its use.

### Model
containig four models: VGG16, InceptionV3, ResNet50 and EfficientNetB7. there is another model- MNIST for example from previous project.

### Loss
Custom loss functions can be implemented in 'model/loss.py'. We implemented the cross entropy loss function.

### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```
We implemented four kinds of metrics: accuracy, top 5 accuracy, TPR and TNR

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
You can specify the name of the training session in config files:
  ```json
  "name": "VGG16_Cifar10",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```
  
## Acknowledgements
This project used the template pytorch project [PyTorch-Template-Project](https://github.com/xieydd/High-Resolution-Neural-Face-Swapping-for-Visual-Effects) by [xieydd](https://github.com/xieydd)
