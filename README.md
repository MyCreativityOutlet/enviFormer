# enviFormer

Code for the paper titled "Predictive Modeling of Biodegradation Pathways
Using Transformer Architectures" published by Brydon *et al.* available [here]().

## Install
Clone this repository with `git clone https://github.com/MyCreativityOutlet/enviFormer`.

The code requires Java to be installed on the system. It has been confirmed to work with Amazon Corretto 22 available for download [here](https://docs.aws.amazon.com/corretto/latest/corretto-22-ug/downloads-list.html).
Check that Java is installed with `java --version`. Under tha Java folder is a `jar` file that handles running enviRule, running reactions with SMIRKS rules and standardising SMILES strings. 
It was created using a fork of enviRule and is available [here](https://github.com/MyCreativityOutlet/enviRule).

This code has been created and confirmed to work with Python 3.11 on Windows 10.

Create a virtual environment with `python -m venv .venv`. 
Make sure the environment is activated with `source .venv/bin/activate` on Linux or `source .venv/Scripts/activate` on Windows.
Then install the required packages with `pip install -r requirements.txt`.

## Data Availability
The three enviPath datasets, Soil, BBD and Sludge are available in the `data/` folder in individual json files.

The United States Patent & Trademark Office (USPTO) dataset is available [here]().

These are all the exact files used to run experiments in the paper.

## Recreate Experiments
In order to recreate the results presented in the paper we provide the following files:
- `PreTrainEnviFormer.py`
- `EnviFormerExperiments.py`
- `EnviRuleExperiments.py`
- `RuntimeExperiments.py`
- `CoverageExperiments.py`

### PreTrainEnviFormer.py
This script will perform the pretraining of the enviFormer model using the USPTO dataset.
Pretraining takes approximately two days on an RTX 4090 as such we provide the pretrained model and associated files we used, available [here]().
Unzip the folder and place its contents in the following directory `results/EnviFormerModel/uspto_regex/`.

### EnviFormerExperiments.py
This script will perform cross validation on the pretrained enviFormer model. 
It needs to be given one positional command line argument indicating which experiment to run.

The available options are:
- `baseline_soil` Do not perform any fine-tuning instead simply test on the Soil dataset.
- `baseline_bbd` Do not perform any fine-tuning instead simply test on the BBD dataset.
- `baseline_sludge` Do not perform any fine-tuning instead simply test on the Sludge dataset.
- `soil` Fine tune on the Soil dataset, test on the Soil dataset.
- `bbd` Fine tune on the BBD dataset, test on the BBD dataset.
- `sludge` Fine tune on the Sludge dataset, test on the Sludge dataset.
- `leave_soil` Fine tune on the BBD and Sludge datasets, test on the Soil dataset.
- `leave_bbd` Fine tune on the Soil and Sludge datasets, test on the BBD dataset.
- `leave_sludge` Fine tune on the Soil and BBD datasets, test on the Sludge dataset.
- `soil_add_bbd` Fine tune on the Soil and BBD datasets, test on the Soil dataset.
- `soil_add_sludge` Fine tune on the Soil and Sludge datasets, test on the Soil dataset.
- `bbd_add_soil` Fine tune on the BBD and Soil datasets, test on the BBD dataset.
- `bbd_add_sludge` Fine tune on the BBD and Sludge datasets, test on the BBD dataset.
- `sludge_add_soil` Fine tune on the Sludge and Soil datasets, test on the Sludge dataset.
- `sludge_add_bbd` Fine tune on the Sludge and BBD datasets, test on the Sludge dataset.

In any case where the train and test dataset are the same they will be appropriately split into train and test sets.

### EnviRuleExperiments.py
This script runs the same experiments as `EnviFormerExperiments.py` but using the Ensemble of Classifier Chains model.
It assumes that the `EnviFormerExperiments.py` has been run as it uses the saved train test split.
Including the `-er` command line argument will run the script with expert rules and not including it will use enviRule to extract rules.

By default, this script will run the list of experiments given in line 140 of the script. However, the `--data-name` command line argument can be used to define a single experiment.

### RunTimeExperiments.py
This script will measure the runtime performance of a given model. 

There are two command line arguments to be aware of:
- `--model-name` The options are envirule or EnviFormerModel.
- `--device` For envirule only the default CPU is available. EnviFormerModel and use CUDA or CPU.

### CoverageExperiments.py
This script calculates the coverage of all the methods on each dataset. It does not require any command line arguments.

# Contact
Author: Liam Brydon

Email: lbry121@aucklanduni.ac.nz
