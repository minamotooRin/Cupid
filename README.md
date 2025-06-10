# Cupid

Test the pesonality of LLMs

## Installation

It is recommended using conda to create a virtual environment for this project. 
To install the required packages, run the following command:

```bash
conda env create -f environment.yml
```

## Usage

Please refer to `demo.ipynb` for a demonstration of the code.

Using following command to run the whole test with the specified GPT configuration file on IPIP test in five languages.

```bash
python -u main.py --config_path configs/conf_GPT.json --excel_dir IPIP/ --langs DE ZH JA MN EN > log/log_GPT.txt 2>&1
```

You can also define your own configuration file in the `configs/` directory. The configuration file should be a JSON file that specifies the model parameters and other settings.

## Directory Structure

* **AgentFactory/**
  A package containing classes and factories for creating and managing AI agent interfaces and remote model wrappers.

* **configs/**
  Directory for configuration files that define experiment settings, model parameters.

* **DeepL/**
  Module for interacting with the DeepL translation API, used for translating IPIP test.

* **IPIP/**
  Folder containing IPIP personality inventory data.

* **Questionnaire/**
  Containing the Questionnaire class.