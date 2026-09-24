# Diff-Opto

This code repository is for the articles _A SIGNAL-FLOW-STRUCTURED STATE-SPACE MODEL OF OPTICAL DYNAMIC RANGE
COMPRESSORS_ (on Review) 

Visit the companion page with [audio samples](https://riccardovib.github.io/Diff-Opto_pages/)

This repository contains all the necessary utilities to use our architectures.

### Folder Structure

```
./
├── src
└── weights
```

### Contents

1. [Datasets](#datasets)
3. [How to Train and Run Inference](#how-to-train-and-run-inference)

<br/>

# Datasets

Datasets are here: 
- [LA2A & CL1B](https://www.kaggle.com/datasets/riccardosimionato/optical-dynamic-range-compressors-la-2a-cl-1b/versions/1)


# How To Train and Run Inference 

First, install Python dependencies:
```
cd ./
pip install -r requirements.txt
```

To train models, use the ```training.py``` script or via SSH with ```run.sh```.
Ensure you have loaded the dataset into the chosen datasets folder.

### Available Options

--data_dir - Root directory where the datasets are stored [str] (default="./data")

--filename - Name of the dataset file [str] (default="L2A2_analog")

--model_name - Path to save or load the model checkpoint [str] (default="./models/")

--model_type - Architecture to train ("tcn", "gcntf", "sptmod", "mamba", "diff-opto", "greycomp"]) [str] (default="diff-opto")

--batch_size - Number of samples per batch [int] (default=128)

--epochs - Number of training epochs [int] (default=60)

--lr - Initial learning rate [float] (default=3e-4)
 
--train_model - When True, train the model before test [bool] (default=False)

Example training case: 
```
cd ./src
python training.py \
  --data_dir ./data \
  --filename L2A2_analog \
  --model_path ./models/my_model \
  --model_type diff-opto \
  --batch_size 11 \
  --epochs 60 \
  --lr 3e-4 \
  --train_model True
```

To only run inference on an existing pre-trained model, set the "train_model" flag to False. In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./
python training.py \
  --data_dir ./data \
  --filename L2A2_analog \
  --model_path ./models/my_model \
  --model_type diff-opto \
  --batch_size 11 \
  --epochs 60 \
  --lr 3e-4 \
  --train_model False
```


# Bibtex

If you use the code included in this repository or any part of it, please acknowledge its authors by adding a reference to these publications:

```

```
