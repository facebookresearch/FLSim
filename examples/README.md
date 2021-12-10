# How to run this code
## Setup
```
python3 -m venv venv
source venv/bin/activate
cd ~/FLSim
pip install -e .
```

## Data Processing
In examples/get_data.sh, change line 8 to alter the dataset size. Currently, it's set to download and preprocess femnist. 

For example,
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample (full-sized dataset)
./preprocess.sh -s niid --sf 0.05 -k 0 -t sample (small-sized dataset)
Make sure to delete the rem_user_data, sampled_data, test, and train subfolders in the data directory before re-running preprocess.sh

For more info about the options, see (LEAF)[https://github.com/TalwalkarLab/leaf/tree/master/data/femnist]

then run 
```
sh get_data.sh
```

## Run 
CIFAR-10

```
python3 cifar10_example.py --config-file configs/cifar10_sarah.json
```