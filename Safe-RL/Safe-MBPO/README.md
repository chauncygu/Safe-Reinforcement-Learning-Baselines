# Safe-MBPO
Code for the NeurIPS 2021 paper "Safe Reinforcement Learning by Imagining the Near Future" by Garrett Thomas, Yuping Luo, and Tengyu Ma.

Some code is borrowed from [Force](https://github.com/gwthomas/force).

## Installation
We are using Python 3.8. The required packages can be installed via

	pip install -r requirements.txt

You also must set the `ROOT_DIR` in `code/defaults.py`.
This is where experiments' logs and checkpoints will be placed.

Once setup is complete, run the code using the following command:

	python main.py -c config/ENV.json

where ENV is replaced appropriately. To override a specific hyperparameter, add `-s PARAM VALUE` where `PARAM` is a string.
Use `.` to specify hierarchical structure in the config, e.g. `-s alg_cfg.horizon 10`.