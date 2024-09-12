## Start a Validator Node

This doc describes how to start a validator node on the Genki-Dama subnet to help evaluating high quality creative models.

***Genki-Dama Subnet is running on the Bittensor TESTNET at this time. Remember to use the correct CLI arguments to select testnet when starting the validator node.***

### Prerequisite

Make sure your machine meets the following requirements before starting the validator node:

|  Hardware  | Requirements                         |
| ---------- | ------------------------------------ |
| GPU        | NVIDIA GPU with 12GB VRAM            |
| Memory     | 32GB                                 |
| Disk Space | 200GB                                |
| Network    | Public network access to Huggingface |
| OS         | Ubuntu or WSL2 on Windows            |

### Install the software

#### NVIDIA driver
Make sure the latest NVIDIA driver is installed:

[Install the NVIDIA Driver](https://www.nvidia.com/Download/index.aspx?lang=en-us)


Check the installation using the following command:

```bash
$ nvidia-smi

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 2000 Ada Gene...    On  |   00000000:41:00.0 Off |                  Off |
| 30%   31C    P8              6W /   70W |    2045MiB /  16380MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

#### Python

Python >= 3.10 is required to run the node. Make sure your Python meets the requirement before preceeding:

```bash
$ python --version

Python 3.10.12
```

#### Nodejs
Nodejs >= 20 is required. Please install Nodejs according to the official docs:

[Install Node.js](https://nodejs.org/en/download/package-manager)

#### Pm2
Pm2 is used to manage the validator processes. Install Pm2 using the NPM package manager:

```bash
$ npm install -g pm2
```

#### ffmpeg

```bash
$ sudo apt install ffmpeg
```

### Bittensor wallet configuration

#### Prepare the wallet
The validator wallet (coldkey and hotkey) should be stored on the machine. If you don't have a wallet yet, create the wallet according to the official Bittensor tutorial:

[Create Wallet | Bittensor](https://docs.bittensor.com/getting-started/wallets)

#### Register and Stake
The validator wallet must be registered to the subnet, and sufficient TAO must be staked to secure a validator permit. Please follow the official Bittensor turorial below to register the validator account and stake TAOs:

[Register, Validate and Mine | Bittensor](https://docs.bittensor.com/subnets/register-validate-mine)

### Prepare WanDB and Huggingface access tokens

WanDB is used to record the execution logs of the node. Please create a project on WanDB, and prepare the API access token for later usage.

Huggingface API access token is used to download the models from Huggingface for evaluation. Please get the access token from your Huggingface account for later usage.

### Prepare the code

#### Clone the repository

```bash
$ git clone https://github.com/crynux-ecosystem/genki-dama.git
```

#### Create the tmp directory

Create a wrirable directory for the node to write tmp files and cache the models:

```bash
# Go to the project root folder
$ cd genki-dama

# Create the tmp folder
$ mkdir evaluation

# Make the folder writable
$ chmod 777 evaluation
```

#### Install the dependencies

1. Create the venv for the node, and install the dependencies:

```bash
# In the project root folder

# Create the virtual environment
$ python -m venv venv

# Activate the venv
$ source venv/bin/activate

# Install the requirements
$ pip install -r requirements.txt

# Install the project as pip package
$ pip install .
```

2. Create the venv for the score server, and install the dependencies:

```bash
# In the project root folder

# Go to the dir of the score server
$ cd genki/model_evaluator/music/scores

# Create the virtual environment
$ python -m venv venv

# Activate the venv
$ source venv/bin/activate

# Install the requirements
$ pip install -r requirements.txt
```

#### Create the env file for access tokens

Create a file with name ```.env``` under the root folder of the project, and fill in the access tokens:

```
HF_ACCESS_TOKEN=[__your_token_here__]
WANDB_ACCESS_TOKEN=[__your_token_here__]
```

### Start the node

Start the node using the following command:

```bash
# In the project root folder

pm2 start ./venv/bin/python \
        --name gd_vali_guard \
         -- \
        ./scripts/start_validator.py \
        --netuid [__network_id__] \
        --subtensor.network test \
        --wallet.name [__your_wallet_name__] \
        --wallet.hotkey default \
        --neuron.axon_off \
        --logging.debug
```

### Monitor the node status

#### Check the running status

Use the pm2 command to check the running status of the validator node.

```bash
$ pm2 list

┌────┬────────────────────┬──────────┬──────┬───────────┬──────────┬──────────┐
│ id │ name               │ mode     │ ↺    │ status    │ cpu      │ memory   │
├────┼────────────────────┼──────────┼──────┼───────────┼──────────┼──────────┤
│ 4  │ gd_vali_main       │ fork     │ 0    │ online    │ 0%       │ 774.4mb  │
│ 0  │ gd_vali_guard      │ fork     │ 0    │ online    │ 0%       │ 336.8mb  │
│ 3  │ gd_score_api       │ fork     │ 0    │ online    │ 0%       │ 1.5gb    │
└────┴────────────────────┴──────────┴──────┴───────────┴──────────┴──────────┘
```

As shown in the command output above, there should be three processes running:

* The validator guard: periodically checks the GitHub repo for updates, and automatically updates & restarts the node when necessary.
* The validator process: the main validator process. Fetch the models submitted by miners, evaluate the modes and set weights.
* The score api server: calculate the model scores. Invoked by the validator process.

#### Check the logs

If something goes wrong, the logs will be very helpful to identify the problems. The logs for all the processes could be printed to the screen using pm2 command:

```bash
$ pm2 logs
```

The log files are located at the following dir by default:

```bash
$ cd ~/.pm2/logs
$ ls
```

The stdout and stderr of each process are recorded in different files.