# Generate model artifacts for execution on SnapDragon devices

## Tested Environment

**Linux x86 PC**

- Distributor ID: Ubuntu
- Description:    Ubuntu 22.04.4 LTS
- Release:        22.04
- Platform: x86_64 AMD

## Setup 
The Python environment can be set up using either Anaconda or Python virtual environment (venv).

**Note:** One of the following two steps to setup the Python environment must be executed before executing the notebook.

If you have already started the jupyter notebook, configure the Python environment before you continue. After configuring the Python environment, restart the notebook server and select the correct kernel.

### Set up Anaconda in an Ubuntu 22.04 terminal

1. Install Anaconda from : https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh.

2. Execute the setup script with the following command.

    ```chmod a+x Anaconda3-2023.03-1-Linux-x86_64.sh && bash Anaconda3-2023.03-1-Linux-x86_64.sh```

3. Configure an Anaconda environment with the following commands in the Ubuntu 22.04 terminal.

    ```conda create --name llama_env python=3.10```
    
    ```conda activate llama_env```
    
    ```conda install ipykernel```
    
    ```ipython kernel install --user --name=llama_env``` 

### Setup venv (non-Anaconda) in an Ubuntu 22.04 terminal

The following steps install the packages required to use the QNN tools in an Ubuntu 22.04 environment (Ubuntu terminal window).

1. Update the package index files.

    ```sudo apt-get update```

2. Install Python3.10 and necessary packages.

    By default Ubuntu 22.04 should come with Python 3.10 and you don't need to install it again. However to reinstall it run the following command.

    ```sudo apt-get update && sudo apt-get install python3.10 python3.10-dev python3-setuptools```

3. Install python3-pip.

    ```sudo apt-get install python3-pip```

4. Install python3 virtual environment support.

    ```sudo apt install python3-virtualenv```

5. Create and activate a Python 3.10 virtual environment by executing the following commands.
    ```
    virtualenv -p /usr/bin/python3.10 venv_llama3
    source venv_llama3/bin/activate
    ```

### Install required Python packages to start notebook

Within the virtual environment, run the following command to install the required Python packages to start the notebook.
```
pip install notebook==7.2.1 \
    jupyter-server==2.4.0 \
    jupyter_client==8.3.0 \
    jupyter_core==5.3.1 \
    jupyterlab-miami-nights==0.4.1
```

## Launch Jupyter Notebook
Launch the jupyter notebook as follows:

```bash
jupyter notebook --ip=* --no-browser --allow-root &
```

Once the server starts, you will be presented a URL(s) that you may copy and paste into your web browser.

From the web browser, click on the Jupyter Notebook `qnn_model_prepare.ipynb` and follow the instructions therein.
