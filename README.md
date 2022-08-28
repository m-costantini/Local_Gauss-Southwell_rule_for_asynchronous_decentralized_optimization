# Gauss-Southwell rule for decentralized optimization

Repository containing the code to reproduce the experiments in M. Costantini, N. Liakopoulos, P. Mertikopoulos, and T. Spyropoulos, “Pick your neighbor: Local Gauss-Southwell rule for fast asynchronous decentralized optimization,” in 61st IEEE Conference on Decision and Control (CDC), 2022.

The paper is available [here](https://arxiv.org/abs/2207.07543).

## Execution instructions

The code is written in python programming language.
Download the complete folder to execute the code in your local computer.
You will need to have installed [python](https://www.python.org/downloads/) and the following libraries:
* [Numpy](https://numpy.org/install/)
* [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
* [Netowrkx](https://networkx.org/)

Files `main_decentralized_setting.py` and `main_parallel_distributed_setting.py` contain the code to execute the experiments in the evaluation section of the paper for the decentralized and parallel distributed settings, respectively.

To execute them:
* Open the [command prompt](https://en.wikipedia.org/wiki/Cmd.exe) if you have a Windows system or the [unix shell](https://en.wikipedia.org/wiki/Unix_shell) if you are in MacOS or Linux
* Type `cd CODE_PATH`, replacing CODE_PATH with the path to the folder containing the code
* Type `python main_decentralized_setting.py` or `python main_parallel_distributed_setting.py` depending on which experiment you want to run
