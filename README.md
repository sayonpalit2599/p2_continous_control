# Continous Control
Deep Deterministic Policy Gradients (DDPG) algorithm based agent for solving Reacher Environment (Unity) as part of Udacity's Deep Reinforcement Nanodegree by <a href = 'https://github.com/sayonpalit2599/'>Sayon Palit</a> 
## Project Environment
* For this project we work with Unity's <strong>Reacher</strong> environment(single version)

 <img src = "https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif">

* In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location.
Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

* The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Solving the Environment
* <b>The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.</b>

### Real World Implications
* Watch this <a href = "https://www.youtube.com/watch?v=ZVIxt2rt1_4">YouTube video</a> to see how some researchers were able to train a similar task on a real robot! The accompanying research paper can be found <a href = "https://arxiv.org/pdf/1803.07067.pdf"> here</a>.

## Requirements
### Getting Started
To run the codes, follow the next steps:
* Create a new environment:
	* __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	* __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
* Perform a minimal install of OpenAI gym
	* If using __Windows__, 
		* download swig for windows and add it the PATH of windows
		* install Microsoft Visual C++ Build Tools
	* then run these commands
	```bash
	pip install gym
	```
* Install the dependencies under the folder python/
```bash
	cd python
	pip install .
```
* Create an IPython kernel for the `drlnd` environment
```bash
	python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
* Download the Unity Environment specific to your operating system
	* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
	* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
	* [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
	* [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

* Start jupyter notebook from the root of this python codes
```bash
jupyter notebook
```
* Once started, change the kernel through the menu `Kernel`>`Change kernel`>`drlnd`
* If necessary, inside the ipynb files, change the path to the unity environment appropriately

* Follow the instructions in `Continuous_Control.ipynb` to train the agent and use `Run_Agent.ipynb` to see the performance of the agent.

* <strong>`It is recommended to use a GPU for running this code`</strong>
## Results
<img src = "https://github.com/sayonpalit2599/p2_continous_control/blob/master/plot.jpg" alt = "result">

* The current model solves the environment in approx ~560 - 600 epsiodes on average with an ` NVIDIA GTX1050TI ` and 
`Intel i8750H`.
