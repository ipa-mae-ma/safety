# Deep Reinforcement Learning for safety in robotics
![](badges/anaconda-v4.5.11-blue.svg)
![](badges/conda_env-safety-blue.svg)
![](badges/python-v3.6.6-blue.svg)
![](badges/Ubuntu-16.04.svg)
[![Build Status](https://travis-ci.com/ipa-mae-ma/safety.svg?branch=master)](https://travis-ci.com/ipa-mae-ma/safety)
![](badges/trousers-shorts-yellow.svg)

## Architectures
### HRA
[Hybrid Reward Architecture](http://arxiv.org/abs/1706.04208)

### A3C
[Asynchronous Advantage Actor Critic](http://arxiv.org/abs/1602.01783) (A3C)

### DQN
[Deep Q-Networks](https://www.nature.com/articles/nature14236.pdf)

## Fruit Game
[Fruit game](https://github.com/Maluuba/hra) from Maluuba will be used to benchmark the different architectures.

# TODO list

## CI
- [x] setup `travis` test for all architectures
  - [x] update `.travis.yml`
  - [x] `pytest` $\rightarrow$ easy to use

- [x] gym environments for architectures:
  - [x] frozen lake
  - [x] crawler


## Architectures
- define functions:
  - [x] `value iteration`
  - [ ] `policy iteration`
  - [x] `q-learning`-function
  - [ ] `deep q`-function
  - [ ] `visualization`-function
    - `matplotlib`?

### DQN

- [ ] use `pickle` to save weights
  - [ ] include `overwrite` parameter
- [ ] use `results` folder
- [ ] implement `warmstart` functionality
- [ ]

- define MDP to pass to architectures:
  - [ ] states
  - [ ] actions
  - [ ] reward
  - [ ] terminal
  - [ ] probability
  - [ ] goals for "HER"?

- hyper parameters:
  - learning rate
  - gamma

## Interaction
all environment information from `fruit_collection_train.py`: API to interact with architectures.


## Structure
The `environment` folder contains the fruit-collection environment game. In `architectures` are the implementations of the chosen architectures stored as one module.
Folder `ci` stores the `test_integration.py` file which handles all the `pytest` tests to sustain maintainability.

```text
ci/
  ├── crawler_env.py
  ├── discrete_env.py
  ├── frozen_lake.py
  ├── function_test.py
  ├── gym_env_test.py
  └── __init__.py
environment/
  ├── architectures/
      ├── a3c.py
      ├── hra.py
      ├── __init__.py
      ├── mdp.py
      └── misc.py
  ├── fruit_collection_pictures.py
  ├── fruit_collection.py
  ├── fruit_collection_train.py
  ├── pictures/
  └── README.md
```
