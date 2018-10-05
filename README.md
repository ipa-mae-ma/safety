# Deep Reinforcement Learning for safety in robotics
![](badges/anaconda-v4.5.11-blue.svg)
![](badges/conda_env-safety-blue.svg)
![](badges/python-v3.6.6-blue.svg)
![](badges/Ubuntu-16.04.svg)
![](https://www.travis-ci.com/ipa-mae-ma/safety-drl.svg?branch=train)
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
- [ ] setup `travis` test for all architectures
  - update `.travis.yml`
  - `pytest` $\rightarrow$ easy to use

- [ ] gym environments for architectures:
  - [ ] frozen lake
  - [ ] crawler

## Architectures
- define functions:
  - [ ] `value iteration`
  - [ ] `policy iteration`
  - [ ] `q-learning`-function
  - [ ] `deep q`-function
  - [ ] `visualization`-function
    - `matplotlib`?

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
all environment information from `fruit_collection_train.py` $\rightarrow$ API to interact with architectures.


## Structure
The `environment` folder contains the fruit-collection environment game. In `architectures` are the implementations of the chosen architectures stored as one module.
Folder `ci` stores the `test_integration.py` file which handles all the `pytest` tests to sustain maintainability.

```text
ci/
  └── test_integration.py
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
