# Training Agents in a Generative World Model

This repository contains the code and experiments developed for my thesis on **Reinforcement Learning with World Models**, focusing on **policy transfer from learned environments to simulators** and **cross-domain evaluation**.

## 📌 Project Overview

The goal of this thesis is to study how **on-policy Reinforcement Learning agents** behave when:

* trained inside a **learned World Model**
* transferred to a **ground-truth simulator**
* evaluated through **cross-analysis between training and testing domains**

The project investigates whether training in a learned latent environment can reduce sample complexity while preserving performance and stability when deployed in a real simulator.

## 🤖 Agents Implemented

The following on-policy agents are implemented and compared:

* **PPO (Proximal Policy Optimization)**
* **Recurrent PPO (RPPO)** – PPO augmented with an LSTM for partial observability
* **A2C (Advantage Actor-Critic)**

All agents share a common actor-critic structure, allowing fair and controlled comparisons.

## 🌍 Environments

Two main environment types are used:

* **World Model Environment**

  * Learned latent dynamics model
  * Used for agent training
* **Simulator Environment**

  * Ground-truth environment
  * Used for testing and transfer evaluation

Agents are trained and tested across both environments to analyze:

* performance degradation
* generalization ability
* stability after transfer

## 📊 Experimental Analysis

The thesis includes:

* Training curve comparisons
* Cross-domain performance tables
* Stability and convergence analysis
* Metrics such as reward trends and statistical variance
* Structural comparison between recurrent and non-recurrent agents

## ⚙️ Requirements

* Python 3.9+
* PyTorch
* Stable-Baselines3
* NumPy
* Gymnasium (or OpenAI Gym)

## 🚀 Running Experiments

### Train an agent in the World Model

```bash
python training/train_agent.py --agent PPO --env world_model
```

### Test a trained agent in the simulator

```bash
python evaluation/test_agent.py --agent PPO --env simulator
```

### Cross-domain evaluation

```bash
python evaluation/cross_analysis.py
```

## 📄 Thesis Context

This repository supports the experimental results presented in the thesis:

> **Training Agents in a Generative World Model**

The focus is on **policy robustness**, **representation learning**, and **transferability** between learned and real environments.

## 📌 Notes

* Hyperparameters are kept consistent across agents where possible
* Random seeds are controlled for reproducibility
* All experiments were run multiple times to reduce variance effects

## 📜 License

This project is intended for academic and research purposes.
