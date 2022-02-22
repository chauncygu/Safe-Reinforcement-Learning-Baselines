# AlwaysSafe

Code for the paper
    ["AlwaysSafe: Reinforcement Learning Without Safety Constraint Violations During Training"](https://tdsimao.github.io/publications/Simao2021alwayssafe/)
    —
        [Thiago D. Simão](https://tdsimao.github.io/ ),
        [Nils Jansen](https://www.cs.ru.nl/personal/nilsjansen/ ) and
        [Matthijs T. J. Spaan](https://www.st.ewi.tudelft.nl/mtjspaan/ ),
     published at AAMAS 2021.

[[Details]](https://tdsimao.github.io/publications/Simao2021alwayssafe/ )


## modules

- `agents`: model based RL agents that interact with the environment.
- `planners`: the planners used by the RL agents to compute the policy in each episode.
- `scripts`: each file is related to one of the experiments from the paper.
- `tests`: mostly unittest scripts.
- `util`: contains common scripts to train an RL agent and evaluate a policy.


## lp solver

By default, the code uses [`gurobipy`](https://www.gurobi.com/) if found, otherwise it uses [`cvxpy`](https://www.cvxpy.org/).


## usage

1. install dependencies
    ```
    pipenv install
   ```
1. run tests
    ```bash
    pipenv run python -m unittest
    ```
1. reproduce the experiments
    ```bash
    pipenv run python -m scripts.simple
    pipenv run python -m scripts.factored
    pipenv run python -m scripts.cliff_walking
    ```


## citing

```text.bibtex
@inproceedings{Simao2021alwayssafe,
  author    = {Sim{\~a}o, Thiago D. and Jansen, Nils and Spaan, Matthijs T. J.},
  title     = {AlwaysSafe: Reinforcement Learning Without Safety Constraint Violations During Training},
  year      = {2021},
  booktitle = {Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS)},
  publisher = {IFAAMAS},
  pages     = {1226–1235},
}
```
