# Artillery RL — Gymnasium (Q-learning + DQN)

This repository is migrated to **Gymnasium** and contains:
- env.py : Gymnasium-compatible environment (obs includes shelter location & wind)
- qlearning.py : Tabular Q-learning (discrete action grid)
- dqn_wrapper.py : Map discrete integer -> (angle_change, speed)
- train_dqn.py : DQN training (stable-baselines3)
- main_demo.py : pygame rendering & interactive demo
- main.py : high-level runner (no PPO in this version)

## Quick setup (macOS)

1. Create & activate venv:
```bash
#python3 -m venv venv
#source venv/bin/activate
python -m venv venv
.\venv\Scripts\Activate.ps1

```

2. Install dependencies (recommended):
```bash
#pip install --upgrade pip
#pip install -r requirements.txt
python -m pip install --upgrade pip
pip install -r requirements.txt

```

3. Commands:

Interactive demo:
```bash
#python main.py demo
python main_demo.py

```

Train Q-learning (tabular):
```bash
#python qlearning.py
##python main.py qlearning
#python test_qlearning.py
python qlearning.py
python test_qlearning.py

```

Train Sac
```bash
#python sac_agent_reward.py --train
#python sac_agent_reward.py --test
python sac_agent_reward.py --train
python sac_agent_reward.py --test
'''


Notes:

- Increase `total_timesteps` in main.py or train_dqn.py for better performance (200k+ recommended).
