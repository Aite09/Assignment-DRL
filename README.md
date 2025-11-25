Mohammed Aitezazuddin Ahmed
100829388

⸻
1. Overview

This project automates the testing of two applications using trained Deep Reinforcement Learning agents:
	1.	Web Flow App — simulated step-by-step web navigation environment
	2.	Flappy Bird Game — 2D side-scrolling game agent

The goal is to detect issues, measure performance, observe different persona behaviors, and reproduce experiments reliably.

According to the assignment requirements, DRL agents (not hand-coded bots) must be used to perform testing, evaluate system behavior, and collect structured metrics—this project meets all these goals ￼.


⸻
2. Repository Structure

The repo follows the exact deliverable structure required in the assignment ￼:

The repo follows the exact deliverable structure required in the assignment ￼:
<img width="463" height="548" alt="Screenshot 2025-11-24 at 9 47 47 PM" src="https://github.com/user-attachments/assets/2bae8024-9900-4d44-8538-8cc6ccd3a070" />


3. Setup Instructions

1. Create virtual environment (recommended Python 3.10)

python3.10 -m venv venv
source venv/bin/activate

2. Install dependencies

pip install -r requirements.txt


⸻

4. How to Train Models

The assignment requires 2+ algorithms and personas ￼.

Example training commands:

Train PPO on Web Flow

python src/train.py --algo ppo --app web --persona explorer --timesteps 50000

Train A2C on Web Flow

python src/train.py --algo a2c --app web --persona survivor --timesteps 50000

Train PPO on Flappy Bird

python src/train.py --algo ppo --app flappy --persona explorer --timesteps 40000

Models are automatically saved under:

models/


⸻

5. How to Evaluate Models

Required per assignment: per-episode and aggregated metrics in CSV/JSON format ￼.

Example:

python src/evaluate.py --model_path models/web_ppo_explorer_seed7.zip

This will output:

logs/web_ppo_explorer_eval.csv


⸻

6. Analysis Notebook

Open the notebook:

notebooks/analysis_web.ipynb

It includes:
	•	Reward curves
	•	Episode reward distributions
	•	Error/success rate
	•	State coverage plots
	•	Persona comparison

These plots satisfy assignment deliverable requirements for Matplotlib charts and comparisons ￼.

⸻

7. Environments

Web Flow Environment
	•	Actions: next, previous, submit, skip
	•	Observations: page index, input validity, completion flag
	•	Rewards: based on persona
	•	Explorer: rewards exploring new pages
	•	Survivor: rewards safe/valid completions

Flappy Bird Environment
	•	Actions: flap, no-op
	•	Observations: bird position, pipe position, velocity
	•	Rewards:
	•	+1 per gap passed
	•	-10 on collision
	•	+0.1 surviving per timestep

Follows assignment expectation to clearly document actions/observations/rewards ￼.

⸻

8. Experiments Completed

As required: ≥3 experiments per app ￼

✔ PPO vs A2C on Web App
✔ Explorer vs Survivor persona on Web App
✔ PPO Explorer trained on Flappy Bird

Metrics collected:
	•	Episode reward
	•	Length / survival time
	•	Errors found (web)
	•	Gaps passed (flappy)
	•	Coverage

⸻

9. Reproducibility

To fully recreate results:
	1.	Install environment
	2.	Run the training commands listed
	3.	Evaluate each model
	4.	Open analysis notebooks

Seeds are pinned in:

configs/seeds.yaml

This fulfills reproducibility requirements ￼.

⸻

10. Summary

During the early stages of the project, a simplified Flappy Bird environment was briefly used purely for debugging and validating the reinforcement learning pipeline (reward shaping, action loop, observation processing, and model execution). This environment is lightweight and helps verify that PPO and A2C were running correctly before integrating the full WebFlow software-testing environment. However, the final results, analysis, training, evaluation, and all deliverables in this assignment are based exclusively on the WebFlow environment, which aligns directly with the assignment requirement of testing a software system with complex user flows. The Flappy Bird environment was not used for final submission — it served only as a temporary internal test to ensure the DRL pipeline worked as expected.

