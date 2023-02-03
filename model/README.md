## Dataset

This folder contains the LSTM baselines and the proposed model.

---


## Installation steps:
1. Inside model directory create python environment and activate the same
```bash
python3 -m venv env
source env/bin/activate
```
2. Install all requirements using below command
```bash
pip install -r requirements.txt
```
3. Load the environment to notebook
```bash
ipython kernel install --user --name=env
```
4. Follow the flow as in readme and main.ipynb in each directory

---

### Folders

1. baseline_classifier: This folder contains the baseline LSTM - classifier model.
2. baseline_mdn: This folder contains the baseline LSTM - MDN model.
3. proposed_model: This folder contains the proposed LSTM - MDN + Classifier model.
4. sample_dataset: This folder contains the sample dataset for the model.