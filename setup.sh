#!/bin/bash

python3 -m venv ./venv &&
source ./venv/bin/activate &&
python -m pip install pip --upgrade &&
python -m pip install pandas numpy matplotlib &&
python -m pip install kaggle &&
python -m pip install jupyter &&
deactivate
