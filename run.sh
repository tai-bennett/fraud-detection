#!/usr/bin/env bash

source venv/bin/activate &&
python3 main.py configs/config1.json &&
deactivate
