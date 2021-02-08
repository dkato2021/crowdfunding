#!/bin/sh
cd run

python3 1.preprocess1.py

python3 1.preprocess2.py

python3 1.preprocess3.py

python3 2.learning.py

python3 3.inference.py