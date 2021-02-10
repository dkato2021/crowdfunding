#!/bin/sh
cd run

echo "Preprocessing"
python3 1.preprocess1.py

python3 1.preprocess2.py

python3 1.preprocess3.py

echo "Learning"
python3 2.learning.py

echo "inferring"
python3 3.inference.py
