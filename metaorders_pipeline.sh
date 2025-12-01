#!/bin/bash

conda activate defi
echo "Running metaorder computation..."
python metaorder_computation.py
echo "Running metaorder statistics..."
python metaorder_statistics.py