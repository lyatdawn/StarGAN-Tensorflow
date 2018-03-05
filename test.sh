#!/bin/bash

# You should change the output_dir after training.
output_dir="model_output_20180304000252"
phase="test"
c_dim=8
# which trained model will be used.
# checkpoint="model-15000"

python main.py --output_dir="$output_dir" \
               --phase="$phase" \
               --c_dim="$c_dim" \
               --checkpoint="$checkpoint"

