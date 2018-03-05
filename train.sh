#!/bin/bash

output_dir="model_output_"`date +"%Y%m%d%H%M%S"`
phase="train"
# Can use TFRecord file to load data, or use Queue or numpy ndarray to load image.
image_root="./datasets/CelebA_nocrop/images"
metadata_path="./datasets/list_attr_celeba.txt"
batch_size=32
c_dim=8
training_steps=100000
summary_steps=500 # 100
save_steps=500 # 500
checkpoint_steps=500 # 1000

# new training
python main.py --output_dir="$output_dir" \
               --phase="$phase" \
               --image_root="$image_root" \
               --metadata_path="$metadata_path" \
               --batch_size="$batch_size" \
               --c_dim="$c_dim" \
               --training_steps="$training_steps" \
               --summary_steps="$summary_steps" \
               --save_steps="$save_steps" \
               --checkpoint_steps="$checkpoint_steps" 