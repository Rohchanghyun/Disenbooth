export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="/workspace/dataset/prcc/rgb/train"
export OUTPUT_DIR="/workspace/DisenBooth/output"

accelerate launch train_disenbooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a person</w> person" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_prompt="A photo of a person</w> person" \
  --validation_epochs=200 \
  --seed="48" \
