# git clone https://github.com/huggingface/diffusers
# cd diffusers
# pip install .

# cd examples
# pip install -r requirements.txt
# cd ../..
# accelerate config

# huggingface-cli login

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="finetune/lora/cartoon_2"
export HUB_MODEL_ID="cartoon-lora_2"
export DATASET_NAME="evilzip/cartoon_blip_captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="two people" \
  --seed=1320