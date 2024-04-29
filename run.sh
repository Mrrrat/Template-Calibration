export WANDB_DISABLED="false"
export WANDB_API_KEY=524696af93e88fa63289e627ea61c2f561bbddc6
export WANDB_NAME="PEFT Calibration"
python peft_template_calibration.py \
--dataset sst2 \
--models google/gemma-2b \
--seed 1 \
--precision fp16 \
--examples_selection_method random \
--prediction_method direct \
--num_templates 3 \
--steps 10 \
--lr 3e-4 \
--sigma 0 \
--loss mean \
--data_size 50 \
--batch_size 2 \
--eval_batch_size 32 \
--select_best True \
--save_dir bebra \
