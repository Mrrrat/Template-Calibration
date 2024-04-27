export WANDB_DISABLED="false"
export WANDB_API_KEY=524696af93e88fa63289e627ea61c2f561bbddc6
export WANDB_NAME="Template Calibration Exps"
python template_calibration.py \
--dataset dbpedia \
--models tiiuae/falcon-7b \
--seed 1  \
--precision fp16 \
--examples_selection_method 0-shot \
--prediction_method direct \
--num_templates 3 \
--steps 10 \
--lr 3e-4 \
--sigma 0 \
--loss mean \
--data_size 50 \
--select_best \
--save_dir bebra \
