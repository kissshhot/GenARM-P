export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY="1256ddbd43ad5b80120e446f3105c432bc9a88aa"
export WANDB_PROJECT="wandb-test"
cuda=6,7
algorithm=arm # arm or dpo
epoch=1
beta=2.0 # for DPO, use beta=0.1, 默认是0.05
gamma=1.0
learning_rate=1e-4
bs=32 # Total batch size, which is $per_device_train_batch_size * $gradient_accumulation_steps * $num_GPU
per_device_train_batch_size=1 # Adjust according to your GPU memory

model_name_or_path="/data1/dyf/model/llama-7b-sft-float32" # sft model in the args paper
model_name_script=test_alpha_1.0_debug # GenARM_SimPO # only used for saving the model

###### the following is automatically set
num_GPU=$(echo $cuda | awk -F, '{print NF}')
gradient_accumulation_steps=$(($bs/$num_GPU/$per_device_train_batch_size))
exp_name=$model_name_script-$algorithm-HH-epoch_$epoch-gamma_$gamma-beta_$beta-lr_$learning_rate-bs_$bs

output_dir=/data1/dyf/GenARM/checkpoints/HH/$algorithm/$exp_name
if [ -d "${output_dir}" ]; then
    echo -e "\n\n"
    echo "Error: Directory "${output_dir}" already exists. Please delete it or choose a new output_dir." >&2
    exit 1
fi
echo "Output dir: $output_dir"

CUDA_VISIBLE_DEVICES=6,7 accelerate launch --gpu_ids $cuda --main_process_port 29500 training_trl/train_arm_llama.py --preference_dataset="HH_RLHF" \
    --algorithm=$algorithm --model_name_or_path=$model_name_or_path \
    --beta=$beta --gamma=$gamma --learning_rate=$learning_rate --num_train_epochs=$epoch \
    --output_dir=$output_dir --run_name=$exp_name \
    --per_device_train_batch_size=$per_device_train_batch_size --gradient_accumulation_steps=$gradient_accumulation_steps --per_device_eval_batch_size=4 \
    --logging_steps=10 --evaluation_strategy="steps" --eval_steps=50 --save_strategy="steps" --save_steps=1000 \
    --lr_scheduler_type="cosine" --warmup_steps=100 --weight_decay=0.05 --gradient_checkpointing=True --bf16=True \
    --max_prompt_length=512 --max_length=1024 --report_to="wandb" --remove_unused_columns=False --length_normalization=True \
    # --entropy_keep_ratio=0.5

echo "Finished training $output_dir"


# (when running DPO model, use DPO model as the base model, and set alpha = 0, temperature = 1)
cd /home/dyf/rl/GenARM/evaluation/HH
cuda=6,7

base_model_pth=/data1/dyf/model/llama-7b-sft-float32/ # SFT model provided by ARGS
base_model_script_name=args-llama-sft-7b
arm_pth=output_dir
arm_script_name=GenARM_SimPO_alpha_1.0_debug_gamma_1.0-beta_2.0-lr_1e-4

alpha=1 # 0; 1 
temperature=0.5 # 0.5 # set to 1 / (1 + alpha) to sample from pi_decode with temperature=1.

num_prompt=300

### automatically set
out_folder=/home/dyf/rl/GenARM/evaluation/HH/model_outputs
if [ "$alpha" -eq 0 ]; then
    echo "alpha is zero; only use the base model"
    out_file=Base_$base_model_script_name-NoArm-temp_$temperature-promptNum_$num_prompt
else
    out_file=Base_$base_model_script_name-Arm_$arm_script_name-Alpha_$alpha-temp_$temperature-promptNum_$num_prompt
fi

echo "Output dir: $out_folder/$out_file"

CUDA_VISIBLE_DEVICES=$cuda python generate_outputs_HH.py \
    --base_model_pth=$base_model_pth --arm_pth=$arm_pth --alpha=$alpha --temperature=$temperature \
    --num_prompt=$num_prompt --out_folder=$out_folder --out_file=$out_file \
    --max_new_tokens 128 --max_prompt_length 2048

export OPENAI_API_KEY="sk-3297eb120ba24740b82f555d16a2e27d"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
run_name_red=/home/dyf/rl/GenARM/evaluation/HH/model_outputs/Base_args-llama-sft-7b-Arm_GenARM_SimPO_alpha_1.0_debug_gamma_1.0-beta_2.0-lr_1e-4-Alpha_1-temp_0.5-promptNum_300.jsonl
run_name_blue=/home/dyf/rl/GenARM/evaluation/HH/model_outputs/Base_args-llama-sft-7b-NoArm-temp_1.0-promptNum_300.jsonl

output_dir=/home/dyf/rl/GenARM/evaluation/HH/gpt-evaluation/ # the evaluation will be saved at output_dir/f"{run_name_red_}_VS_{run_name_blue_}.json" (see gpt4_eval.py)

python gpt4_eval.py --run_name_red $run_name_red --run_name_blue $run_name_blue --output_dir $output_dir