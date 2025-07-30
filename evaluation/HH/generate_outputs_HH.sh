# (when running DPO model, use DPO model as the base model, and set alpha = 0, temperature = 1)

cuda=3,4

base_model_pth=/data1/dyf/GenARM/checkpoints/HH/dpo/args-llama-sft-7b-dpo-HH-epoch_1-beta_0.1-lr_5e-4-bs_32/ # SFT model provided by ARGS
base_model_script_name=args-llama-sft-7b
arm_pth=/data1/dyf/GenARM/checkpoints/HH/dpo/args-llama-sft-7b-dpo-HH-epoch_1-beta_0.1-lr_5e-4-bs_32/
arm_script_name=dpo

alpha=0 # 0; 1 
temperature=1.0 # 0.5 # set to 1 / (1 + alpha) to sample from pi_decode with temperature=1.

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