# (when running DPO model, use DPO model as the base model, and set alpha = 0, temperature = 1)

cuda=0,1

base_model_pth=/data1/dyf/model/tulu-2-7b # SFT model provided by ARGS
base_model_script_name=args-tulu-2-7b
arm_pth=/data1/dyf/GenARM/checkpoints/HH/arm/tulu_2_alpha_1.5_debug-arm-HH-epoch_1-gamma_0.5-beta_1.0-lr_1e-4-bs_32
arm_script_name=GenARM_SimPO_alpha_1.0_debug_gamma_0.5-beta_1.0-lr_1e-4

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

# export OPENAI_API_KEY="sk-3297eb120ba24740b82f555d16a2e27d"
# export OPENAI_BASE_URL="https://api.deepseek.com/v1"
# run_name_red=/home/dyf/rl/GenARM/evaluation/HH/model_outputs/Base_args-llama-sft-7b-Arm_GenARM_SimPO_alpha_2.0_debug_gamma_0.5-beta_1.0-lr_1e-4-Alpha_1.0-temp_0.5-promptNum_300.jsonl
# run_name_blue=/home/dyf/rl/GenARM/evaluation/HH/model_outputs/Base_args-llama-sft-7b-NoArm-temp_1.0-promptNum_300.jsonl

# output_dir=/home/dyf/rl/GenARM/evaluation/HH/gpt-evaluation/ # the evaluation will be saved at output_dir/f"{run_name_red_}_VS_{run_name_blue_}.json" (see gpt4_eval.py)

# python gpt4_eval.py --run_name_red $run_name_red --run_name_blue $run_name_blue --output_dir $output_dir