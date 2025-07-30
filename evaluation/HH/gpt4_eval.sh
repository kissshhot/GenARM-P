
export OPENAI_API_KEY=""
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
run_name_red=/home/dyf/rl/GenARM/evaluation/HH/model_outputs/Base_args-tulu-2-7b-Arm_GenARM_SimPO_alpha_2.0_debug_gamma_0.5-beta_1.0-lr_1e-4-Alpha_1-temp_0.5-promptNum_300.jsonl
run_name_blue=/home/dyf/rl/GenARM/evaluation/HH/only_arm/Base_args-tulu-2-7b-Arm_only_arm_tulu2-Alpha_1-temp_0.5-promptNum_300.jsonl

output_dir=/home/dyf/rl/GenARM/evaluation/HH/gpt-evaluation/ # the evaluation will be saved at output_dir/f"{run_name_red_}_VS_{run_name_blue_}.json" (see gpt4_eval.py)

python gpt4_eval.py --run_name_red $run_name_red --run_name_blue $run_name_blue --output_dir $output_dir