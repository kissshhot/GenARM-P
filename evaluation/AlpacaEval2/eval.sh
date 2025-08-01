export OPENAI_API_KEY="sk-3297eb120ba24740b82f555d16a2e27d"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"

alpaca_eval --model_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/wo-combine/Base-tulu-2-7b-ARM-tulu_2_debug_UB-arm-HH-epoch_1-gamma_1.0-beta_2.0-lr_1e-4-bs_32-alpha_1-temp_1.json --reference_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/tulu-2-7b-temp_0.json --output_path ./AE2_result/GenARM_SimPO_wo_combine_lr_1e-4_vs_tulu2 --annotators_config 'alpaca_eval_gpt4_turbo_fn'

# alpaca_eval --model_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/lr-5e-5/gamma_1/Base-tulu-2-7b-ARM-tulu_2_alpha_1.5_debug_UB-arm-HH-epoch_1-gamma_1.0-beta_2.0-lr_5e-5-bs_32-alpha_1-temp_1.json --reference_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/tulu-2-7b-temp_0.json --output_path ./AE2_result/GenARM_SimPO_combine_lr_1e-5_gamma_1_vs_tulu2 --annotators_config 'alpaca_eval_gpt4_turbo_fn'

# export OPENAI_BASE_URL="https://api.deepseek.com/v1"

# alpaca_eval --model_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/Base-tulu-2-7b-ARM-Gen_base_tulu_2_UB-arm-HH-epoch_1-beta_0.05-lr_5e-4-bs_32-alpha_1-temp_1.json --reference_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/tulu-2-7b-temp_0.json --output_path ./AE2_result/GenARM_SimPO_combine_vs_tulu2 --annotators_config 'alpaca_eval_gpt4_turbo_fn'