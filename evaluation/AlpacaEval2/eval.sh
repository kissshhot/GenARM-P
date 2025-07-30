export OPENAI_API_KEY=""
export OPENAI_BASE_URL="https://api.deepseek.com/v1"

alpaca_eval --model_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/lr/Base-tulu-2-7b-ARM--alpha_1-temp_1.json --reference_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/tulu-2-7b-temp_0.json --output_path ./AE2_result/GenARM_SimPO_combine_lr_5e-5_vs_tulu2 --annotators_config 'alpaca_eval_gpt4_turbo_fn'

alpaca_eval --model_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/lr-2e-4/Base-tulu-2-7b-ARM--alpha_1-temp_1.json --reference_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/tulu-2-7b-temp_0.json --output_path ./AE2_result/GenARM_SimPO_combine_lr_2e-4_vs_tulu2 --annotators_config 'alpaca_eval_gpt4_turbo_fn'

# export OPENAI_BASE_URL="https://api.deepseek.com/v1"

# alpaca_eval --model_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/Base-tulu-2-7b-ARM-Gen_base_tulu_2_UB-arm-HH-epoch_1-beta_0.05-lr_5e-4-bs_32-alpha_1-temp_1.json --reference_outputs /home/dyf/rl/GenARM/evaluation/AlpacaEval2/AlpacaEval_results/model_generation/tulu-2-7b-temp_0.json --output_path ./AE2_result/GenARM_SimPO_combine_vs_tulu2 --annotators_config 'alpaca_eval_gpt4_turbo_fn'