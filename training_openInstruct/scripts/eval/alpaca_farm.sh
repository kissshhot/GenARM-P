# Please make sure OPENAI_API_KEY is set in your environment variables

# Use V1 of alpaca farm evaluation.
export IS_ALPACA_EVAL_2=True
export OPENAI_API_KEY="sk-3297eb120ba24740b82f555d16a2e27d"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
# use vllm for generation
python -m eval.alpaca_farm.run_eval \
    --model_name_or_path ../checkpoints/tulu_v1_7B/ \
    --tokenizer_name_or_path ../checkpoints/tulu_v1_7B/ \
    --save_dir results/alpaca_farm/tulu_v1_7B/ \
    --eval_batch_size 20 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # use normal huggingface generation function
# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path ../checkpoints/tulu_v1_7B/ \
#     --tokenizer_name_or_path ../checkpoints/tulu_v1_7B/ \
#     --save_dir results/alpaca_farm/tulu_v1_7B/ \
#     --eval_batch_size 20 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --load_in_8bit
