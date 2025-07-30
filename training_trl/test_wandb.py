import wandb

# 初始化 W&B 项目
wandb.init(
    project="wandb-test",  # 可自定义项目名称
    name="test-run",                  # 本次 run 的名字
    mode="online",                     # 确保尝试连接 W&B 云端
    api_key="1256ddbd43ad5b80120e446f3105c432bc9a88aa"
)

# 可选：记录一点数据，确保交互
wandb.log({"ping": 1})

# 结束运行
wandb.finish()