# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'fast'
wandb_run_name='gpt2-124M_500K_8x'

out_dir = 'out-fast-long'
init_from = 'scratch'

# these make the total batch size be ~0.2M
# 12 batch size * 1024 block size * 8 gradaccum * 5 GPUs = 0.5M
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 8 * 5

# use 300 * params ~ 30B tokens for training for about 10x chinchilla
max_iters = 15000 * 10
lr_decay_iters = 15000 * 10
# use 40 * params ~ 5B tokens for training for about 2x chinchilla
# 5e9 / (12 * 1024 * 8 * 5) = 10e3 iters
max_iters = ((10000 * 1 * 5) // 5) * 8
lr_decay_iters = ((10000 * 1 * 5) // 5) * 8
always_save_checkpoint = False

# eval stuff
eval_interval = 200
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
n_house = 0
n_loop = 1
bias = True # do we use bias inside LayerNorm and Linear layers?