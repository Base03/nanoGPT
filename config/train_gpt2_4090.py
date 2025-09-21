# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M_4090'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 * 8 gradaccum * 1 GPUs = 491,520
batch_size = 24
block_size = 1024
gradient_accumulation_steps = 5 * 4
#gradient_accumulation_steps = 2

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# model
n_layer = 16
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
n_house = 0
n_loop = 1
bias = False # do we use bias inside LayerNorm and Linear layers?