import os

root_dir = os.path.expanduser("~")

# train_data_path = "finished_files/train.bin"
# train_data_path = "finished_files/chunked/train_*"
# eval_data_path = "finished_files/val.bin"
# decode_data_path = "finished_files/test.bin"
# vocab_path = "finished_files/vocab"
# log_root = "log"
# fasttext_path = "/home/datbtd/torch_sum/ft_summarizer/data_util/fasttext_model.bin"

train_data_path = os.path.join(root_dir, "/media/lab/F1A3DFE4EA00A6D7/trung anh/data/finished_files/chunked/train_*")
eval_data_path = os.path.join(root_dir, "/media/lab/F1A3DFE4EA00A6D7/trung anh/data/finished_files/val.bin")
decode_data_path = os.path.join(root_dir, "/media/lab/F1A3DFE4EA00A6D7/trung anh/data/finished_files/test.bin")
vocab_path = os.path.join(root_dir, "/media/lab/F1A3DFE4EA00A6D7/trung anh/data/finished_files/vocab")
log_root = os.path.join(root_dir, "/media/lab/8BC523CC9D2874DB/trunganh/main-code-pg-word2vec-ad/ft_sum/log")
w2v = "/media/lab/8BC523CC9D2874DB/trunganh/data-gg-word2vec/GoogleNews-vectors-negative300.bin"

# Hyperparameters
hidden_dim= 256
emb_dim= 300
batch_size= 8
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 2000000

use_gpu=True

lr_coverage=0.15
