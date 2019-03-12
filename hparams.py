# Audio
num_mels = 40
num_freq = 257
sample_rate = 16000
frame_length_ms = 25
frame_shift_ms = 10
tisv_frame = 180
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5

# Model
n_mels_channel = 40
hidden_dim = 768
num_layer = 3
speaker_dim = 256
re_num = 1e-6

# Train
dataset_path = "./dataset"
dataset_test_path = "./dataset_test"
origin_data = "D:/wav48"
total_utterance = 200
N = 9  # batch_size
M = 10
learning_rate = 0.01
epochs = 500
checkpoint_path = "./model_new"
save_step = 500
log_step = 5
clear_Time = 20
