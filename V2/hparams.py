# Audio
num_mels = 80
num_freq = 1025
sample_rate = 20000
frame_length_ms = 50
frame_shift_ms = 12.5
tisv_frame = 180
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5

# Model
pre_input_size = 80
pre_output_size = 60
hidden_dim = 256
num_layer = 2
speaker_dim = 64
re_num = 1e-6
layers = [2, 2, 2]

# Train
dataset_path = "./dataset"
dataset_test_path = "./dataset_test"
origin_data = "C:/Users/28012/Desktop/SV-dataset/wav_data"
N = 2  # batch_size
M = 10
learning_rate = 0.01
epochs = 500
checkpoint_path = "./model_new"
save_step = 5
log_step = 5
clear_Time = 20
