#!/usr/bin/env sh
epochs=20
step_size=10
save_dir="./data/comparison"
data_path="./data"

cd ../

python  train.py --model_save_path="${save_dir}/cifar100_small_decoder_latent_3" --latent_channel=3 --data_root=${data_path} --batch_size=256 \
        --image_dim=32 --lr=0.001 --cw --model="cnn_small_decoder" --epochs=${epochs} --dataset="cifar100" --step_size=${step_size}

python  train.py --model_save_path="${save_dir}/cifar100_regular_decoder_latent_3" --latent_channel=3 --data_root=${data_path} --batch_size=256 \
        --image_dim=32 --lr=0.001 --cw --model="cnn" --epochs=${epochs} --dataset="cifar100" --step_size=${step_size}

python  train.py --model_save_path="${save_dir}/cifar100_small_decoder_latent_1" --latent_channel=1 --data_root=${data_path} --batch_size=256 \
        --image_dim=32 --lr=0.001 --cw --model="cnn_small_decoder" --epochs=${epochs} --dataset="cifar100" --step_size=${step_size}

python  train.py --model_save_path="${save_dir}/cifar100_regular_decoder_latent_1" --latent_channel=1 --data_root=${data_path} --batch_size=256 \
        --image_dim=32 --lr=0.001 --cw --model="cnn" --epochs=${epochs} --dataset="cifar100" --step_size=${step_size}

python  train.py --model_save_path="${save_dir}/unsplash_small_decoder_latent_3" --latent_channel=3 --data_root=${data_path} --batch_size=16 \
        --image_dim=256 --lr=0.001 --cw --model="cnn_small_decoder" --epochs=${epochs} --dataset="unsplash" --step_size=${step_size}

python  train.py --model_save_path="${save_dir}/unsplash_regular_decoder_latent_3" --latent_channel=3 --data_root=${data_path} --batch_size=16 \
        --image_dim=256 --lr=0.001 --cw --model="cnn" --epochs=${epochs} --dataset="unsplash" --step_size=${step_size}
