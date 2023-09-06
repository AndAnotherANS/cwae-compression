## Implementation of CWAE for image compression

### Training:

    python3 train.py --model_save_path="./data/model_test" --latent_channel=3 --data_root="./data" --batch_size=256 --image_dim=32 --lr=0.001 --cw --model="cnn" --epochs=20 --dataset="cifar100"

### Encode:

    python3 encode.py "./data/example_images/1.jpg" "encoded.txt" --model_save_path="./data/model_test/checkpoint_best_20.pt" --model="cnn_small_decoder" --block_size=2

### Decode:

    python3 decode.py "encoded.txt" "encoded.txt" --model_save_path="./data/model_test/checkpoint_best_20.pt" --model="cnn_small_decoder"

