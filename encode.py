from argparse import ArgumentParser

from PVQ import PVQ
from utils import *

def add_arguments(parser: ArgumentParser):
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--model", choices=["cnn", "cnn_small_decoder"], default="cnn", help="model architecture to use")
    parser.add_argument("--model_save_path", type=str, required=True, help="path to model weights file")
    parser.add_argument("--latent_channel", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--pvq_k", type=int, default=5)

    parser.add_argument("--pvq_cache_dir", type=str, default="./data/pvq")


def pad(s, l):
    while len(s) < l:
        s = "0" + s
    return s

def make_header(args):
    block_size_header = pad(np.binary_repr(args.block_size), 8)
    pvq_k_header = pad(np.binary_repr(args.pvq_k), 8)
    latent_channel_header = pad(np.binary_repr(args.latent_channel), 8)

    header = "1" + block_size_header + pvq_k_header + latent_channel_header
    return header


def encode():
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    args.image_channels=3
    args.image_dim = 32

    model = get_model(args)
    model.load_state_dict(torch.load(args.model_save_path)["model"])
    model = model.to(args.device).eval()
    pvq = PVQ(args.block_size, args.pvq_k, cache_dir=args.pvq_cache_dir)

    encoded = make_header(args)


    image = torch.tensor(np.asarray(Image.open(args.input_file))).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.
    latent = model.forward_encoder(image.to(args.device))

    latent = latent.flatten(1)
    enc = pvq.encode(latent.cpu().detach().squeeze().numpy())
    encoded = encoded + enc

    with open(args.output_file, "wb+") as file:
        while len(encoded) % 8 != 0:
            encoded = "0" + encoded
        bit_strings = [encoded[i:i + 8] for i in range(0, len(encoded), 8)]
        byte_list = [int(b, 2) for b in bit_strings]
        print(bytearray(byte_list))
        file.write(bytearray(byte_list))
        file.close()

    with open("test.txt", "w+") as file:
        file.write(encoded)

if __name__ == '__main__':
    encode()