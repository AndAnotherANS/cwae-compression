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
    parser.add_argument("--pvq_k", default=5)

    parser.add_argument("--pvq_cache_dir", type=str, default="./data/pvq")


def pad(s, l):
    while len(s) < l:
        s = "0" + s
    return s

def read_from_stream(stream, num):
    chunk, stream = stream[:num], stream[num:]
    return chunk, stream


def decode():
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    args.image_channels=3
    args.image_dim = 32

    model = get_model(args)
    model.load_state_dict(torch.load(args.model_save_path)["model"])
    model = model.to(args.device).eval()


    with open(args.output_file, "rb") as file:
        import sys
        encoded = np.binary_repr(int.from_bytes(file.read(), byteorder="big"))

    while encoded[0] == "0":
        encoded = encoded[1:]

    encoded = encoded[1:]

    block_size, encoded = read_from_stream(encoded, 8)
    pvq_k, encoded = read_from_stream(encoded, 8)
    latent_channels, encoded = read_from_stream(encoded, 8)

    block_size, pvq_k, latent_channels = int(block_size, 2), int(pvq_k, 2), int(latent_channels, 2)

    pvq = PVQ(block_size, pvq_k, cache_dir=args.pvq_cache_dir)

    decoded = pvq.decode(encoded)
    dim = np.sqrt(decoded.shape[-1] // latent_channels).astype(int)
    decoded = decoded.reshape(1, latent_channels, dim, dim)

    image = model.forward_decoder(torch.tensor(decoded).float().to(args.device)).squeeze().permute(1, 2, 0).cpu().detach().numpy()

    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    decode()