from llava.train.train import train
from llava.dist_utils import init_distributed_mode


if __name__ == "__main__":
    init_distributed_mode()
    train()
