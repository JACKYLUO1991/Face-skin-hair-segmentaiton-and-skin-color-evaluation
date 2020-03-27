import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import time
import numpy as np
import argparse
from tqdm import tqdm

from model.hlnet import HLNet
from model.dfanet import DFANet
from model.enet import ENet
from model.lednet import LEDNet
from model.segnet import SegNet
from model.fast_scnn import Fast_SCNN

parser = argparse.ArgumentParser()
parser.add_argument("--image_size", '-i',
                    help="image size", type=int, default=256)
parser.add_argument("--batch_size", '-b',
                    help="batch size", type=int, default=3)
parser.add_argument("--model_name", help="model's name",
                    choices=['hlnet', 'fastscnn', 'lednet', 'dfanet', 'enet', 'segnet'],
                    type=str, default='hlnet')
parser.add_argument("--nums", help="output num",
                    type=int, default=1)
args = parser.parse_args()

IMG_SIZE = args.image_size
CLS_NUM = args.nums


def get_model(name):
    if name == 'hlnet':
        model = HLNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), cls_num=CLS_NUM)
    elif name == 'fastscnn':
        model = Fast_SCNN(num_classes=CLS_NUM, input_shape=(IMG_SIZE, IMG_SIZE, 3)).model()
    elif name == 'lednet':
        model = LEDNet(groups=2, classes=CLS_NUM, input_shape=(IMG_SIZE, IMG_SIZE, 3)).model()
    elif name == 'dfanet':
        model = DFANet(input_shape=(IMG_SIZE, IMG_SIZE, 3), cls_num=CLS_NUM, size_factor=2)
    elif name == 'enet':
        model = ENet(input_shape=(IMG_SIZE, IMG_SIZE, 3), cls_num=CLS_NUM)
    elif name == 'segnet':
        model = SegNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), cls_num=CLS_NUM)
    else:
        raise NameError("No corresponding model...")

    return model


def main():
    """Benchmark your model in your local pc."""

    model = get_model(args.model_name)
    inputs = np.random.randn(args.batch_size, args.image_size, args.image_size, 3)

    time_per_batch = []

    for i in tqdm(range(100)):
        start = time.time()
        model.predict(inputs, batch_size=args.batch_size)
        elapsed = time.time() - start
        time_per_batch.append(elapsed)

    time_per_batch = np.array(time_per_batch)

    # Remove the first item
    print(time_per_batch[1:].mean())


if __name__ == '__main__':
    main()
