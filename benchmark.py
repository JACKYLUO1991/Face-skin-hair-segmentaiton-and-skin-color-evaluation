import time
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

from model.hrnet import HRNet
from model.hlrnet import HLRNet
from segmentation_models import PSPNet, Unet, FPN, Linknet
from experiments.CamVid.models import hlrnet


parser = argparse.ArgumentParser()
parser.add_argument("--image_size", '-i',
                    help="image size", type=int, default=224)
parser.add_argument("--batch_size", '-b',
                    help="batch size", type=int, default=1)
args = parser.parse_args()


def main():
    """
    Benchmark your model in your local pc.
    """
    model = hlrnet.HLRNet(input_shape=(args.image_size, args.image_size, 3))
    inputs = np.random.randn(args.batch_size, args.image_size, args.image_size, 3)

    time_per_batch = []

    for i in range(20):
        start = time.time()
        model.predict(inputs, batch_size=args.batch_size)
        elapsed = time.time() - start
        time_per_batch.append(elapsed)
        print("[info] time use {}".format(elapsed))

    time_per_batch = np.array(time_per_batch)

    # 去除第一次的测试结果
    print(time_per_batch[1:].mean())


if __name__ == '__main__':
    main()
