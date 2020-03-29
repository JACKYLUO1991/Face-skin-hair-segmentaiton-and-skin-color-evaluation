import argparse
from data_loader import HairGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler
import os
import warnings
from keras import optimizers
from keras.regularizers import l2
from metric import *
from segmentation_models.losses import *
import numpy as np

from albumentations import *
from model.hlnet import HLNet
from model.dfanet import DFANet
from model.enet import ENet
from model.lednet import LEDNet
from model.mobilenet import MobileNet
from model.fast_scnn import Fast_SCNN

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", '-b',
                    help="batch size", type=int, default=64)
parser.add_argument("--image_size", '-i',
                    help="image size", type=int, default=256)
parser.add_argument("--backbone", '-bb',
                    help="backbone of the network", type=str, default=None)
parser.add_argument("--epoches", '-e', help="epoch size",
                    type=int, default=150)
parser.add_argument("--model_name", help="model's name",
                    choices=['hlnet', 'fastscnn', 'lednet', 'dfanet', 'enet', 'mobilenet'],
                    type=str, default='hlnet')
parser.add_argument("--learning_rate", help="learning rate", type=float, default=2.5e-3)
parser.add_argument("--checkpoints",
                    help="where is the checkpoint", type=str, default='./weights')
parser.add_argument("--class_number",
                    help="number of output", type=int, default=3)
parser.add_argument("--data_dir",
                    help="path of dataset", type=str, default='./data/CelebA')
args = parser.parse_args()


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
    elif name == 'mobilenet':
        model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), cls_num=CLS_NUM)
    else:
        raise NameError("No corresponding model...")

    return model


class PolyDecay:
    '''Exponential decay strategy implementation'''

    def __init__(self, initial_lr, power, n_epochs):
        self.initial_lr = initial_lr
        self.power = power
        self.n_epochs = n_epochs

    def scheduler(self, epoch):
        return self.initial_lr * np.power(1.0 - 1.0 * epoch / self.n_epochs, self.power)


def set_regularization(model,
                       kernel_regularizer=None,
                       bias_regularizer=None,
                       activity_regularizer=None):
    '''Parameter regularization processing to prevent model overfitting'''
    for layer in model.layers:
        if kernel_regularizer is not None and hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = kernel_regularizer

        if bias_regularizer is not None and hasattr(layer, 'bias_regularizer'):
            layer.bias_regularizer = bias_regularizer

        if activity_regularizer is not None and hasattr(layer, 'activity_regularizer'):
            layer.activity_regularizer = activity_regularizer


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    global IMG_SIZE
    global CLS_NUM

    ROOT_DIR = args.data_dir
    BACKBONE = args.backbone
    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.image_size
    EPOCHS = args.epoches
    LR = args.learning_rate
    CHECKPOINT = args.checkpoints
    CLS_NUM = args.class_number
    MODEL_NAME = args.model_name

    train_transformer = Compose([  # GaussNoise(p=0.2),
        ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
        HorizontalFlip(p=0.5),
        #  HueSaturationValue(p=0.5),
        #  RandomBrightnessContrast(0.5),
        # GridDistortion(distort_limit=0.2, p=0.5),
        Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
    ])
    val_transformer = Compose(
        [Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True)])

    train_generator = HairGenerator(
        train_transformer, ROOT_DIR, mode='Training', batch_size=BATCH_SIZE, nb_classes=CLS_NUM,
        backbone=BACKBONE, shuffle=True)

    val_generator = HairGenerator(
        val_transformer, ROOT_DIR, mode='Testing', batch_size=BATCH_SIZE, nb_classes=CLS_NUM,
        backbone=BACKBONE)

    # Loading models
    model = get_model(MODEL_NAME)
    set_regularization(model, kernel_regularizer=l2(2e-5))
    model.compile(optimizer=optimizers.SGD(lr=LR, momentum=0.98),
                  loss=cce_dice_loss, metrics=[mean_iou, frequency_weighted_iou, mean_accuracy, pixel_accuracy])

    CHECKPOINT = CHECKPOINT + '/' + MODEL_NAME
    if not os.path.exists(CHECKPOINT):
        os.makedirs(CHECKPOINT)

    checkpoint = ModelCheckpoint(filepath=os.path.join(CHECKPOINT, 'model-{epoch:03d}.h5'),
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join(CHECKPOINT, 'logs'))
    csvlogger = CSVLogger(
        os.path.join(CHECKPOINT, "result.csv"))

    lr_decay = LearningRateScheduler(PolyDecay(LR, 0.9, EPOCHS).scheduler)

    model.fit_generator(
        train_generator,
        len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=EPOCHS,
        verbose=1,
        callbacks=[checkpoint, tensorboard, csvlogger, lr_decay]
    )

    K.clear_session()


if __name__ == '__main__':
    main()
