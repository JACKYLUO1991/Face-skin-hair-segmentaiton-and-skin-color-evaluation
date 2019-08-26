import argparse
from data_loader import HairGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler
import tensorflow as tf
import os
import warnings
from keras import optimizers
from keras import backend as K
from keras.regularizers import l2
# from SGDR import SGDRScheduler
from metric import *
from segmentation_models.losses import *
import numpy as np
# Albumentations: fast and flexible image augmentations
from albumentations import *
# ADAPTIVE GRADIENT METHODS WITH DYNAMIC BOUND OF LEARNING RATE
# from keras_adabound import AdaBound
from model.hrnet import HRNet
from model.hlrnet import HLRNet

# https://github.com/qubvel/segmentation_models
# 可以自动选择backbone以及网络深度
# from segmentation_models import PSPNet, Unet, FPN, Linknet

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", '-b',
                    help="batch size", type=int, default=4)
parser.add_argument("--image_size", '-i',
                    help="image size", type=int, default=224)
parser.add_argument("--backbone", '-bb',
                    help="backbone of the network", type=str, default=None)
parser.add_argument("--epoches", '-e', help="epoch size",
                    type=int, default=100)
parser.add_argument("--learning_rate", '-lr',
                    help="learning rate", type=float, default=1e-3)
parser.add_argument("--checkpoints", '-cp',
                    help="where is the checkpoint", type=str, default='./checkpoints')
parser.add_argument("--class_number", '-cn',
                    help="number of output", type=int, default=3)
parser.add_argument("--data_dir", '-dd',
                    help="path of dataset", type=str, default='./data/CelebA')
args = parser.parse_args()


class PolyDecay:
    '''
    指数衰减策略实现
    '''

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
    '''参数正则化处理，以防止模型过拟合'''
    for layer in model.layers:
        if kernel_regularizer is not None and hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = kernel_regularizer

        if bias_regularizer is not None and hasattr(layer, 'bias_regularizer'):
            layer.bias_regularizer = bias_regularizer

        if activity_regularizer is not None and hasattr(layer, 'activity_regularizer'):
            layer.activity_regularizer = activity_regularizer


def main():
    # 设置GPU环境
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # 可以根据需要的数据集进行更新
    ROOT_DIR = args.data_dir
    BACKBONE = args.backbone
    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.image_size
    EPOCHS = args.epoches
    LR = args.learning_rate
    CHECKPOINT = args.checkpoints
    CLS_NUM = args.class_number

    # 数据扩增处理
    train_transformer = Compose([Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
                                 #  GaussNoise(p=0.2),
                                 ShiftScaleRotate(
                                     shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
                                 HorizontalFlip(p=0.5),
                                 #  HueSaturationValue(p=0.5),
                                 #  RandomBrightnessContrast(0.5),
                                 GridDistortion(distort_limit=0.2, p=0.5)])
    val_transformer = Compose(
        [Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True)])

    # 1. 数据生成器
    # 2. 注意：如果是自己设计的模型需要将normalization置为True
    train_generator = HairGenerator(
        train_transformer, ROOT_DIR, mode='Training', batch_size=BATCH_SIZE,
        backbone=BACKBONE, shuffle=True)

    val_generator = HairGenerator(
        val_transformer, ROOT_DIR, mode='Testing', batch_size=BATCH_SIZE,
        backbone=BACKBONE)

    # 使用 pre-trained 模型进行训练
    # model = HRNet(input_size=(IMG_SIZE, IMG_SIZE, 3), cls_num=CLS_NUM)
    model = Unet(backbone_name=BACKBONE, input_shape=(
        IMG_SIZE, IMG_SIZE, 3), classes=CLS_NUM, encoder_weights='imagenet', activation='softmax')
    # model = HLRNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), cls_num=CLS_NUM)
    # 正则化防止模型过拟合
    set_regularization(model, kernel_regularizer=l2(2e-5))
    model.compile(optimizer=optimizers.SGD(lr=LR, momentum=0.98),
                  loss=cce_dice_loss, metrics=[mean_accuracy, mean_iou, frequency_weighted_iou, pixel_accuracy])

    if not os.path.exists(CHECKPOINT):
        os.mkdir(CHECKPOINT)

    checkpoint = ModelCheckpoint(filepath=os.path.join(CHECKPOINT, 'model-{epoch:03d}-{val_loss:.3f}.h5'),
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join(CHECKPOINT, 'logs'))
    csvlogger = CSVLogger(
        os.path.join(CHECKPOINT, "result.csv"))

    # SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS
    # scheduler = SGDRScheduler(
    #     min_lr=LR*(0.1**3), max_lr=LR, steps_per_epoch=len(train_generator), lr_decay=0.9,
    #     cycle_length=5, mult_factor=1.5)
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
