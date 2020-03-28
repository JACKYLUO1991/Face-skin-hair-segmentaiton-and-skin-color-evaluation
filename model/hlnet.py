# Fast-SCNN
# HRNet
# MobileNetv2-v3
# ASPP
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model

import keras.backend as K


def _conv_block(inputs, filters, kernel, strides=1, padding='same', use_activation=False):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding=padding, strides=strides,
               use_bias=False)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)

    if use_activation:
        x = Activation('relu')(x)

    return x


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1))

    x = DepthwiseConv2D(kernel, strides=(
        s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    # relu6
    x = ReLU(max_value=6)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """
    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def _depthwise_separable_block(inputs, kernel, strides, padding='same', depth_multiplier=1):
    '''Depth separable point convolution module'''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = DepthwiseConv2D(kernel_size=kernel, strides=strides, padding=padding,
                        depth_multiplier=depth_multiplier)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation('relu')(x)


def HLNet(input_shape, cls_num=3):
    """Higt-Low Resolution Information fusion Network"""
    # input_shape: input image shape
    # cls_num: output class number
    inputs = Input(input_shape)
    # Step 1: Feature dimension drops to 1/4
    x = _conv_block(inputs, 32, (3, 3), strides=2, use_activation=True)
    x = _depthwise_separable_block(x, (3, 3), strides=2, depth_multiplier=2)
    x = _depthwise_separable_block(x, (3, 3), strides=2)

    # step 2:
    x21 = _inverted_residual_block(
        x, 64, kernel=(3, 3), t=6, strides=1, n=3
    )
    x22 = _inverted_residual_block(
        x, 96, kernel=(3, 3), t=6, strides=2, n=3
    )
    x23 = _inverted_residual_block(
        x, 128, kernel=(3, 3), t=6, strides=4, n=3
    )

    # step 3:
    x31_t1 = x21
    x31_t2 = UpSampling2D(interpolation='bilinear')(
        _conv_block(x22, 64, (1, 1), use_activation=True))
    x31_t3 = UpSampling2D(size=(4, 4), interpolation='bilinear')(
        _conv_block(x23, 64, (1, 1), use_activation=True))
    x31 = Add()([x31_t1, x31_t2, x31_t3])

    x32_t1 = _conv_block(x21, 96, (1, 1), strides=2, use_activation=True)
    x32_t2 = _conv_block(x22, 96, (1, 1), use_activation=True)
    x32_t3 = UpSampling2D(interpolation='bilinear')(
        _conv_block(x23, 96, (1, 1), use_activation=True))
    x32 = Add()([x32_t1, x32_t2, x32_t3])

    x33_t1 = _conv_block(x21, 128, (1, 1), strides=4, use_activation=True)
    x33_t2 = _conv_block(x22, 128, (1, 1), strides=2, use_activation=True)
    x33_t3 = _conv_block(x23, 128, (1, 1), use_activation=True)
    x33 = Add()([x33_t1, x33_t2, x33_t3])

    # step 4:
    x41 = _conv_block(x33, 96, (1, 1))
    x42 = UpSampling2D(interpolation='bilinear')(x41)
    x43 = Concatenate()([x42, x32])
    x44 = _conv_block(x43, 64, (1, 1))
    x45 = UpSampling2D(interpolation='bilinear')(x44)
    x46 = Concatenate()([x45, x31])

    # step 5: FFM module in BiSeNet
    x50 = _conv_block(x46, 64, (3, 3))
    x51 = AveragePooling2D(pool_size=(1, 1))(x50)
    x52 = Conv2D(64, (1, 1), use_bias=False, activation='relu')(x51)
    x53 = Conv2D(64, (1, 1), use_bias=False, activation='sigmoid')(x52)
    x54 = Multiply()([x53, x50])
    x55 = Add()([x50, x54])

    # step6:
    x61 = Conv2D(32, (3, 3), padding='same', dilation_rate=2)(x55)
    x62 = Conv2D(32, (3, 3), padding='same', dilation_rate=4)(x55)
    x63 = Conv2D(32, (3, 3), padding='same', dilation_rate=8)(x55)
    x64 = Add()([x61, x62, x63])
    # x61 = _conv_block(x62, cls_num, (1, 1), use_activation=False)
    x65 = UpSampling2D(size=(8, 8), interpolation='bilinear')(x64)
    x66 = _conv_block(x65, cls_num, (1, 1), use_activation=False)
    out = Activation('softmax')(x66)

    return Model(inputs, out)


if __name__ == "__main__":
    from flops import get_flops

    # Testing network design
    model = HLNet(input_shape=(256, 256, 3), cls_num=3)
    model.summary()

    print(get_flops(model))


