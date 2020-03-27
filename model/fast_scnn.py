import keras
from keras.backend import tf as ktf


class Fast_SCNN:
    def __init__(self, num_classes=3, input_shape=(256, 256, 3)):
        self.classes = num_classes
        self.input_shape = input_shape
        self.height = input_shape[0]
        self.width = input_shape[1]

    def conv_block(self, inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
        if conv_type == 'ds':
            x = keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides=strides)(inputs)
        else:
            x = keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides=strides)(inputs)

        x = keras.layers.BatchNormalization()(x)

        if relu:
            x = keras.layers.ReLU()(x)

        return x

    def learning_to_downsample(self):
        # Input Layer
        self.input_layer = keras.layers.Input(shape=self.input_shape, name='input_layer')

        self.lds_layer = self.conv_block(self.input_layer, 'conv', 32, (3, 3), strides=(2, 2))
        self.lds_layer = self.conv_block(self.lds_layer, 'ds', 48, (3, 3), strides=(2, 2))
        self.lds_layer = self.conv_block(self.lds_layer, 'ds', 64, (3, 3), strides=(2, 2))

    def global_feature_extractor(self):
        self.gfe_layer = self.bottleneck_block(self.lds_layer, 64, (3, 3), t=6, strides=2, n=3)
        self.gfe_layer = self.bottleneck_block(self.gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
        self.gfe_layer = self.bottleneck_block(self.gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
        # self.gfe_layer = self.pyramid_pooling_block(self.gfe_layer, [2, 4, 6, 8])
        self.gfe_layer = self.pyramid_pooling_block(self.gfe_layer, [1, 2, 4])

    def _res_bottleneck(self, inputs, filters, kernel, t, s, r=False):
        tchannel = keras.backend.int_shape(inputs)[-1] * t

        x = self.conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

        x = keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = self.conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

        if r:
            x = keras.layers.add([x, inputs])
        return x

    def bottleneck_block(self, inputs, filters, kernel, t, strides, n):
        x = self._res_bottleneck(inputs, filters, kernel, t, strides)

        for i in range(1, n):
            x = self._res_bottleneck(x, filters, kernel, t, 1, True)

        return x

    def pyramid_pooling_block(self, input_tensor, bin_sizes):
        concat_list = [input_tensor]
        w = self.width // 32
        h = self.height // 32

        for bin_size in bin_sizes:
            x = keras.layers.AveragePooling2D(pool_size=(bin_size, bin_size),
                                              strides=(bin_size, bin_size))(input_tensor)
            x = keras.layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.UpSampling2D(size=(bin_size * 2, bin_size * 2))(x)
            concat_list.append(x)

        return keras.layers.concatenate(concat_list)

    def feature_fusion(self):
        ff_layer1 = self.conv_block(self.lds_layer, 'conv', 128, (1, 1), padding='same', strides=(1, 1), relu=False)

        ff_layer2 = keras.layers.UpSampling2D((4, 4))(self.gfe_layer)
        ff_layer2 = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
        ff_layer2 = keras.layers.BatchNormalization()(ff_layer2)
        ff_layer2 = keras.layers.ReLU()(ff_layer2)
        ff_layer2 = keras.layers.Conv2D(128, (1, 1), strides=1, padding='same', activation=None)(ff_layer2)

        self.ff_final = keras.layers.add([ff_layer1, ff_layer2])
        self.ff_final = keras.layers.BatchNormalization()(self.ff_final)
        self.ff_final = keras.layers.ReLU()(self.ff_final)

    def classifier(self):
        self.classifier = keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1),
                                                       name='DSConv1_classifier')(self.ff_final)
        self.classifier = keras.layers.BatchNormalization()(self.classifier)
        self.classifier = keras.layers.ReLU()(self.classifier)

        self.classifier = keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1),
                                                       name='DSConv2_classifier')(self.classifier)
        self.classifier = keras.layers.BatchNormalization()(self.classifier)
        self.classifier = keras.layers.ReLU()(self.classifier)

        self.classifier = self.conv_block(self.classifier, 'conv', self.classes, (1, 1), strides=(1, 1), padding='same',
                                          relu=False)

        self.classifier = keras.layers.Lambda(lambda image: ktf.image.resize_images(image, (256, 256)), name='Resize')(
            self.classifier)
        self.classifier = keras.layers.Dropout(0.3)(self.classifier)

    def activation(self, activation='softmax'):
        x = keras.layers.Activation(activation,
                                    name=activation)(self.classifier)
        return x

    def model(self, activation='softmax'):
        self.learning_to_downsample()
        self.global_feature_extractor()
        self.feature_fusion()
        self.classifier()
        self.output_layer = self.activation(activation)

        model = keras.Model(inputs=self.input_layer,
                            outputs=self.output_layer,
                            name='Fast_SCNN')
        return model


if __name__ == '__main__':
    from flops import get_flops

    model = Fast_SCNN(num_classes=3, input_shape=(256, 256, 3)).model()
    model.summary()

    print(get_flops(model))
