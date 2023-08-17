import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    ReLU,
    Activation,
)
from tensorflow.keras.initializers import RandomNormal
from custom_padding import ReflectionPadding


class StyleTransferModel(keras.models.Model):
    def __init__(self, **kwargs):
        super(StyleTransferModel, self).__init__(**kwargs)

        self.model = keras.models.Sequential(
            [
                ConvBlock(
                    filters=32,
                    kernel_size=9,
                    strides=2,
                    padding_type="reflect",
                    padding=(4, 4),
                    activation="relu",
                ),
                ConvBlock(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    padding_type="reflect",
                    padding=(1, 1),
                    activation="relu",
                ),
                ConvBlock(
                    filters=128,
                    kernel_size=3,
                    strides=2,
                    padding_type="reflect",
                    padding=(1, 1),
                    activation="relu",
                ),
                ResidualBlock(filters=128, kernel_size=3, strides=1),
                ResidualBlock(filters=128, kernel_size=3, strides=1),
                ResidualBlock(filters=128, kernel_size=3, strides=1),
                ResidualBlock(filters=128, kernel_size=3, strides=1),
                ResidualBlock(filters=128, kernel_size=3, strides=1),
                UpSamplingBlock(
                    filters=64, kernel_size=3, strides=2, activation="relu"
                ),
                UpSamplingBlock(
                    filters=32, kernel_size=3, strides=2, activation="relu"
                ),
                UpSamplingBlock(filters=3, kernel_size=9, strides=2, activation="tanh"),
            ]
        )

    def call(self, x, training=True):
        return self.model(x, training=training)


class ConvBlock(keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding_type="same",
        padding=(1, 1),
        activation="relu",
        **kwargs
    ):
        super(ConvBlock, self).__init__(**kwargs)

        layers = []

        if padding_type == "reflect":
            layers.append(ReflectionPadding(paddings=padding))
            layers.append(
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="valid",
                    bias_initializer=RandomNormal(mean=0.0, stddev=0.02),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                )
            )
        elif padding_type == "same":
            layers.append(
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                    bias_initializer=RandomNormal(mean=0.0, stddev=0.02),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                )
            )
        else:
            raise ValueError("Padding method hasn't implemented yet.")

        layers.append(BatchNormalization())

        if activation == "relu":
            layers.append(ReLU())
        elif activation == "linear":
            layers.append(Activation(lambda x: x, name="linear"))
        else:
            raise ValueError("Activation's not supported")

        self.block = keras.models.Sequential(layers)

    def call(self, x):
        return self.block(x)


class UpSamplingBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, activation="relu", **kwargs):
        super(UpSamplingBlock, self).__init__(**kwargs)

        self.block = keras.models.Sequential(
            [
                Conv2DTranspose(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                    bias_initializer=RandomNormal(mean=0.0, stddev=0.02),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                ),
                BatchNormalization(),
            ]
        )

        if activation == "relu":
            self.block.add(ReLU())
        elif activation == "tanh":
            self.block.add(Activation(keras.activations.tanh))
        else:
            raise ValueError("Activation's supported")

    def call(self, x):
        return self.block(x)


class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.block = keras.models.Sequential(
            [
                ConvBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                    activation="relu",
                ),
                ConvBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                    activation="linear",
                ),
            ]
        )
        self.batch_norm = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        y = self.block(x) + x
        y = self.batch_norm(y)
        return self.relu(y)


if __name__ == "__main__":
    x = tf.random.uniform([7, 256, 256, 3])
    model = StyleTransferModel()

    print(model(x).shape)
