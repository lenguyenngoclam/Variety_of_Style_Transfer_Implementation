import tensorflow as tf
import tensorflow.keras as keras


class ReflectionPadding(keras.layers.Layer):
    def __init__(self, paddings=(1, 1), **kwargs):
        super(ReflectionPadding, self).__init__(**kwargs)
        self.paddings = tuple(paddings)

    def call(self, x):
        padding_height, padding_width = self.paddings
        return tf.pad(
            x,
            [
                [0, 0],
                [padding_height, padding_height],
                [padding_width, padding_width],
                [0, 0],
            ],
        )


if __name__ == "__main__":
    x = tf.random.uniform([4, 256, 256, 3])
    padding = ReflectionPadding()
    print(padding(x).shape)
