import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg19 import preprocess_input
import helpers
import losses
import config


class StyleContentModel(keras.models.Model):
    def __init__(self, **kwargs):
        super(StyleContentModel, self).__init__(**kwargs)
        self.style_model = helpers.get_intermediate_layers(config.STYLE_LAYERS)
        self.content_model = helpers.get_intermediate_layers(config.CONTENT_LAYERS)

        self.style_model.trainable = False
        self.content_model.trainable = False

    def call(self, inputs):
        # Expect inputs to be in range [-1, 1]
        inputs = inputs * 127.5 + 127.5

        processed_input = preprocess_input(inputs)

        content_outputs = self.content_model(processed_input)
        style_outputs = self.style_model(processed_input)

        # Calculate gram matrix for style outputs
        style_outputs = [
            losses.gram_matrix(style_output) for style_output in style_outputs
        ]

        return {"content": content_outputs, "style": style_outputs}


if __name__ == "__main__":
    img = tf.random.uniform([7, 256, 256, 3], minval=-1, maxval=1)
    extractor = StyleContentModel()

    print(extractor(img)["style"][0].shape)
