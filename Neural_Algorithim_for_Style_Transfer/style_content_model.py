import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg19 import preprocess_input
import helpers
import losses
import config


class StyleContentModel(keras.models.Model):
    def __init__(self, content_layers, style_layers):
        super(StyleContentModel, self).__init__()

        self.content_model = helpers.get_intermediate_layers(content_layers)
        self.style_model = helpers.get_intermediate_layers(style_layers)

        self.content_model.trainable = False
        self.style_model.trainable = False

    def call(self, inputs):
        """
        Expect inputs to be in range [0, 1]
        """
        inputs = inputs * 255.0

        processed_inputs = preprocess_input(inputs)

        style_outputs = self.style_model(processed_inputs)
        content_outputs = self.content_model(processed_inputs)

        style_outputs = [
            losses.gram_matrix(style_output) for style_output in style_outputs
        ]

        return {"content": content_outputs, "style": style_outputs}


if __name__ == "__main__":
    extractor = StyleContentModel(
        content_layers=config.CONTENT_LAYERS, style_layers=config.STYLE_LAYERS
    )

    content_img = helpers.load_image(
        "../test_image/content_image/YellowLabradorLooking_new.jpeg"
    )

    content_img = tf.image.resize(content_img, [224, 224])
    print(content_img)
