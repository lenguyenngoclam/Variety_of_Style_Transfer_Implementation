from tensorflow.keras.applications import VGG19
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL


def get_intermediate_layers(layers):
    """Create intermediate model

    @param layers: List of intermeditate layers you want to get

    @return keras Model
    """
    model = VGG19(include_top=False, weights="imagenet")
    model.trainable = False

    outputs = [model.get_layer(layer).output for layer in layers]

    return keras.models.Model(inputs=[model.input], outputs=outputs)


def load_image(path):
    max_dim = 512
    # Read image
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Scale image
    shape = tf.cast(img.shape[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    # Resize image
    img = tf.image.resize(img, new_shape)
    # Add batch dimension
    img = tf.expand_dims(img, axis=0)

    return img


def tensor_to_image(img):
    # Expect image value to be in range [0, 1]
    img = img * 255

    img = tf.cast(img, tf.uint8)

    if len(img.shape) == 4:
        assert img.shape[0] == 1
        img = img[0]

    return img


if __name__ == "__main__":
    img = load_image("../test_image/content_image/content_image_3.jpeg")

    plt.imshow(tensor_to_image(img))
    plt.show()
