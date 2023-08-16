from tensorflow.keras.applications import VGG19
import tensorflow.keras as keras
import tensorflow as tf
import config
import matplotlib.pyplot as plt


def get_intermediate_layers(layers):
    model = VGG19(include_top=False, weights="imagenet")
    model.trainable = False

    outputs = [model.get_layer(layer).output for layer in layers]

    return keras.models.Model(inputs=[model.input], outputs=outputs)


def load_image(filepath):
    # Read image
    img = tf.io.read_file(filepath)
    img = tf.io.decode_image(img)
    img = tf.cast(img, tf.float32)

    # Rescale image
    max_dims = 512
    img_shape = tf.cast(img.shape[:-1], tf.float32)
    long_dim = max(img_shape)
    scale = max_dims / long_dim

    new_shape = tf.cast(scale * img_shape, tf.int32)
    # Resize image
    img = tf.image.resize(img, new_shape)

    # Add batch dimension
    img = tf.expand_dims(img, axis=0)

    return img


def tensor_to_image(input_tensor):
    # Expect input to be in range [-1, 1]
    input_tensor = (input_tensor * 127.5) + 127.5

    input_tensor = tf.cast(input_tensor, tf.uint8)

    if len(input_tensor.shape) > 3:
        input_tensor = input_tensor[0]

    return input_tensor.numpy()


def generate_and_visualize(model, filepath):
    img = load_image(filepath=filepath)

    img = (img - 127.5) / 127.5

    generated_img = model.predict(img)

    plt.imshow(tensor_to_image(generated_img))
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    model = get_intermediate_layers(config.STYLE_LAYERS)

    outputs = model(tf.random.uniform([4, 256, 256, 3]))

    for name, output in zip(config.STYLE_LAYERS, outputs):
        print(f"Name : {name}")
        print(f"Shape: {output.shape}")
