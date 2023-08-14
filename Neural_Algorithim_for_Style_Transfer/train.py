import config
import losses
import helpers
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
import matplotlib.pyplot as plt
from style_content_model import StyleContentModel


def train(epochs: int, content_image, style_image):
    """Train generated image

    @param epochs: Number of epochs to train model
    @param content_image: Content image
    @param style_image: Style image

    @return generated_image
    """
    # Initialize extractor
    extractor = StyleContentModel(
        content_layers=config.CONTENT_LAYERS, style_layers=config.STYLE_LAYERS
    )

    # Initialize generated image
    input_image = tf.Variable(content_image)

    # Initilize optimizer
    optim = keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE, epsilon=1e-1, beta_1=0.99
    )

    # Calculate content targets and style targets
    content_targets = extractor(content_image)["content"]
    style_targets = extractor(style_image)["style"]

    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            with tf.GradientTape() as tape:
                # Calculate content outputs and style outputs
                generated_outputs = extractor(input_image)

                content_outputs, style_outputs = (
                    generated_outputs["content"],
                    generated_outputs["style"],
                )

                total_loss = losses.style_content_loss(
                    image=input_image,
                    style_outputs=style_outputs,
                    content_outputs=content_outputs,
                    style_targets=style_targets,
                    content_targets=content_targets,
                )

            # Calculate gradient of total loss with respect to input image
            grad = tape.gradient(total_loss, input_image)
            # Update input image
            optim.apply_gradients([(grad, input_image)])

            # Clip value of input image
            input_image.assign(
                tf.clip_by_value(input_image, clip_value_min=0.0, clip_value_max=1.0)
            )

            pbar.set_description(f"Epoch {epoch}. Total loss : {total_loss}")
    return input_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="Epoch to train image", type=int)
    parser.add_argument("--content-image", help="Filepath of content image")
    parser.add_argument("--style-image", help="Filepath of style image")

    args = vars(parser.parse_args())

    content_image = helpers.load_image(args["content_image"])
    style_image = helpers.load_image(args["style_image"])

    generated_image = train(
        epochs=args["epochs"], content_image=content_image, style_image=style_image
    )

    plt.imshow(helpers.tensor_to_image(generated_image))
    plt.axis("off")
    plt.show()
