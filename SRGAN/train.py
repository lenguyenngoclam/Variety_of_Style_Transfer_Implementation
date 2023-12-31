import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import losses
import data_loader
import config
from style_content_extractor import StyleContentModel
from model import StyleTransferModel
import argparse
from tqdm import tqdm
import os


def train(
    epochs,
    content_folder,
    style_image,
    checkpoint_dir: str = None,
    checkpoint_prefix: str = None,
):
    extractor = StyleContentModel()

    loader = data_loader.DataLoader(
        feature_img_folder=content_folder, style_image_path=style_image
    )

    # Load dataset
    dataset = loader.load_dataset()

    # Initialize model
    model = StyleTransferModel()

    # Initialize optimizer
    optim = Adam(learning_rate=config.LEARNING_RATE)

    # Create checkpoint
    checkpoint = tf.train.Checkpoint(model=model, optim=optim)

    for epoch in range(epochs):
        for batch_idx, (content_img, style_img) in enumerate(tqdm(dataset)):
            with tf.GradientTape() as tape:
                generated_images = model(content_img, training=True)

                # Calculate content and style outputs
                outputs = extractor(generated_images, training=False)
                content_outputs, style_outputs = outputs["content"], outputs["style"]

                # Calculate content targets
                content_targets = extractor(content_img, training=False)["content"]
                style_targets = extractor(style_img, training=False)["style"]

                # Calculate loss
                loss = losses.feature_style_loss(
                    input_images=generated_images,
                    feature_outputs=content_outputs,
                    style_outputs=style_outputs,
                    feature_targets=content_targets,
                    style_targets=style_targets,
                )

            grads = tape.gradient(loss, model.trainable_variables)
            # Update weights
            optim.apply_gradients(zip(grads, model.trainable_variables))

            if (batch_idx + 1) % 1000 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(dataset)} \
                      Content Style Loss: {loss:.4f}"
                )

        checkpoint.save(os.path.join(checkpoint_dir, checkpoint_prefix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="Number of epochs", type=int)
    parser.add_argument("--content-folder", help="Folder of content image for training")
    parser.add_argument("--style-image", help="Path of style image")
    parser.add_argument("--checkpoint-dir", help="Directory for storing checkpoint")
    parser.add_argument("--checkpoint-prefix", help="Checkpoint prefix")

    args = vars(parser.parse_args())

    train(
        args["epochs"],
        content_folder=args["content_folder"],
        style_image=args["style_image"],
        checkpoint_dir=args["checkpoint_dir"],
        checkpoint_prefix=args["checkpoint_prefix"],
    )
