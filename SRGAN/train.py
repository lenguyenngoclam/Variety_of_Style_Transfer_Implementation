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
    X_dataset, y_image = loader.load_dataset()

    # Extract feature targets
    style_targets = extractor(y_image)["style"]
    style_targets = [
        tf.repeat(style_target, [config.BATCH_SIZE], axis=0)
        for style_target in style_targets
    ]

    # Initialize model
    model = StyleTransferModel()

    # Initialize optimizer
    optim = Adam(learning_rate=config.LEARNING_RATE)

    # Create checkpoint
    checkpoint = tf.train.Checkpoint(model=model, optim=optim)

    for epoch in range(epochs):
        for batch_idx, input_images in enumerate(tqdm(X_dataset)):
            input_images = input_images[0]
            with tf.GradientTape() as tape:
                generated_images = model(input_images)

                # Calculate content and style outputs
                outputs = extractor(generated_images)
                content_outputs, style_outputs = outputs["content"], outputs["style"]

                # Calculate content targets
                content_targets = extractor(input_images)["content"]

                # Calculate loss
                loss = losses.feature_style_loss(
                    input_images=input_images,
                    feature_outputs=content_outputs,
                    style_outputs=style_outputs,
                    feature_targets=content_targets,
                    style_targets=style_targets,
                )

            grads = tape.gradient(loss, list(model.trainable_variables))
            # Update weights
            optim.apply_gradients(zip(grads, list(model.trainable_variables)))

            if (batch_idx + 1) % 1000 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(X_dataset)} \
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
