import tensorflow as tf
import os
import config


class DataLoader:
    def __init__(self, feature_img_folder, style_image_path):
        self.feature_img_folder = tf.data.Dataset.list_files(
            os.path.join(feature_img_folder, "*.jpg")
        )
        self.style_image_path = style_image_path

    def load_image(self, img_path):
        img = tf.io.read_file(img_path)

        img = tf.io.decode_image(img, channels=config.IMG_CHANNELS)

        img = tf.cast(img, tf.float32)

        return img

    def preprocessing_image(self, image):
        # Resize
        image = tf.image.resize(image, [config.IMG_SIZE, config.IMG_SIZE])
        # Rescale to [-1, 1]
        image = (image - 127.5) / 127.5

        return image

    def configure_for_performance(self, dataset):
        if config.SHUFFLE:
            dataset = dataset.shuffle(buffer_size=config.BUFFLE_SIZE)
        dataset = dataset.batch(config.BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def load_dataset(self):
        X_dataset = self.feature_img_folder.map(
            lambda x: tf.numpy_function(self.load_image, [x], [tf.float32]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        X_dataset = X_dataset.map(
            lambda x: tf.numpy_function(self.preprocessing_image, [x], [tf.float32]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        X_dataset = self.configure_for_performance(X_dataset)

        y_img = self.load_image(self.style_image_path)
        y_img = self.preprocessing_image(y_img)
        y_img = tf.expand_dims(y_img, axis=0)

        return X_dataset, y_img


if __name__ == "__main__":
    loader = DataLoader(
        "../test_image/content_image", "../test_image/style_image/artwork_2.jpeg"
    )
    X_dataset, y_dataset = loader.load_dataset()

    print(X_dataset, y_dataset)
