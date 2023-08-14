import helpers
import config
from style_content_model import StyleContentModel
import tensorflow as tf

if __name__ == "__main__":
    style_image = helpers.load_image(
        "../test_image/content_image/YellowLabradorLooking_new.jpeg"
    )
    print(style_image.shape)

    extractor = StyleContentModel(
        content_layers=config.CONTENT_LAYERS, style_layers=config.STYLE_LAYERS
    )

    outputs = extractor(tf.constant(style_image))

    print("CONTENT : \n")
    for name, output in zip(config.CONTENT_LAYERS, outputs["content"]):
        print(f"Name: {name}")
        print(f"Shape: {output.numpy().shape}")
        print(f"Min: {output.numpy().min()}")
        print(f"Max: {output.numpy().max()}")
        print(f"Mean: {output.numpy().mean()}")

    print("\n\n STYLE : \n\n")

    for name, output in zip(config.STYLE_LAYERS, outputs["style"]):
        print(f"Name: {name}")
        print(f"Shape: {output.numpy().shape}")
        print(f"Min: {output.numpy().min()}")
        print(f"Max: {output.numpy().max()}")
        print(f"Mean: {output.numpy().mean()}")
