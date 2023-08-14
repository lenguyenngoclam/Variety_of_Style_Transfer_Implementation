import gradio as gr
from train import train
import helpers

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            content_image = gr.Image(label="Content Image", type="filepath")
        with gr.Column():
            style_image = gr.Image(label="Style Image", type="filepath")
    with gr.Row():
        epoch_slider = gr.Slider(minimum=100, maximum=1000, step=10, interactive=True)
    with gr.Row():
        process_btn = gr.Button(value="Transfer")
    with gr.Row():
        result_image = gr.Image(interactive=False, label="Output Image")

    # process_btn.click(
    #     train, inputs=[epoch_slider, content_image, style_image], outputs=[result_image]
    # )

if __name__ == "__main__":
    demo.launch()
