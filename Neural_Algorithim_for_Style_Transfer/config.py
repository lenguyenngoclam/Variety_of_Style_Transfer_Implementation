CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2
TOTAL_VARIATION_WEIGHT = 30

LEARNING_RATE = 2e-2

# We use the the content representation on layer 'block5_conv2'
CONTENT_LAYERS = ["block5_conv2"]

# We use the style representation on layer 'block1_conv1',
# 'block2_conv1', 'block3_conv1', block4_conv1', 'block5_conv1'
STYLE_LAYERS = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
