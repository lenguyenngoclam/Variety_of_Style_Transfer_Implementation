LEARNING_RATE = 1e-3

FEATURE_WEIGHT = 1e10
STYLE_WEIGHT = 1e5
TOTAL_VARIATION_WEIGHT = 1e-6

# Dataset configuration
IMG_SIZE = 256
IMG_CHANNELS = 3

SHUFFLE = True
BATCH_SIZE = 4
BUFFLE_SIZE = 1000

# Style layers
STYLE_LAYERS = [
    "block1_conv2",
    "block2_conv2",
    "block3_conv3",
    "block4_conv3",
]

# Content layers
CONTENT_LAYERS = ["block2_conv2"]
