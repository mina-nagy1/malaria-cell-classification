# Training configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.008

# Evaluation configuration
# Optimized decision threshold based on ROC analysis
CLASSIFICATION_THRESHOLD = 0.672

# Class mapping
PARASITIZED_CLASS = 0
UNINFECTED_CLASS = 1
