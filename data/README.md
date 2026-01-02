## Dataset

This project uses the **Malaria Cell Images Dataset** provided by
**TensorFlow Datasets (TFDS)**.

The dataset is **not included in this repository** due to size and
licensing considerations.

Instead, it is downloaded automatically at runtime using
`tensorflow_datasets`.

### Dataset Source
- TFDS Catalog: https://www.tensorflow.org/datasets/catalog/malaria

### Loading the Dataset

The dataset is loaded programmatically as follows:

```python
import tensorflow_datasets as tfds

dataset, info = tfds.load(
    "malaria",
    as_supervised=True,
    with_info=True
)
