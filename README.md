# Phenotator Toolbox - TensorFlow 2

This implementation is based on the paper: **Bridging technology and ecology: enhancing applicability of deep learning and UAV-based flower recognition** (Marie Schnalke, Jonas Funk, Andreas Wagner). Accesible [here](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1498913/full).
The Phenotator Toolbox TF2 is an updated version that has been migrated to TensorFlow 2. The original toolbox can be found here: [Phenotator-Toolbox](https://github.com/gallmann/Phenotator-Toolbox), where the detailed documentation is also available. This version supports not only Faster R-CNN but also EfficientDet and SSD.

**Note:**  
It is recommended to use Linux as the operating system, as certain packages may be system-dependent.

---

## Installation

To install the toolbox, follow these steps:

### Clone the repository

Open a terminal and clone the repository:

```bash
git clone --recursive https://github.com/marieschnalke/Phenotator-Toolbox-TF2
```

### Navigate to the following directory:

```bash
cd Phenotator-Toolbox-TF2/Tensorflow/models/research
```

### Install the necessary packages:

```bash
pip install .
```

Install tf-slim:

```bash
cd ../tf-slim
pip install .
```

Now the toolbox is ready to use

The usage of the toolbox is almost the same as the original version. An additional feature is the support for SSD and EfficientDet. To use these, the corresponding lines in /utils/constants.py and image-preprocessing.py need to be uncommented.

For more detailed instructions, please refer to the original documentation.

An additional feature introduced is a dashboard providing an overview of the model predictions. To launch the dashboard, use the command ``` python cli.py dashboard ```.
