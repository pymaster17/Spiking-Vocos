# Spiking Vocos: An Energy-Efficient Neural Vocoder

This repository is the official implementation of the paper "Spiking Vocos: An Energy-Efficient Neural Vocoder".

[中文版](./README_zh.md)

Spiking Vocos is a novel vocoder based on Spiking Neural Networks (SNNs), designed to achieve high-quality audio synthesis with ultra-low energy consumption. It is built upon the efficient Vocos framework, and by leveraging the event-driven nature of SNNs, it significantly reduces energy consumption while maintaining audio quality comparable to the original Vocos model.

## ✨ Key Features

- **Ultra-High Energy Efficiency**: Based on Spiking Neural Networks (SNNs), it replaces high-cost Multiply-Accumulate (MAC) operations with low-power Accumulate (AC) operations, making it ideal for deployment on resource-constrained edge devices.
- **Spiking ConvNeXt Module**: A specialized Spiking ConvNeXt module is designed, and an amplitude shortcut path is introduced to mitigate the information bottleneck issue in SNNs, preserving crucial signal dynamics.
- **Self-Architectural Distillation**: A self-architectural knowledge distillation strategy is employed to effectively transfer knowledge from a pre-trained ANN model (Vocos) to the SNN student model, bridging the performance gap.
- **Temporal Shift Module (TSM)**: A lightweight Temporal Shift Module is integrated to enhance the model's ability to fuse information across the temporal dimension with minimal computational overhead.

## ⚙️ Installation

First, clone this repository to your local machine:

```bash
git clone https://github.com/pymaster17/Spiking-Vocos.git
cd Spiking-Vocos
```

Then, it is recommended to create a virtual environment and install dependencies using `uv`.

```bash
# First, please install uv according to the official guide: https://astral.sh/docs/uv/installation

# Then, create and activate the virtual environment (Python 3.10 is required)
uv venv --python 3.10
source .venv/bin/activate

# Use uv sync to install dependencies defined in pyproject.toml
uv sync
```

This will install all necessary libraries from the `pyproject.toml` file, mainly including `torch`, `pytorch-lightning`, and `spikingjelly`.

## 🚀 Usage

### Inference

You can use the `inference.py` script to reconstruct waveforms from audio files. First, you need a trained model checkpoint (`.ckpt` file).

1.  Prepare a text file containing the paths to the audio files to be processed, for example, `input_filelist.txt`, with one file path per line.

2.  Run the following command for inference:

```bash
python inference.py \
    --checkpoint_path /path/to/your/model.ckpt \
    --model_type <model_type> \
    --input_files input_filelist.txt \
    --output_dir ./reconstructed_audio/
```

Where `<model_type>` is selected based on your model checkpoint type, which can be one of the following:
- `standard`: The standard Spiking Vocos model.
- `distill`: The model using knowledge distillation.
- `tsm`: The model with the TSM module integrated.
- `tsm_distill`: The model using both TSM and knowledge distillation.

### Spike Activity Visualization

This repository also provides a powerful visualization tool to generate images of spike activity in the network.

```bash
python inference.py \
    --checkpoint_path /path/to/your/model.ckpt \
    --model_type <model_type> \
    --input_files input_filelist.txt \
    --visualize_spikes \
    --visualize_output_path ./spike_activity.png
```
This command will randomly select an input audio and generate a 3D scatter plot showing the spike firing of each layer in the network.

## 🏋️ Training

The model training process is controlled by the `train.py` script and the YAML configuration files in the `configs/` directory.

1.  **Prepare Dataset File Lists**:
    Create audio file lists for your training and validation sets:
    ```bash
    find /path/to/train-dataset -name *.wav > filelist.train
    find /path/to/val-dataset -name *.wav > filelist.val
    ```

2.  **Configure Training Parameters**:
    Select a configuration file (e.g., `configs/vocos-spiking.yaml`) and modify the `train_files` and `val_files` paths to point to the file lists you just created. You can also adjust other hyperparameters as needed.

3.  **Start Training**:
    Use the following command to start training. The training process is automatically managed by PyTorch Lightning.
    ```bash
    python train.py fit --config configs/vocos-spiking.yaml
    ```
    You can choose different configuration files as needed, for example, `vocos-spiking-distill.yaml` for training a distilled model.
