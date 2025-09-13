# Spiking Vocos: 一款高能效神经声码器

本仓库是论文 [Spiking Vocos: An Energy-Efficient Neural Vocoder]的官方实现。

Spiking Vocos 是一款基于脉冲神经网络（SNN）的新型声码器，旨在以超低能耗实现高质量的音频合成。它构建于高效的 [Vocos]框架之上，通过利用 SNN 的事件驱动特性，在保持与原始 Vocos 模型相当的音频质量的同时，显著降低了能量消耗。

## ✨ 主要特性

- **超高能效**: 基于脉冲神经网络（SNN），将高计算量的乘累加（MAC）操作替换为低功耗的累加（AC）操作，非常适合在计算资源受限的边缘设备上部署。
- **Spiking ConvNeXt 模块**: 设计了专门的 Spiking ConvNeXt 模块，并引入幅度快捷路径（amplitude shortcut path）来缓解 SNN 的信息瓶颈问题，保留关键的信号动态。
- **自架构蒸馏**: 采用一种自架构知识蒸馏策略，有效将预训练的 ANN 模型（Vocos）的知识迁移到 SNN 学生模型，以弥合性能差距。
- **时间移位模块 (TSM)**: 集成了轻量级的时间移位模块，以极小的计算开销增强了模型在时间维度上融合信息的能力。

## ⚙️ 安装

首先，克隆本仓库到本地：

```bash
git clone https://github.com/pymaster17/Spiking-Vocos.git
cd Spiking-Vocos
```

然后，建议使用 `uv` 创建虚拟环境并安装依赖。

```bash
# 首先，请根据官方指南安装 uv: https://astral.sh/docs/uv/installation

# 然后，创建并激活虚拟环境 (需要 Python 3.10)
uv venv --python 3.10
source .venv/bin/activate

# 使用 uv sync 安装 pyproject.toml 中定义的依赖
uv sync
```

这将从 `pyproject.toml` 文件中安装所有必需的库，主要包括 `torch`, `pytorch-lightning` 和 `spikingjelly`。

## 🚀 使用方法

### 推理

你可以使用 `inference.py` 脚本从音频文件重建波形。首先，你需要一个训练好的模型检查点（`.ckpt` 文件）。

1.  准备一个包含待处理音频文件路径的文本文件，例如 `input_filelist.txt`，每行一个文件路径。

2.  运行以下命令进行推理：

```bash
python inference.py \
    --checkpoint_path /path/to/your/model.ckpt \
    --model_type <model_type> \
    --input_files input_filelist.txt \
    --output_dir ./reconstructed_audio/
```

其中 `<model_type>` 根据你的模型检查点类型选择，可以是以下几种：
- `standard`: 标准的 Spiking Vocos 模型。
- `distill`: 使用了知识蒸馏的模型。
- `tsm`: 集成了 TSM 模块的模型。
- `tsm_distill`: 同时使用 TSM 和知识蒸馏的模型。

### 脉冲活动可视化

本仓库还提供了一个强大的可视化工具，可以生成网络中脉冲活动的可视化图像。

```bash
python inference.py \
    --checkpoint_path /path/to/your/model.ckpt \
    --model_type <model_type> \
    --input_files input_filelist.txt \
    --visualize_spikes \
    --visualize_output_path ./spike_activity.png
```
该命令会随机选择一个输入音频，并生成一个展示网络各层脉冲发放情况的 3D 散点图。

## 🏋️ 训练

模型的训练流程由 `train.py` 脚本和 `configs/` 目录下的 YAML 配置文件控制。

1.  **准备数据集文件列表**:
    为你的训练集和验证集创建音频文件列表：
    ```bash
    find /path/to/train-dataset -name *.wav > filelist.train
    find /path/to/val-dataset -name *.wav > filelist.val
    ```

2.  **配置训练参数**:
    选择一个配置文件（例如 `configs/vocos-spiking.yaml`），并修改其中的 `train_files` 和 `val_files` 路径，指向你刚刚创建的文件列表。你也可以根据需要调整其他超参数。

3.  **开始训练**:
    使用以下命令启动训练。训练过程由 PyTorch Lightning 自动管理。
    ```bash
    python train.py fit --config configs/vocos-spiking.yaml
    ```
    你可以根据需要选择不同的配置文件，例如 `vocos-spiking-distill.yaml` 用于训练蒸馏模型。