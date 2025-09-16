import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def find_latest_version_dir(base_dir):
    """Finds the latest version directory (e.g., 'version_0', 'version_1') in a PyTorch Lightning log dir."""
    if not os.path.isdir(base_dir):
        return None
    
    version_dirs = [d for d in os.listdir(base_dir) if d.startswith("version_") and os.path.isdir(os.path.join(base_dir, d))]
    if not version_dirs:
        return None
        
    version_dirs.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
    return os.path.join(base_dir, version_dirs[0])

def extract_scalar_data(log_dir, scalar_tag):
    """Extracts scalar data from a TensorBoard log directory."""
    ea = event_accumulator.EventAccumulator(log_dir,
                                            size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    
    if scalar_tag not in ea.Tags()['scalars']:
        print(f"Warning: Scalar tag '{scalar_tag}' not found in {log_dir}")
        return [], []
        
    scalar_events = ea.Scalars(scalar_tag)
    
    steps = [e.step for e in scalar_events]
    values = [e.value for e in scalar_events]
    
    return steps, values

def main():
    # --- Configuration ---
    EXPERIMENTS = ['spikingv3', 'spikingv3-TSM','spikingv3-distill', 'spikingv3-TSM-distill']
    LOGS_ROOT = 'logs'
    UTMOS_TAG = 'val/utmos_score'
    PESQ_TAG = 'val/pesq_score'

    # --- Find latest log directories for each experiment ---
    name_to_logdir = {}
    for name in EXPERIMENTS:
        base_dir = os.path.join(LOGS_ROOT, name)
        latest_dir = find_latest_version_dir(base_dir)
        if latest_dir is None:
            print(f"Warning: 未找到实验 '{name}' 的日志目录（期望位置：{base_dir} 下的 version_*）")
            continue
        name_to_logdir[name] = latest_dir

    if not name_to_logdir:
        print("Error: 未找到任何有效的实验日志目录，请检查 EXPERIMENTS 与 LOGS_ROOT 配置。")
        return

    for name, path in name_to_logdir.items():
        print(f"Found {name} log: {path}")

    # --- Extract Data ---
    print("Extracting data from TensorBoard logs...")
    utmos_data = {}
    pesq_data = {}

    for name, log_dir in name_to_logdir.items():
        utmos_steps, utmos_values = extract_scalar_data(log_dir, UTMOS_TAG)
        pesq_steps, pesq_values = extract_scalar_data(log_dir, PESQ_TAG)
        utmos_data[name] = (utmos_steps, utmos_values)
        pesq_data[name] = (pesq_steps, pesq_values)

    if not any(len(v[1]) > 0 for v in utmos_data.values()):
        print("Error: 未能从任一实验中提取到 UTMOS 数据，请检查日志与标签名。")
        return

    # --- Plotting ---
    print("Generating plot...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Vocoder Performance Comparison', fontsize=16)

    # UTMOS Score Plot
    for name in EXPERIMENTS:
        if name not in utmos_data:
            continue
        steps, values = utmos_data[name]
        if len(values) == 0:
            print(f"Warning: 实验 '{name}' 缺少 UTMOS 数据，已跳过该曲线。")
            continue
        axes[0].plot(steps, values, label=name, alpha=0.8)
    axes[0].set_ylabel('UTMOS Score')
    axes[0].set_title('UTMOS Score Comparison')
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # PESQ Score Plot
    for name in EXPERIMENTS:
        if name not in pesq_data:
            continue
        steps, values = pesq_data[name]
        if len(values) == 0:
            print(f"Warning: 实验 '{name}' 缺少 PESQ 数据，已跳过该曲线。")
            continue
        axes[1].plot(steps, values, label=name, alpha=0.8)
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('PESQ Score')
    axes[1].set_title('PESQ Score Comparison')
    axes[1].legend()
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = 'metrics_comparison.png'
    plt.savefig(output_filename)
    print(f"Plot saved successfully to {output_filename}")

if __name__ == '__main__':
    main() 