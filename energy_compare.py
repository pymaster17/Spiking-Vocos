import yaml
import argparse

# ==============================================================================
# 1. 定义基本参数和常量
# ==============================================================================

# 根据 SpikeLM 论文 (45nm 工艺)，32位浮点运算的能耗
ENERGY_CONSTANTS = {
    'E_MAC_pJ': 4.6,  # 乘加操作能耗 (Multiply-Accumulate), 单位: pJ
    'E_AC_pJ': 0.9,   # 累加操作能耗 (Accumulate), 单位: pJ
}

# ConvNeXtBlock 中 dwconv 层的固定 kernel_size
# 这个值在 SNN 和 ANN 的模型代码中都被硬编码为 7
DWCONV_KERNEL_SIZE = 7

# ==============================================================================
# 2. 核心计算函数
# ==============================================================================

def load_and_extract_params(config_path):
    """从 YAML 文件加载配置并提取 backbone 参数。"""
    print(f"📄 正在从 '{config_path}' 加载模型配置...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 修正路径以匹配 vocos-spiking-TSM-distill.yaml 的结构
        backbone_params = config['model']['init_args']['backbone']['init_args']
        
        # SNN 模型有 snn_timestep，ANN 模型没有，这里做兼容处理
        params = {
            'input_channels': backbone_params.get('input_channels'),
            'dim': backbone_params.get('dim'),
            'intermediate_dim': backbone_params.get('intermediate_dim'),
            'num_layers': backbone_params.get('num_layers'),
            'snn_timestep': backbone_params.get('snn_timestep', 1), # ANN默认为1
        }
        print("✅ 模型参数加载成功！")
        return params
    except FileNotFoundError:
        print(f"错误：配置文件 '{config_path}' 未找到。")
        exit(1)
    except KeyError as e:
        print(f"错误：配置文件中缺少必要的键: {e}。请检查 YAML 文件结构。")
        exit(1)
    except Exception as e:
        print(f"解析 YAML 文件时出错: {e}")
        exit(1)


def calculate_snn_convnext_energy(params, sequence_length, spike_rate):
    """计算 SNN 模型 ConvNeXt 部分的能耗。"""
    L = sequence_length
    r = spike_rate
    p = params
    T = p['snn_timestep']

    print("\n" + "="*20 + " ⚡ SNN 模型能耗计算 (ConvNeXt Only) " + "="*20)
    print(f"输入参数: L={L}, r={r:.3f}, T={T}")

    # --- ANN 部分 (dwconv) ---
    macs_dwconv_per_block = p['dim'] * DWCONV_KERNEL_SIZE * L * T
    total_macs = p['num_layers'] * macs_dwconv_per_block
    energy_ann_pj = total_macs * ENERGY_CONSTANTS['E_MAC_pJ']

    # --- SNN 部分 (pwconv1, pwconv2) ---
    potential_ops_per_pw = L * p['dim'] * p['intermediate_dim']
    acs_per_pw_per_block = potential_ops_per_pw * T * r
    total_acs = p['num_layers'] * (acs_per_pw_per_block * 2) # pwconv1 和 pwconv2
    energy_snn_pj = total_acs * ENERGY_CONSTANTS['E_AC_pJ']

    total_energy_pj = energy_ann_pj + energy_snn_pj

    print(f"ANN (MAC) 操作总数: {total_macs:,.0f}")
    print(f"SNN (AC) 操作总数 : {total_acs:,.0f}")
    print(f"SNN 模型 ConvNeXt 部分总能耗: {total_energy_pj:,.2f} pJ ({total_energy_pj / 1e6:,.6f} μJ)")
    print("="*75)
    
    return total_energy_pj


def calculate_ann_convnext_energy(params, sequence_length):
    """计算纯 ANN 模型 ConvNeXt 部分的能耗。"""
    L = sequence_length
    p = params

    print("\n" + "="*20 + " 💻 ANN 模型能耗计算 (ConvNeXt Only) " + "="*20)
    print(f"输入参数: L={L}")

    # 在纯 ANN 模型中，所有操作都是 MAC
    # 1. dwconv
    macs_dwconv_per_block = p['dim'] * DWCONV_KERNEL_SIZE * L
    # 2. pwconv1
    macs_pwconv1_per_block = L * p['dim'] * p['intermediate_dim']
    # 3. pwconv2
    macs_pwconv2_per_block = L * p['intermediate_dim'] * p['dim']

    total_macs_per_block = macs_dwconv_per_block + macs_pwconv1_per_block + macs_pwconv2_per_block
    total_macs = p['num_layers'] * total_macs_per_block
    total_energy_pj = total_macs * ENERGY_CONSTANTS['E_MAC_pJ']

    print(f"ANN (MAC) 操作总数: {total_macs:,.0f}")
    print(f"ANN 模型 ConvNeXt 部分总能耗: {total_energy_pj:,.2f} pJ ({total_energy_pj / 1e6:,.6f} μJ)")
    print("="*75)

    return total_energy_pj

# ==============================================================================
# 3. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="对比 SNN 和 ANN 模型 ConvNeXt 部分的能耗。")
    parser.add_argument(
        '--snn-config', 
        type=str, 
        default='configs/vocos-spiking-TSM-distill.yaml',
        help="指向 SNN 模型配置文件 (vocos-spiking-TSM-distill.yaml) 的路径。"
    )
    parser.add_argument(
        '--ann-config', 
        type=str, 
        default='configs/vocos.yaml',
        help="指向纯 ANN 模型配置文件 (vocos.yaml) 的路径。"
    )
    parser.add_argument(
        '--length', 
        type=int, 
        default=1000, 
        help="输入序列的长度 (L)。"
    )
    parser.add_argument(
        '--rate', 
        type=float, 
        default=0.176, 
        help="SNN 神经元的平均脉冲发放率 (r)，通常在 0.1 到 0.3 之间。"
    )
    
    args = parser.parse_args()

    # 加载两种模型的参数
    snn_params = load_and_extract_params(args.snn_config)
    ann_params = load_and_extract_params(args.ann_config)

    # 分别计算能耗
    snn_energy = calculate_snn_convnext_energy(snn_params, args.length, args.rate)
    ann_energy = calculate_ann_convnext_energy(ann_params, args.length)

    # 打印最终对比总结
    print("\n" + "#"*25 + " 📊 最终能耗对比总结 " + "#"*25)
    print(f"序列长度 (L) = {args.length}, 脉冲发放率 (r) = {args.rate:.3f}\n")
    print(f"ANN 模型能耗: {ann_energy:,.2f} pJ")
    print(f"SNN 模型能耗: {snn_energy:,.2f} pJ\n")
    
    if ann_energy > 0:
        energy_ratio = snn_energy / ann_energy
        print(f"SNN 能耗是 ANN 的 {energy_ratio:.2%}。")
        print(f"相比 ANN，SNN 节约了 {(1 - energy_ratio):.2%} 的能耗。")
    print("#"*75)
