import torch
import torchaudio
import argparse
import os
import yaml
from tqdm import tqdm
import numpy as np
import random


def plot_spike_visualization(records, output_path, threshold=0.5, sample_ratio=0.1):
    """
    Generates and saves a 3D scatter plot of spike firing rates across network layers.
    This function defaults to showing a specific subset of layers (0, 4, 8, 14).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use a non-interactive backend to prevent opening a window
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("\n--- Visualization Error ---")
        print("Matplotlib is required for visualization. Please install it: pip install matplotlib")
        return

    # Default behavior: filter for a specific subset of layers for targeted analysis.
    layers_to_plot = [0, 4, 8, 14]
    print(f"Defaulting to show specific layers for analysis: {layers_to_plot}")
    
    original_indices = list(range(len(records)))
    filtered_records = []
    filtered_indices = []

    # Ensure specified layers are within bounds and filter
    valid_layers_to_plot = [l for l in layers_to_plot if l < len(records)]
    for i, record in enumerate(records):
        if i in valid_layers_to_plot:
            filtered_records.append(record)
            filtered_indices.append(i)
    
    records = filtered_records
    original_indices = filtered_indices

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Colors to cycle through for different layers, similar to the example image
    colors = ['#a9cce3', '#5499c7', '#f8c471', '#af7ac5', '#7dcea0', '#f1948a', '#f5b7b1', '#aed6f1']
    
    num_layers = len(records)
    if num_layers == 0:
        print("No spike data recorded to generate visualization.")
        return

    print(f"Visualizing data from {num_layers} spiking layers...")

    for i, spike_tensor in enumerate(records):
        # We want to plot layers from top to bottom, consistent with network depth.
        # So, the shallowest layer (i=0) gets the highest z-coordinate.
        z = num_layers - 1 - i

        # The tensor shape is [T, B, C, L]. B is 1 during inference.
        # We average over the time dimension (T) to get the firing rate at each (C, L) position.
        firing_rate_map = spike_tensor.mean(dim=0).squeeze(0).cpu().numpy()

        # Find the coordinates (channel, sequence) where the firing rate is greater than the threshold.
        channel_coords, seq_coords = np.where(firing_rate_map > threshold)
        
        # Use the original layer index for logging
        original_layer_idx = original_indices[i]
        num_found = len(channel_coords)

        if num_found == 0:
            print(f"  - Layer {original_layer_idx}: No spikes detected above threshold {threshold}.")
            continue  # Skip layers with no spikes
        
        # Apply random sampling if the ratio is less than 1.0
        num_to_plot = int(num_found * sample_ratio)
        if num_to_plot < num_found:
            print(f"  - Layer {original_layer_idx}: Found {num_found} firing neurons above threshold, randomly sampling {num_to_plot} for plotting.")
            indices = np.random.choice(num_found, num_to_plot, replace=False)
            channel_coords = channel_coords[indices]
            seq_coords = seq_coords[indices]
        else:
            print(f"  - Layer {original_layer_idx}: Found {num_found} firing neurons above threshold.")


        # Create an array of z-coordinates for all points in this layer.
        z_coords = np.full_like(channel_coords, z)

        # Plot the points for the current layer.
        ax.scatter(seq_coords, channel_coords, z_coords, s=5, alpha=0.8, color=colors[i % len(colors)])

    ax.set_xlabel('Sequence', labelpad=10)
    ax.set_ylabel('Channel', labelpad=10)
    ax.set_zlabel('Depth', labelpad=0)
    
    # Set the z-axis ticks to correspond to the ConvNeXt block index.
    convnext_block_labels = [idx // 2 for idx in original_indices]
    ax.set_zticks(np.arange(num_layers))
    ax.set_zticklabels(reversed(convnext_block_labels))
    
    # Adjust view angle to be similar to the reference image.
    ax.view_init(elev=20., azim=-65)
    ax.dist = 11 # Zoom out a bit to see all data
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\nVisualization successfully saved to {output_path}")
    except Exception as e:
        print(f"\nError saving visualization: {e}")

# Ensure the script can find the vocos module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vocos.experiment import VocosExp
from vocos.experiment_distill import VocosDistillExp
from vocos.experiment_TSM_distill import VocosDistillExp as VocosTSMDistillExp
from vocos.pretrained import instantiate_class


def main():
    parser = argparse.ArgumentParser(description="Inference script for trained Vocos and Spiking-Vocos models.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt) file."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['standard', 'distill', 'tsm', 'tsm_distill'],
        help="Type of the model to load. 'standard' for VocosExp (including non-distilled spiking models), 'distill' for VocosDistillExp, 'tsm' for VocosExp with TSM backbone, and 'tsm_distill' for the TSM-based distilled model."
    )
    parser.add_argument(
        "--input_files",
        type=str,
        required=True,
        help="Path to a text file containing a list of input WAV files, one per line."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the reconstructed audio files."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (e.g., 'cuda' or 'cpu')."
    )
    parser.add_argument(
        "--visualize_spikes",
        action="store_true",
        help="Enable spike visualization mode. This will process one random audio file and generate a 3D plot of spike activity."
    )
    parser.add_argument(
        "--visualize_output_path",
        type=str,
        default="spike_visualization.png",
        help="Path to save the spike visualization plot."
    )
    parser.add_argument(
        "--plot_threshold",
        type=float,
        default=0.5,
        help="Minimum firing rate for a neuron to be included in the visualization plot. Recommended: 0.1-0.5. Default: 0.0."
    )
    parser.add_argument(
        "--plot_sample_ratio",
        type=float,
        default=0.1,
        help="Ratio of firing neurons (after thresholding) to randomly sample for plotting (e.g., 0.1 for 10%%). Default: 1.0."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Model Loading with manual instantiation ---
    print(f"Loading a '{args.model_type}' model from {args.checkpoint_path}...")
    try:
        # Find and load the config file
        config_path = os.path.join(os.path.dirname(args.checkpoint_path), "..", "config.yaml")
        if not os.path.exists(config_path):
             # Try another common path for PL checkpoints
            config_path = os.path.join(os.path.dirname(args.checkpoint_path), "..", "hparams.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found. Looked in {os.path.dirname(config_path)}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Instantiate model components from config
        if 'model' in config and 'init_args' in config['model']:
             # Handle nested config for some training setups
            model_config = config['model']['init_args']
        else:
            # Handle flat config from standard Lightning hparams.yaml
            model_config = config
        
        feature_extractor = instantiate_class(args=(), init=model_config['feature_extractor'])
        backbone = instantiate_class(args=(), init=model_config['backbone'])
        head = instantiate_class(args=(), init=model_config['head'])

        # Select model class based on user input
        if args.model_type == 'distill':
            model_class = VocosDistillExp
        elif args.model_type == 'tsm_distill':
            model_class = VocosTSMDistillExp
        else:  # standard or tsm
            model_class = VocosExp
        
        # Load the model with instantiated components
        model = model_class.load_from_checkpoint(
            args.checkpoint_path,
            feature_extractor=feature_extractor,
            backbone=backbone,
            head=head,
            map_location="cpu",
            strict=False
        )
        model.eval()
        model.to(args.device)
        print("Model loaded successfully.")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the checkpoint path is correct and all required model class definitions are available.")
        return

    # --- Firing Rate Monitoring Setup ---
    spike_monitor = None
    all_firing_rates = []
    try:
        from spikingjelly.activation_based import monitor
        from spikingjelly.activation_based.neuron import ParametricLIFNode

        # Check if the model has spiking neurons to monitor
        if any(isinstance(m, ParametricLIFNode) for m in model.modules()):
            if args.visualize_spikes:
                print("Setting up monitor for spike visualization (collecting raw spike tensors).")
                spike_monitor = monitor.OutputMonitor(model, ParametricLIFNode)
            else:
                print("Spiking neurons (ParametricLIFNode) detected. Setting up firing rate monitor.")
                
                def get_mean_firing_rate(spike_seq: torch.Tensor):
                    """
                    Calculates the mean of a tensor, which for a spike train tensor,
                    is its average firing rate.
                    """
                    return spike_seq.mean().item()

                spike_monitor = monitor.OutputMonitor(model, ParametricLIFNode, function_on_output=get_mean_firing_rate)
        else:
            print("No spiking neurons (ParametricLIFNode) found in the model. Skipping firing rate monitoring.")
    except ImportError:
        print("Warning: SpikingJelly library not found. Firing rate monitoring will be skipped. "
              "Please install it (`pip install spikingjelly`) to enable this feature.")
    except Exception as e:
        print(f"An error occurred during monitor setup: {e}")

    # Read the list of input files
    with open(args.input_files, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]

    # --- Visualization Mode ---
    if args.visualize_spikes:
        if not spike_monitor:
            print("\nCannot visualize: No spiking neurons found or SpikingJelly is not installed.")
            return
        
        if not file_list:
            print("\nCannot visualize: Input file list is empty.")
            return

        file_path = random.choice(file_list)
        print(f"\n--- Visualization Mode ---")
        print(f"Processing a random sample for visualization: {file_path}")
        
        try:
            # Load audio
            audio, sr = torchaudio.load(file_path)
            duration_sec = audio.shape[1] / sr
            audio = audio.to(args.device)

            # If stereo, convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != model.hparams.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, model.hparams.sample_rate).to(args.device)
                audio = resampler(audio)

            # Perform inference to populate monitor
            with torch.no_grad():
                print("Performing forward pass...")
                model(audio)
            
            # Calculate and print audio stats before visualization
            if spike_monitor and spike_monitor.records:
                # Calculate overall average firing rate for this specific audio file across the monitored layers
                avg_firing_rate = torch.cat([t.flatten() for t in spike_monitor.records]).mean().item()
                print(f"\n--- Audio Sample Statistics ---")
                print(f"Audio duration: {duration_sec:.2f} seconds")
                print(f"Average firing rate (for visualized layers): {avg_firing_rate:.6f}\n")

            # Generate and save the plot
            plot_spike_visualization(
                spike_monitor.records, 
                args.visualize_output_path,
                threshold=args.plot_threshold,
                sample_ratio=args.plot_sample_ratio
            )

        except Exception as e:
            print(f"Failed to process {file_path} for visualization: {e}")
        finally:
            if spike_monitor:
                spike_monitor.remove_hooks()
        
        return  # Exit after visualization

    # --- Standard Inference Mode ---
    print(f"Found {len(file_list)} files to process.")

    gt_scp_content = []
    pred_scp_content = []

    # Process each file
    for file_path in tqdm(file_list, desc="Processing files"):
        try:
            # Load audio
            audio, sr = torchaudio.load(file_path)
            audio = audio.to(args.device)

            # If stereo, convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != model.hparams.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, model.hparams.sample_rate).to(args.device)
                audio = resampler(audio)

            # Perform inference
            with torch.no_grad():
                # The forward pass is consistent for both VocosExp and VocosDistillExp
                audio_hat = model(audio)

            # Collect firing rates if monitor is active
            if spike_monitor:
                # spike_monitor.records will be a list of mean firing rates for each PLIF layer
                all_firing_rates.extend(spike_monitor.records)
                spike_monitor.clear_recorded_data()

            # Save the output audio
            output_filename = os.path.basename(file_path)
            output_path = os.path.join(args.output_dir, output_filename)
            torchaudio.save(output_path, audio_hat.cpu(), model.hparams.sample_rate)

            # Generate SCP file entries
            utt_id = os.path.splitext(output_filename)[0]
            gt_scp_content.append(f"{utt_id} {os.path.abspath(file_path)}")
            pred_scp_content.append(f"{utt_id} {os.path.abspath(output_path)}")

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    # Write SCP files
    if gt_scp_content:
        with open(os.path.join(args.output_dir, "gt.scp"), "w") as f_gt:
            f_gt.write("\n".join(gt_scp_content))
        with open(os.path.join(args.output_dir, "pred.scp"), "w") as f_pred:
            f_pred.write("\n".join(pred_scp_content))
        print(f"\nSuccessfully generated gt.scp and pred.scp in {args.output_dir}")

    if spike_monitor:
        if all_firing_rates:
            average_firing_rate = np.mean(all_firing_rates)
            print(f"\n--- Firing Rate Statistics ---")
            print(f"Overall average firing rate of the network across all files and layers: {average_firing_rate:.6f}")
        else:
            print("\n--- Firing Rate Statistics ---")
            print("No firing rates were recorded for any file.")
        spike_monitor.remove_hooks()
    
    print(f"Inference complete. Reconstructed audio saved in {args.output_dir}")


if __name__ == "__main__":
    main() 