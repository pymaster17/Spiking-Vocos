# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torchaudio
from tqdm import tqdm
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

# Configuration
INPUT_FILE_LIST = "filelist.test"
OUTPUT_DIR = "results/hifigan"
TARGET_SAMPLE_RATE = 22050
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_audio():
    """
    Loads audio files from a list, generates mel spectrograms,
    then reconstructs the audio using a pretrained HiFi-GAN vocoder
    and saves the output. Also generates scp files for ground truth and predictions.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load pretrained HiFi-GAN vocoder
    print(f"Loading HiFi-GAN model on {DEVICE}...")
    try:
        hifi_gan = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-22050Hz",
            savedir="hifigan",
            run_opts={"device": DEVICE}
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have an internet connection to download the model.")
        return

    # Read the list of input audio files
    try:
        with open(INPUT_FILE_LIST, 'r', encoding='utf-8') as f:
            filepaths = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file list '{INPUT_FILE_LIST}' not found.")
        return

    print(f"Found {len(filepaths)} audio files to process.")

    # Prepare scp files
    gt_scp_path = os.path.join(OUTPUT_DIR, "gt.scp")
    pred_scp_path = os.path.join(OUTPUT_DIR, "pred.scp")
    
    with open(gt_scp_path, 'w', encoding='utf-8') as gt_scp, \
         open(pred_scp_path, 'w', encoding='utf-8') as pred_scp:
        
        # Process each audio file
        for filepath in tqdm(filepaths, desc="Processing audio files"):
            try:
                # Generate file ID from filename (without extension)
                base_filename = os.path.basename(filepath)
                file_id = os.path.splitext(base_filename)[0]
                
                # Load the audio file
                signal, rate = torchaudio.load(filepath)
                signal = signal.to(DEVICE)

                # Resample audio to the target sample rate if necessary
                if rate != TARGET_SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=TARGET_SAMPLE_RATE).to(DEVICE)
                    signal = resampler(signal)

                # Convert to mono by averaging channels if stereo
                if signal.shape[0] > 1:
                    signal = torch.mean(signal, dim=0, keepdim=True)

                # Compute mel spectrogram
                # The audio signal needs to be on CPU for mel_spectogram
                # Squeeze to (time,)
                mel_spec, _ = mel_spectogram(
                    audio=signal.squeeze(0).cpu(),
                    sample_rate=TARGET_SAMPLE_RATE,
                    hop_length=256,
                    win_length=1024,
                    n_mels=80,
                    n_fft=1024,
                    f_min=0.0,
                    f_max=8000.0,
                    power=1,
                    normalized=False,
                    min_max_energy_norm=True,
                    norm="slaney",
                    mel_scale="slaney",
                    compression=True,
                )

                # Move spectrogram to device and add batch dimension
                mel_spec = mel_spec.to(DEVICE).unsqueeze(0)

                # Use HiFi-GAN to convert spectrogram back to waveform
                waveforms = hifi_gan.decode_batch(mel_spec)

                # Save the reconstructed waveform
                output_path = os.path.join(OUTPUT_DIR, base_filename)
                torchaudio.save(output_path, waveforms.cpu().squeeze(1), TARGET_SAMPLE_RATE)
                
                # Convert to absolute paths
                abs_output_path = os.path.abspath(output_path)
                
                # Write to scp files
                gt_scp.write(f"{file_id} {filepath}\n")
                pred_scp.write(f"{file_id} {abs_output_path}\n")

            except Exception as e:
                print(f"Failed to process {filepath}: {e}")

    print("All audio files processed successfully.")
    print(f"Reconstructed files are saved in '{OUTPUT_DIR}' directory.")
    print(f"Ground truth scp file: {gt_scp_path}")
    print(f"Prediction scp file: {pred_scp_path}")

if __name__ == "__main__":
    process_audio()
