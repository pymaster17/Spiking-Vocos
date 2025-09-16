import torch
from torch import nn
from torch.nn import functional as F
import os
import glob
import yaml
from vocos.experiment import VocosExp
from vocos.models import Backbone, VocosBackbone
from vocos.modules import FeatureAdapter
from vocos.loss import PhaseLoss, DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss
from vocos.pretrained import Vocos, instantiate_class
from vocos.feature_extractors import FeatureExtractor
from vocos.heads import FourierHead, ISTFTHead
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
import transformers


class VocosDistillExp(VocosExp):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        sample_rate: int,
        initial_learning_rate: float,
        teacher_model_path: str,
        phase_loss_coeff: float,
        feature_distill_coeff: float,
        output_distill_coeff: float,
        num_warmup_steps: int,
        mel_loss_coeff: float,
        mrd_loss_coeff: float,
        pretrain_mel_steps: int,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
    ):
        # Call the parent constructor, but we will override some of its properties
        super().__init__(
            feature_extractor=feature_extractor,
            backbone=backbone,
            head=head,
            sample_rate=sample_rate,
            initial_learning_rate=initial_learning_rate,
            num_warmup_steps=num_warmup_steps,
            mel_loss_coeff=mel_loss_coeff,
            mrd_loss_coeff=mrd_loss_coeff,
            pretrain_mel_steps=pretrain_mel_steps,
            decay_mel_coeff=decay_mel_coeff,
            evaluate_utmos=evaluate_utmos,
            evaluate_pesq=evaluate_pesq,
            evaluate_periodicty=evaluate_periodicty,
        )
        # Manually save hyperparameters for the distillation experiment
        self.save_hyperparameters('teacher_model_path', 'feature_distill_coeff', 'output_distill_coeff')

        # Instantiate losses for distillation
        self.feature_loss = nn.MSELoss()
        if isinstance(head, ISTFTHead):
            self.phase_loss = PhaseLoss(n_fft=head.n_fft)
        else:
            self.phase_loss = None

        # Re-initialize GAN components as they are not set by the parent's __init__ when overridden
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()
        self.discriminator = nn.ModuleList([self.multiperioddisc, self.multiresddisc])
        
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.mel_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)

    def configure_optimizers(self):
        gen_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.backbone.parameters()},
            {"params": self.head.parameters()},
            {"params": self.feature_adapters.parameters()},
        ]
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresddisc.parameters()},
        ]

        opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))

        max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )

        return (
            [opt_disc, opt_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )
        
    def setup(self, stage: str):
        # Load teacher model
        teacher_path = self.hparams.teacher_model_path
        if os.path.isdir(teacher_path):
            # It's a local directory from a Pytorch Lightning run
            config_path = os.path.join(teacher_path, "config.yaml")
            
            # Find the checkpoint file, preferring last.ckpt
            checkpoint_dir = os.path.join(teacher_path, "checkpoints")
            ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
            if not os.path.exists(ckpt_path):
                ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
                if not ckpt_files:
                    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
                # simple heuristic: take the first one found.
                ckpt_path = os.path.join(checkpoint_dir, ckpt_files[0])
            
            # Load config and instantiate model parts
            with open(config_path, 'r') as f:
                teacher_config = yaml.safe_load(f)
            
            model_config = teacher_config['model']['init_args']
            feature_extractor = instantiate_class(args=(), init=model_config['feature_extractor'])
            backbone = instantiate_class(args=(), init=model_config['backbone'])
            head = instantiate_class(args=(), init=model_config['head'])

            # Create the Vocos model
            self.teacher_model = Vocos(feature_extractor=feature_extractor, backbone=backbone, head=head)

            # Load weights
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # In lightning checkpoints, the model weights are under the 'state_dict' key.
            # Lightning checkpoints have module prefixes that need to be removed
            state_dict_raw = ckpt['state_dict']
            state_dict = {}
            for k, v in state_dict_raw.items():
                if k.startswith('feature_extractor.'):
                    state_dict[k] = v
                elif k.startswith('backbone.'):
                    state_dict[k] = v
                elif k.startswith('head.'):
                    state_dict[k] = v
            # Load only the generator components, ignore discriminator weights
            self.teacher_model.load_state_dict(state_dict, strict=False)
        else:
            # Assume it's a HuggingFace Hub repo ID
            self.teacher_model = Vocos.from_pretrained(teacher_path)

        self.teacher_model.eval()
        self.teacher_model.to(self.device)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Ensure student is a spiking model and teacher is a standard vocos model
        assert hasattr(self.backbone, 'snn_timestep'), "Student model must be a spiking model with 'snn_timestep' attribute"
        assert isinstance(self.teacher_model.backbone, VocosBackbone), "Teacher model must be a VocosBackbone"
        
        # Ensure student and teacher have the same number of layers for feature distillation
        assert self.backbone.num_layers == self.teacher_model.backbone.num_layers, \
            f"Student model has {self.backbone.num_layers} layers, but teacher model has {self.teacher_model.backbone.num_layers} layers"

        self.feature_adapters = nn.ModuleList([
            FeatureAdapter(
                in_channels=self.backbone.dim, 
                out_channels=self.teacher_model.backbone.dim,
                snn_timestep=self.backbone.snn_timestep
            ) for _ in range(self.backbone.num_layers)
        ])

    def training_step(self, batch, batch_idx, optimizer_idx):
        audio = batch

        # train discriminator
        if optimizer_idx == 0:
            if not self.train_discriminator:
                return None  # Skip discriminator training during warmup

            with torch.no_grad():
                # Student forward pass without hidden states for efficiency
                student_features = self.feature_extractor(audio)
                student_backbone_out = self.backbone(student_features, output_hidden_states=False)
                student_audio_hat, _, _ = self.head(student_backbone_out, return_spec=True)

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio, y_hat=student_audio_hat.detach())
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio, y_hat=student_audio_hat.detach())
            loss_mp, loss_mp_real, _ = self.disc_loss(disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp)
            loss_mrd, loss_mrd_real, _ = self.disc_loss(disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd)
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            disc_loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd

            self.log("loss/disc/total", disc_loss, prog_bar=True)
            self.log("loss/disc/multi_period_loss", loss_mp)
            self.log("loss/disc/multi_res_loss", loss_mrd)
            return disc_loss

        # train generator
        if optimizer_idx == 1:
            self.teacher_model.eval()

            # Shared arguments for teacher and student models
            shared_kwargs = {"output_hidden_states": True}

            # Teacher forward pass
            with torch.no_grad():
                teacher_features = self.teacher_model.feature_extractor(audio)
                teacher_backbone_out, teacher_hidden_states = self.teacher_model.backbone(
                    teacher_features, **shared_kwargs
                )
                _, teacher_mag, teacher_phase = self.teacher_model.head(teacher_backbone_out, return_spec=True)

            # Student forward pass
            student_features = self.feature_extractor(audio)
            student_backbone_out, student_hidden_states = self.backbone(student_features, **shared_kwargs)
            student_audio_hat, student_mag, student_phase = self.head(student_backbone_out, return_spec=True)

            # --- Distillation Losses ---
            # 1. Intermediate feature matching loss
            feature_distill_loss = torch.tensor(0.0, device=self.device)
            if self.feature_adapters:
                for i in range(len(self.feature_adapters)):
                    adapted_student_feature = self.feature_adapters[i](student_hidden_states[i])
                    feature_distill_loss += self.feature_loss(
                        adapted_student_feature, teacher_hidden_states[i].detach()
                    )
                feature_distill_loss /= len(self.feature_adapters)

            # 2. Output spectral matching loss (magnitude and phase)
            mag_loss = F.l1_loss(torch.log(student_mag + 1e-7), torch.log(teacher_mag.detach() + 1e-7))
            if self.phase_loss is not None:
                ip_loss, gd_loss, ptd_loss = self.phase_loss(student_phase, teacher_phase.detach())
                phase_loss = ip_loss + gd_loss + ptd_loss
            else:
                phase_loss = F.mse_loss(student_phase, teacher_phase.detach())
            output_distill_loss = mag_loss + self.hparams.phase_loss_coeff * phase_loss

            # --- GAN and Reconstruction Losses (from original VocosExp) ---
            if self.train_discriminator:
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(y=audio, y_hat=student_audio_hat)
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(y=audio, y_hat=student_audio_hat)

                loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
                loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)
            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0
            
            mel_loss = self.mel_loss(student_audio_hat.squeeze(1), audio.squeeze(1))
            
            # --- Total Generator Loss ---
            total_gen_loss = (
                loss_gen_mp
                + self.hparams.mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + self.hparams.mrd_loss_coeff * loss_fm_mrd
                + self.hparams.mel_loss_coeff * mel_loss
                + self.hparams.feature_distill_coeff * feature_distill_loss
                + self.hparams.output_distill_coeff * output_distill_loss
            )

            self.log_dict({
                "loss/gen/total": total_gen_loss,
                "loss/gen/multi_period_loss": loss_gen_mp,
                "loss/gen/multi_res_loss": loss_gen_mrd,
                "loss/gen/feature_matching_mp": loss_fm_mp,
                "loss/gen/feature_matching_mrd": loss_fm_mrd,
                "loss/gen/mel": mel_loss,
                "loss/gen/feature": feature_distill_loss,
                "loss/gen/output": output_distill_loss,
                "loss/gen/mag": mag_loss,
                "loss/gen/phase": phase_loss,
            }, prog_bar=True)
            return total_gen_loss

    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx)