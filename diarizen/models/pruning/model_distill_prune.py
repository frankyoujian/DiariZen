# Licensed under the MIT license.
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
# This work is inspired by: https://github.com/pyf98/DPHuBERT

import copy
import os

import torch
import torch.nn as nn

from diarizen.models.module.wav2vec2.model import wav2vec2_model 

def _print_on_main_process(message: str) -> None:
    if os.environ.get("RANK", "0") == "0":
        print(message)

class Model(nn.Module):
    def __init__(
        self,
        teacher_ckpt: str,
        student_ckpt: str,
        pruning_units: str = "conv,head,interm",
        distill_layers: str = "0,4,8,12",
    ):
        super().__init__()

        self.distill_layers = [int(l) for l in distill_layers.split(",")]
        
        self.teacher_model = self.build_teacher(teacher_ckpt)
        self.student_model, self.student_config = self.build_student(student_ckpt, pruning_units)
        
    def build_teacher(self, teacher_ckpt):
        teacher_checkpoint = torch.load(
            teacher_ckpt, map_location="cpu", weights_only=False
        )
        teacher_model = wav2vec2_model(**teacher_checkpoint["config"])
        teacher_model.load_state_dict(teacher_checkpoint["state_dict"], strict=True)
        _print_on_main_process("Loaded pretrained teacher checkpoint with strict key matching")

        # freeze teacher 
        for p in teacher_model.parameters():
            p.requires_grad = False
        _print_on_main_process("Froze all teacher parameters")
        teacher_model.eval()
        return teacher_model

    def build_student(self, student_ckpt, pruning_units):
        student_checkpoint = torch.load(
            student_ckpt, map_location="cpu", weights_only=False
        )
        pruning_units = {unit.strip() for unit in pruning_units.split(",") if unit.strip()}
        supported_units = {"conv", "head", "attlayer", "interm", "ffnlayer"}
        unknown_units = sorted(pruning_units - supported_units)
        if unknown_units:
            raise ValueError(f"Unsupported pruning units: {unknown_units}")
        _print_on_main_process(f"Pruning units: {sorted(pruning_units)}")
        student_config = copy.deepcopy(student_checkpoint['config'])
        student_config.update(
            dict(
                extractor_prune_conv_channels = "conv" in pruning_units,
                encoder_prune_attention_heads = "head" in pruning_units,
                encoder_prune_attention_layer = "attlayer" in pruning_units,
                encoder_prune_feed_forward_intermediate = "interm" in pruning_units,
                encoder_prune_feed_forward_layer = "ffnlayer" in pruning_units,
            )
        )
        student_model = wav2vec2_model(**student_config)
        pretrained_state = student_checkpoint["state_dict"]
        student_state = student_model.state_dict()
        pretrained_keys = set(pretrained_state)
        student_keys = set(student_state)
        missing_keys = sorted(student_keys - pretrained_keys)
        unexpected_keys = sorted(pretrained_keys - student_keys)
        pruning_gate_keys = {
            key
            for key in student_keys
            if ".hard_concrete" in key and key.endswith(".log_alpha")
        }
        checkpoint_gate_keys = pretrained_keys & pruning_gate_keys
        if checkpoint_gate_keys and checkpoint_gate_keys != pruning_gate_keys:
            raise RuntimeError(
                "Student checkpoint is incompatible: partial pruning-gate state "
                f"({len(checkpoint_gate_keys)} of "
                f"{len(pruning_gate_keys)} tensors present)"
            )
        missing_non_gate_keys = sorted(set(missing_keys) - pruning_gate_keys)
        if missing_non_gate_keys or unexpected_keys:
            raise RuntimeError(
                "Student checkpoint is incompatible: "
                f"missing non-gate keys={missing_non_gate_keys}, "
                f"unexpected keys={unexpected_keys}"
            )

        # Dense WavLM checkpoints do not contain the pruning gates. Keep those
        # constructor-initialized values, merge in every pretrained tensor, and
        # then use a strict load so no field remains unresolved.
        student_state.update(pretrained_state)
        student_model.load_state_dict(student_state, strict=True)
        _print_on_main_process(
            f"Loaded all {len(pretrained_state)} pretrained student tensors; "
            f"initialized {len(missing_keys)} pruning-gate tensors from model defaults"
        )
        return student_model, student_config
        
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_hiddens, _ = self.teacher_model.extract_features(waveforms)
            teacher_hiddens = torch.stack(
                [teacher_hiddens[idx] for idx in self.distill_layers], dim=1
            )  # (batch, layer, time, feature)
        
        student_hiddens, _ = self.student_model.extract_features(waveforms)
        student_hiddens = torch.stack(
            [student_hiddens[idx] for idx in self.distill_layers], dim=1
        )  # (batch, layer, time, feature)
        
        return student_hiddens, teacher_hiddens  