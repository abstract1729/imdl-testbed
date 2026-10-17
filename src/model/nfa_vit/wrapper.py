import torch
import torch.nn as nn
import sys
from pathlib import Path
import argparse


CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[3]
NFA_VIT_PATH = PROJECT_ROOT / "models" / "nfa-vit"

if str(NFA_VIT_PATH) not in sys.path:
    sys.path.insert(0, str(NFA_VIT_PATH))


class NFAViTInference(nn.Module):
    def __init__(
        self,
        model_root,
        checkpoint_path,
        np_weights,
        seg_b0_weights,
        seg_b2_weights,
        device="cpu"
    ):
        super().__init__()

        self.device = torch.device(device)

        model_root = Path(model_root)
        if str(model_root) not in sys.path:
            sys.path.append(str(model_root))

        from model_zoo.nfa_vit.nfa_vit import NFA_ViT_modify1

        # ---- REQUIRED: pass pretrained weights ----
        self.model = NFA_ViT_modify1(
            np_pretrain_weights=np_weights,
            seg_b0_pretrain_weights=seg_b0_weights,
            seg_b2_pretrain_weights=seg_b2_weights,
        )

        # ---- Load checkpoint ----
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])

        ckpt = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        self.model.load_state_dict(ckpt["model"])

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, image_tensor):
        """
        Args:
            image_tensor: (1, 3, H, W)

        Returns:
            dict:
                {
                    "pred_mask": (H, W),
                    "pred_score": float
                }
        """

        image_tensor = image_tensor.to(self.device)

        B, _, H, W = image_tensor.shape

        # ---- Dummy inputs (still required internally) ----
        dummy_mask = torch.zeros((B, 1, H, W), device=self.device)
        dummy_label = torch.zeros((B,), device=self.device)

        # ---- Forward ----
        output = self.model(image_tensor, dummy_mask, dummy_label)

        # ---- Extract outputs ----
        pred_mask = output["pred_mask"]      # (B,1,H,W)
        pred_label = output["pred_label"]    # (B,)

        pred_mask = pred_mask.squeeze().detach().cpu().numpy()
        pred_score = pred_label.item()

        return {
            "pred_mask": pred_mask,
            "pred_score": pred_score,
        }