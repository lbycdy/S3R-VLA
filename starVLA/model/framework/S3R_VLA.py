# S3R-VLA: Self-Supervised Skill Learning for Robust Vision-Language-Action Models.
#
# All released versions are defined in this file; the config selects one with `framework.name`:
#   S3R_VLA_v1   skill code -> predictor and action expert (S = 4 separately quantized skill vectors, V = 32)
#   S3R_VLA_v2   v1 + conditional motion decoder + posterior world-model loss; the skill is a training-time regularizer
#                (not fed to the action expert); S = 2, V = 16
#   S3R_VLA_v3   method of the paper (posterior on observation changes + actions, DASQ, full attention among skill tokens)
#
#   S3RVLABase                    base model, backbone pass, V-JEPA 2 features, prior branch, flow-matching loss, inference
#     |-- S3R_VLA_v1 -- S3R_VLA_v2
#     |-- S3R_VLA_v3
#
# The base model (Qwen3-VL backbone, flow-matching action expert, frozen V-JEPA 2 encoder, latent world-model predictor,
# tokenizer expansion) is adapted from VLA-JEPA (https://github.com/ginwind/VLA-JEPA) and starVLA
# (https://github.com/starVLA/starVLA, MIT License, Copyright 2025 starVLA community); the flow-matching action head
# follows GR00T N1.5.
#
# Skill modules (attribute names follow the parameter layout of the released checkpoints):
#   skill_rvq             shared residual codebook, written only by the posterior branch
#   skill_posterior       posterior Q-Former (training only)
#   skill_head            prior Q-Former on the latent action / skill tokens (training and inference)
#   skill_rec_decoder     motion decoder (training only)
#   skill_action_decoder  action decoder (training only)
#   skill_dequant         v1, v2: skill projector, quantized skill -> backbone hidden size
#   skill_teacher         v1, v2: EMA copy of the posterior Q-Former, not used by any loss
# Future observations and actions are only used as training targets; the posterior branch is not run at inference.
import contextlib
import contextvars
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer
import transformers.models.qwen3_vl.modeling_qwen3_vl as _qwen3_vl_modeling
from transformers.masking_utils import or_masks

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.model.modules.skill_model import (
    ResidualSkillQuantizer, DriftAwareSkillQuantizer, PosteriorQFormer, PriorQFormer, SkillProjector,
    MotionDecoder, ConditionalMotionDecoder, TransformerActionDecoder, MLPActionDecoder, PosteriorEMA,
    cumulative_action_targets, coarse_to_fine_ce,
)

logger = initialize_overwatch(__name__)


# ---------------------------------------------------------------- full attention among skill tokens (v3)
# Qwen3-VL builds a causal mask in Qwen3VLTextModel.forward through `create_causal_mask`. While a [B, L] boolean tensor
# of skill-token positions is set in `_SKILL_POSITIONS`, the wrapper adds the block "skill token -> skill token" to the
# causal mask through `or_mask_function`; otherwise it is a pass-through.
_SKILL_POSITIONS = contextvars.ContextVar("s3r_skill_positions", default=None)


def _install_skill_attention():
    if getattr(_qwen3_vl_modeling, "_s3r_skill_attention_installed", False):
        return
    original = _qwen3_vl_modeling.create_causal_mask

    def create_causal_mask_with_skill_attention(*args, **kwargs):
        positions = _SKILL_POSITIONS.get()
        if positions is not None:
            def skill_block(batch_idx, head_idx, q_idx, kv_idx):
                return positions[batch_idx, q_idx] & positions[batch_idx, kv_idx]
            previous = kwargs.get("or_mask_function")
            kwargs["or_mask_function"] = skill_block if previous is None else or_masks(previous, skill_block)
        return original(*args, **kwargs)

    _qwen3_vl_modeling.create_causal_mask = create_causal_mask_with_skill_attention
    _qwen3_vl_modeling._s3r_skill_attention_installed = True


_install_skill_attention()


@contextlib.contextmanager
def skill_token_attention(positions: Optional[torch.Tensor]):
    """positions: [B, L] bool (True at skill tokens) or None (plain causal attention)."""
    token = _SKILL_POSITIONS.set(positions)
    try:
        yield
    finally:
        _SKILL_POSITIONS.reset(token)


def _print_rank0(msg: str):
    # the logger output is swallowed during training, so print to stdout directly (rank 0 only)
    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
        print(msg, flush=True)


def _normalized_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 normalized by the mean target magnitude (observation changes are much smaller than the features)."""
    scale = target.abs().mean().detach().clamp_min(1e-6)
    return F.l1_loss(pred.float() / scale, target.float() / scale)


# ==================================================================================================== shared base
class S3RVLABase(baseframework):
    """Base model and everything the versions share. A version sets the class attributes below and implements
    `_read_skill_config`, `_build_skill_modules`, `_skill_terms` (training), and `_inference_condition`."""

    # loss weights: term -> default; the config key of each weight is "<term>_weight" in framework.skill_model
    default_loss_weights: Dict[str, float] = {}
    full_skill_attention = False     # full attention among the latent action / skill tokens
    predictor_skill_tokens = 0       # skill tokens appended to the predictor condition of every time step

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        sk = config.framework.get("skill_model", {})
        vj = config.framework.vj2_model
        self.loss_weights = {k: float(sk.get(f"{k}_weight", d)) for k, d in self.default_loss_weights.items()}
        self.rvq_levels = int(sk.get("rvq_levels", 2))                                       # D
        self.level_weights = list(sk.get("ce_level_weights", [1.0] * self.rvq_levels))
        self.ce_warmup = int(sk.get("ce_warmup_forwards", 2000))
        self.log_every = int(sk.get("log_every_forwards", 100))
        self._read_skill_config(sk)

        # ---------------- base model ----------------
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        attn_impl = config.framework.qwenvl.get("attn_implementation", "flash_attention_2")
        if self.full_skill_attention and attn_impl not in ("sdpa", "eager"):
            raise ValueError(f"full attention among skill tokens needs attn_implementation sdpa or eager, got {attn_impl}")
        embodied_token = vj.get("embodied_action_token", "<|embodied_action|>")
        latent_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=vj.special_action_token,
            max_action_tokens=config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_token)
        config.framework.action_model.diffusion_model_cfg.cross_attention_dim = \
            self.qwen_vl_interface.model.config.hidden_size
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.future_action_window_size = config.framework.action_model.future_action_window_size

        self.vj_encoder = AutoModel.from_pretrained(vj.base_encoder)
        self.vj_encoder.requires_grad_(False)                                  # frozen V-JEPA 2 encoder
        self.vj_processor = AutoVideoProcessor.from_pretrained(vj.base_encoder)
        vj_hidden = self.vj_encoder.config.hidden_size                         # 1024 per view
        vlm_hidden = self.qwen_vl_interface.model.config.hidden_size           # 2048
        self.num_latent_frames = vj.num_frames // self.vj_encoder.config.tubelet_size    # K = 4 time steps
        self.tokens_per_step = vj.num_action_tokens_per_timestep                        # 8 latent action tokens

        # world-model predictor: time step k is conditioned on the k-th group of latent action / skill tokens
        # (+ `predictor_skill_tokens` skill vectors). num_add_tokens only changes the block-causal mask and the input
        # packing; the parameter shapes match the pretrained predictor.
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.num_latent_frames,
            img_size=(self.vj_encoder.config.image_size, self.vj_encoder.config.image_size),
            tubelet_size=1,
            depth=vj.depth,
            num_heads=vj.num_heads,
            embed_dim=vj_hidden * 2,                                           # two views
            action_embed_dim=vlm_hidden,
            num_add_tokens=self.tokens_per_step + self.predictor_skill_tokens,
        )
        # K-1 groups of 8 latent action / skill tokens between the instruction and the 32 embodied tokens
        self.replace_prompt = "".join(t * self.tokens_per_step for t in latent_tokens[:self.num_latent_frames - 1])
        self.embodied_replace_prompt = embodied_token * vj.num_embodied_action_tokens_per_instruction

        # ---------------- skill modules ----------------
        self._build_skill_modules(sk, vj_hidden, vlm_hidden)
        self.register_buffer("skill_step", torch.zeros((), dtype=torch.long))   # training forward passes
        self._first_forward_logged = False
        logger.info(f"[{type(self).__name__}] codebook D{self.skill_rvq.D} x V{self.skill_rvq.K}, code_dim "
                    f"{self.skill_rvq.dim}, loss weights {self.loss_weights}, ce_warmup {self.ce_warmup}, "
                    f"full_skill_attention {self.full_skill_attention}")

    # ---------------------------------------------------------------- version hooks
    def _read_skill_config(self, sk) -> None:
        raise NotImplementedError

    def _build_skill_modules(self, sk, vj_hidden: int, vlm_hidden: int) -> None:
        raise NotImplementedError

    def _skill_terms(self, z_la, z_act, video_emb, K, P, E, actions) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Training: returns the unweighted skill / world-model loss terms and the action-expert condition tokens."""
        raise NotImplementedError

    def _inference_condition(self, z_la, z_act) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference: returns the action-expert condition tokens and the prior skill codes."""
        raise NotImplementedError

    def _on_training_forward(self) -> None:
        pass

    # ---------------------------------------------------------------- base model
    def expand_tokenizer(self,
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>"):
        """Add the latent action (skill) tokens and the embodied action token to the tokenizer and resize the
        embeddings."""
        action_tokens, action_token_ids = [], []
        for i in range(max_action_tokens):
            token = special_action_token.format(i)
            action_tokens.append(token)
            if token not in tokenizer.get_vocab() and tokenizer.add_tokens([token], special_tokens=True) == 0:
                logger.warning(f"0 tokens added (they may already exist): {token}")
            action_token_ids.append(tokenizer.convert_tokens_to_ids(token))
        if embodied_action_token not in tokenizer.get_vocab() and \
                tokenizer.add_tokens([embodied_action_token], special_tokens=True) == 0:
            logger.warning(f"0 tokens added (they may already exist): {embodied_action_token}")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    def _build_inputs(self, images, instructions, prompt_template=None):
        kwargs = {} if prompt_template is None else {"prompt_template": prompt_template}
        return self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions,
            prompt_replace_dict={"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt},
            **kwargs)

    def _run_backbone(self, qwen_inputs):
        """Backbone pass. Returns the last hidden states [B, L, 2048], the outputs of the latent action / skill tokens
        z_la [B, (K-1)*8, 2048], and the outputs of the embodied tokens z_act [B, 32, 2048]."""
        input_ids = qwen_inputs["input_ids"]
        latent_pos = torch.isin(input_ids, torch.tensor(self.action_token_ids, device=input_ids.device))
        embodied_pos = input_ids == self.embodied_action_token_id
        with skill_token_attention(latent_pos if self.full_skill_attention else None):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.qwen_vl_interface(
                    **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        B, _, H = last_hidden.shape
        li, ei = latent_pos.nonzero(as_tuple=True), embodied_pos.nonzero(as_tuple=True)
        z_la = last_hidden[li[0], li[1], :].view(B, -1, H)
        z_act = last_hidden[ei[0], ei[1], :].view(B, -1, H)
        return last_hidden, z_la, z_act

    def _encode_videos(self, videos: List[np.ndarray]):
        """videos: B arrays [V, T, H, W, 3] -> frozen V-JEPA 2 features [B, K*P, V*E] (time-step-major, the views
        concatenated along the feature dimension), K time steps, P patches per step, E feature size per view."""
        videos = np.stack(videos).transpose(0, 1, 2, 5, 3, 4)                # [B, V, T, 3, H, W]
        B, V, T, C, Hh, Ww = videos.shape
        videos = videos.reshape(B * V, T, C, Hh, Ww)
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            pixels = torch.cat([self.vj_processor(videos=videos[i], return_tensors="pt")["pixel_values_videos"]
                                .to(self.vj_encoder.device) for i in range(B * V)], dim=0)
            emb = self.vj_encoder.get_vision_features(pixel_values_videos=pixels)
            K = T // self.vj_encoder.config.tubelet_size
            E = emb.shape[-1]
            P = emb.shape[1] // K
            emb = emb.reshape(B, V, K, P, E).permute(0, 2, 3, 1, 4).contiguous().reshape(B, K * P, V * E)
        return emb, K, P, E

    # ---------------------------------------------------------------- shared skill / world-model pieces
    def _codebook_kwargs(self, sk, default_size: int) -> dict:
        return dict(
            num_levels=self.rvq_levels,                                         # D
            codebook_size=int(sk.get("codebook_size", default_size)),           # V
            code_dim=self.code_dim,
            ema_decay=float(sk.get("codebook_ema_decay", 0.99)),                # gamma
            commit_weight=1.0,
            revive_every=int(sk.get("revive_every", 100)),                      # T_r (0 disables dead-code restart)
            revive_threshold=float(sk.get("revive_threshold", 1.0)),            # N_min
            revive_decay=float(sk.get("revive_decay", 0.99)),
            revive_noise=float(sk.get("revive_noise", 0.01)),                   # sigma_r
            tau_ema=float(sk.get("tau_ema", 0.99)),                             # EMA coefficient of the temperature
            tau_target_gap=float(sk.get("tau_target_gap", 1.0)),
        )

    def _prior_branch(self, z_la, post_out):
        """Prior Q-Former on the latent action / skill tokens, read-only codebook, level-wise cross-entropy against the
        posterior codes (stop-gradient target, teacher-forced residuals); zero for the first `ce_warmup_forwards`
        training forward passes."""
        with torch.no_grad():
            q_star = post_out["codes"].detach()
        pri_out = self.skill_rvq(self.skill_head(z_la), update_codebook=False, ce_labels=q_star)
        loss_ce, ce_acc = coarse_to_fine_ce(pri_out["logits"], q_star, self.level_weights)
        if self.skill_step.item() < self.ce_warmup:
            loss_ce = loss_ce * 0.0
        return pri_out, loss_ce, ce_acc

    def _wm_loss(self, video_emb, K, P, cond):
        """Latent world model: features of time steps 0..K-2 + condition -> features of time steps 1..K-1 (L1)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            input_states = video_emb[:, :P * (K - 1), :]
            gt_states = video_emb[:, P:, :]
            predicted = self.vj_predictor(input_states, cond.to(input_states.dtype))
            return F.l1_loss(predicted, gt_states, reduction="mean")

    def _action_loss(self, cond_tokens, actions: np.ndarray, state):
        """Flow-matching loss of the action expert, with `trainer.repeated_diffusion_steps` (tau, eps) pairs per sample."""
        with torch.autocast("cuda", dtype=torch.float32):
            dev, dt = cond_tokens.device, cond_tokens.dtype
            target = torch.tensor(actions, device=dev, dtype=dt)[:, -(self.future_action_window_size + 1):, :]
            rds = self.config.trainer.get("repeated_diffusion_steps", 4)
            if state is not None:
                state = torch.tensor(np.array(state), device=dev, dtype=dt).repeat(rds, 1, 1)
            return self.action_model(cond_tokens.repeat(rds, 1, 1), target.repeat(rds, 1, 1), state)

    def _end_training_forward(self, post_out, pri_out, ce_acc, terms):
        """Counts training forward passes and prints a [skill] line every `log_every_forwards` passes (rank 0).
        The trainer never calls model.train()/eval(), so torch.is_grad_enabled() tells training from inference."""
        if not self._first_forward_logged:
            self._first_forward_logged = True
            _print_rank0(f"[skill] first forward: {type(self).__name__} grad_enabled={torch.is_grad_enabled()} "
                         f"loss terms {list(terms)} full_skill_attention={self.full_skill_attention}")
        if not torch.is_grad_enabled():
            return
        with torch.no_grad():
            self._on_training_forward()
            self.skill_step += 1
            step = int(self.skill_step.item())
            if step % self.log_every:
                return
            cb = self.skill_rvq
            line = (f"[skill] step={step} ce_acc={ce_acc.item():.3f} "
                    f"ent_post={['%.2f' % e for e in cb.batch_code_entropy(post_out['codes'])]} "
                    f"ent_prior={['%.2f' % e for e in cb.batch_code_entropy(pri_out['codes'])]} "
                    f"active={cb.active_codes()} kappa={cb.tau:.4f} div={post_out['diversity'].item():.3f} "
                    f"restarted={int(cb.revive_total.item())} quant_err={cb.quant_err.item():.4f}")
            if isinstance(cb, DriftAwareSkillQuantizer):
                line += f" delta_bar={float(cb.delta_bar):.5f} kernel_mass={float(cb.kernel_mass):.3f}"
            line += "".join(f" {k}={terms[k].item():.4f}" for k in ("wm", "wm_post", "motion", "action_rec") if k in terms)
            _print_rank0(line)

    # ---------------------------------------------------------------- training / inference
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        if "action" not in examples[0]:
            raise NotImplementedError("S3R-VLA is trained on robot data; every example needs an action chunk")
        actions = np.array([ex["action"] for ex in examples])                  # [B, H, 7]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        qwen_inputs = self._build_inputs([ex["image"] for ex in examples], [ex["lang"] for ex in examples],
                                         prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""))
        _, z_la, z_act = self._run_backbone(qwen_inputs)
        video_emb, K, P, E = self._encode_videos([ex["video"] for ex in examples])
        terms, cond_tokens = self._skill_terms(z_la, z_act, video_emb, K, P, E, actions)
        assert set(terms) <= set(self.loss_weights), f"loss terms without a weight: {set(terms) - set(self.loss_weights)}"
        out = {"action_loss": self._action_loss(cond_tokens, actions, state)}
        out.update({f"{k}_loss": w * terms[k] for k, w in self.loss_weights.items() if k in terms})
        return out

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> dict:
        """One backbone pass, prior branch, and flow matching (4 Euler steps). The posterior branch and the world model
        are not run; the prior skill codes are returned for analysis."""
        image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if image_size:
            batch_images = resize_images(batch_images, target_size=image_size)
        _, z_la, z_act = self._run_backbone(self._build_inputs(batch_images, instructions))
        cond_tokens, skill_codes = self._inference_condition(z_la, z_act)
        if state is not None:
            state = torch.from_numpy(np.array(state)).to(z_act.device, dtype=z_act.dtype)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(cond_tokens, state)
        return {
            "normalized_actions": pred_actions.detach().cpu().numpy(),
            "embodied_action_tokens": z_act.to(dtype=torch.float32).detach().cpu().numpy(),
            "skill_codes": skill_codes.detach().cpu().numpy(),
        }


# ==================================================================================================== v1 / v2
@FRAMEWORK_REGISTRY.register("S3R_VLA_v1")
class S3R_VLA_v1(S3RVLABase):
    """Posterior branch: pooled observation changes of the third-person view -> S separately quantized skill vectors.
    Prior branch: the 24 latent action tokens of the base model (causal attention) -> prior Q-Former -> codebook ->
    skill projector -> s_hat [B, S, 2048], which is appended to the predictor condition of every time step
    ([8 latent action tokens | S skill tokens]) and to the action-expert condition ([32 embodied tokens | S skill tokens]).
    Motion decoder: skill -> pooled future observation changes; action decoder: skill -> action chunk (Transformer).
    Loss: L_fm + 0.1 L_wm + 0.05 L_ce + 0.5 L_motion + 0.5 L_act + 1.0 L_commit + 0.1 L_div."""

    default_loss_weights = dict(wm=0.1, ce=0.05, action_rec=0.5, motion=0.5, commit=1.0, diversity=0.1, wm_post=0.0)
    conditional_motion_decoder = False   # motion decoder conditioned on the pooled current time step
    skill_to_action_expert = True        # s_hat is appended to the action-expert condition

    def _read_skill_config(self, sk):
        self.S = int(sk.get("num_skill_tokens", 4))
        self.code_dim = int(sk.get("code_dim", 32))
        self.predictor_skill_tokens = self.S
        self.action_rec_steps = int(sk.get("action_rec_steps", 8))

    def _build_skill_modules(self, sk, vj_hidden, vlm_hidden):
        K, hidden = self.num_latent_frames, int(sk.get("qformer_hidden", 256))
        self.skill_rvq = ResidualSkillQuantizer(**self._codebook_kwargs(sk, default_size=32))
        self.skill_posterior = PosteriorQFormer(
            feat_dim=vj_hidden, hidden=hidden, num_frames=K, num_skill_tokens=self.S, code_dim=self.code_dim)
        self.skill_teacher = PosteriorEMA(self.skill_posterior, decay=float(sk.get("ema_teacher_decay", 0.999)))
        self.skill_head = PriorQFormer(vlm_dim=vlm_hidden, hidden=hidden, num_skill_tokens=self.S, code_dim=self.code_dim)
        self.skill_dequant = SkillProjector(code_dim=self.code_dim, out_dim=vlm_hidden)
        motion_decoder = ConditionalMotionDecoder if self.conditional_motion_decoder else MotionDecoder
        self.skill_rec_decoder = motion_decoder(
            code_dim=self.code_dim, num_skill_tokens=self.S, num_future=K - 1, feat_dim=vj_hidden,
            hidden=int(sk.get("decoder_hidden", 512)))
        self.skill_action_decoder = TransformerActionDecoder(
            code_dim=self.code_dim, hidden=hidden, n_steps=self.action_rec_steps,
            action_dim=int(sk.get("action_rec_action_dim", 7)), num_layers=int(sk.get("action_rec_layers", 2)))

    def _on_training_forward(self):
        self.skill_teacher.update(self.skill_posterior)

    def _predictor_cond(self, z_la, s_hat, n_steps):
        """Per time step k: [z_la[:, 8k:8k+8] | s_hat] -> [B, n_steps * (8 + S), 2048] (time-step-major, as the
        predictor splits its condition with a.view(B, T, -1, D))."""
        pf = self.tokens_per_step
        s = s_hat.to(z_la.dtype)
        return torch.cat([torch.cat([z_la[:, k * pf:(k + 1) * pf, :], s], dim=1) for k in range(n_steps)], dim=1)

    def _skill_terms(self, z_la, z_act, video_emb, K, P, E, actions):
        # ---------------- posterior branch (training only): third-person view ----------------
        frames = video_emb[..., :E].detach().reshape(-1, K, P, E)           # [B, K, P, 1024]
        post_z = self.skill_posterior((frames - frames[:, :1]).mean(dim=2))  # pooled observation changes -> [B, S, 32]
        post_out = self.skill_rvq(post_z, update_codebook=True)             # the codebook is written only here
        target = (frames[:, 1:] - frames[:, :1]).mean(dim=2)                 # pooled future observation changes
        cond = (frames[:, 0].mean(dim=1).detach(),) if self.conditional_motion_decoder else ()
        terms = {"motion": _normalized_l1(self.skill_rec_decoder(post_out["quantized"], *cond), target)}
        terms["action_rec"] = post_z.new_zeros(())                           # action decoder (no observation input)
        if self.loss_weights["action_rec"] > 0:
            a = torch.tensor(actions[:, :min(self.action_rec_steps, actions.shape[1]), :],
                             device=post_z.device, dtype=torch.float32)
            terms["action_rec"] = F.l1_loss(self.skill_action_decoder(post_out["quantized"])[:, :a.shape[1], :].float(), a)

        # ---------------- prior branch (same path at training and inference) ----------------
        pri_out, terms["ce"], ce_acc = self._prior_branch(z_la, post_out)
        s_hat = self.skill_dequant(pri_out["quantized"])                     # [B, S, 2048] (straight-through)
        terms["commit"] = post_out["commit"] + pri_out["commit"]
        terms["diversity"] = post_out["diversity"]                           # posterior branch only

        # ---------------- latent world model ----------------
        terms["wm"] = self._wm_loss(video_emb, K, P, self._predictor_cond(z_la, s_hat, K - 1))
        if self.loss_weights["wm_post"] > 0:
            # posterior skill -> same projector -> same predictor (latent action tokens detached); training only
            s_post = self.skill_dequant(post_out["quantized"])
            terms["wm_post"] = self._wm_loss(video_emb, K, P, self._predictor_cond(z_la.detach(), s_post, K - 1))

        self._end_training_forward(post_out, pri_out, ce_acc, terms)
        cond_tokens = torch.cat([z_act, s_hat.to(z_act.dtype)], dim=1) if self.skill_to_action_expert else z_act
        return terms, cond_tokens

    def _inference_condition(self, z_la, z_act):
        quantized, codes = self.skill_rvq.infer_quantized(self.skill_head(z_la), restrict_seen=True)
        if not self.skill_to_action_expert:
            return z_act, codes
        return torch.cat([z_act, self.skill_dequant(quantized).to(z_act.dtype)], dim=1), codes


@FRAMEWORK_REGISTRY.register("S3R_VLA_v2")
class S3R_VLA_v2(S3R_VLA_v1):
    """v1 with (S = 2, V = 16 in the config):
      - a motion decoder conditioned on the pooled current time step, so the skill only needs to carry what changes;
      - a posterior world-model loss: the posterior skill, projected by the same skill projector, conditions the same
        predictor (latent action tokens detached) to predict the future latents (training only);
      - the skill as a training-time regularizer: the action expert is conditioned only on the 32 embodied tokens; the
        prior skill code is computed at inference and returned for analysis.
    Loss: L_fm + 0.1 L_wm + 0.05 L_ce + 0.5 L_motion + 0.5 L_act + 1.0 L_commit + 0.1 L_div + 1.0 L_wm_post."""

    default_loss_weights = dict(S3R_VLA_v1.default_loss_weights, wm_post=1.0)
    conditional_motion_decoder = True
    skill_to_action_expert = False


# ==================================================================================================== v3
@FRAMEWORK_REGISTRY.register("S3R_VLA_v3")
class S3R_VLA_v3(S3RVLABase):
    """Method of the paper.
    Posterior branch (training only): observation changes dh_k = h_bar_k - h_bar_0 of the pooled V-JEPA 2 features of
      both views and action tokens alpha_k = W_alpha [A_2k; A_2k+1] built from the cumulative actions -> posterior
      Q-Former -> u_post [B, 1, S * 16] -> shared residual codebook (DASQ) -> skill code q* and quantized vector e*.
      e* is supervised by the action decoder (action chunk with relative gripper commands) and the motion decoder
      ([e*, h_bar_0] -> dh_1..dh_{K-1}).
    Prior branch (training = inference): 24 skill tokens at the positions of the VLA-JEPA latent action tokens
      (initialized from them: each group of 8 repeats one token embedding and is distinguished by position), with full
      attention among them and causal attention elsewhere. Their outputs condition the predictor as in the base model
      and feed the prior Q-Former, matched to q* by the level-wise cross-entropy loss. The losses of the prior branch
      (cross-entropy and prior commitment) are added after the first `ce_warmup_forwards` iterations.
    The skill code is fed neither to the predictor nor to the action expert, which is conditioned on the 32 embodied
      tokens and the proprioception; skills act on it only through attention to the skill tokens.
    Loss: L_FM + 0.3 L_wm + 0.5 L_CE + 0.5 L_act + 0.5 L_motion + 1.0 (L_commit^post + L_commit^pri) + 0.2 L_div."""

    default_loss_weights = dict(wm=0.3, ce=0.5, action_rec=0.5, motion=0.5, commit=1.0, diversity=0.2)

    def _read_skill_config(self, sk):
        self.num_queries = int(sk.get("num_queries", 2))                     # S
        self.query_dim = int(sk.get("query_dim", 16))
        self.code_dim = self.num_queries * self.query_dim                     # 32
        self.full_skill_attention = bool(sk.get("skill_full_attention", True))
        self.posterior_inputs = list(sk.get("posterior_inputs", ["obs", "action"]))
        assert set(self.posterior_inputs) <= {"obs", "action"} and self.posterior_inputs, self.posterior_inputs
        self.frame_difference = bool(sk.get("frame_difference", True))

    def _build_skill_modules(self, sk, vj_hidden, vlm_hidden):
        K = self.num_latent_frames
        action_dim = int(self.config.framework.action_model.action_dim)
        self.action_horizon = 2 * K - 1                                       # H = 2K - 1 actions per 2K frames
        hidden, heads = int(sk.get("qformer_hidden", 256)), int(sk.get("qformer_heads", 8))
        decoder_hidden = int(sk.get("decoder_hidden", 512))
        self.skill_rvq = DriftAwareSkillQuantizer(
            **self._codebook_kwargs(sk, default_size=16),
            kernel_weight=float(sk.get("kernel_weight", 0.1)),                # mu (0 disables the kernel)
            kernel_scale=float(sk.get("kernel_scale", 4.0)),                  # zeta
            kernel_ema=float(sk.get("kernel_ema", 0.99)))                     # EMA coefficient of delta_bar
        self.skill_posterior = PosteriorQFormer(
            feat_dim=2 * vj_hidden, hidden=hidden, num_frames=K, num_skill_tokens=self.num_queries,
            code_dim=self.query_dim, num_layers=int(sk.get("posterior_layers", 2)), num_heads=heads,
            use_obs="obs" in self.posterior_inputs,
            action_dim=action_dim if "action" in self.posterior_inputs else 0, concat_queries=True)
        self.skill_head = PriorQFormer(vlm_dim=vlm_hidden, hidden=hidden, num_skill_tokens=self.num_queries,
                                       code_dim=self.query_dim, num_heads=heads, concat_queries=True)
        self.skill_rec_decoder = MotionDecoder(code_dim=self.code_dim, num_skill_tokens=1, num_future=K - 1,
                                               feat_dim=2 * vj_hidden, hidden=decoder_hidden, cond_dim=2 * vj_hidden)
        self.skill_action_decoder = MLPActionDecoder(code_dim=self.code_dim, horizon=self.action_horizon,
                                                     action_dim=action_dim, hidden=decoder_hidden)

    def _skill_terms(self, z_la, z_act, video_emb, K, P, E, actions):
        # ---------------- posterior branch (training only) ----------------
        B = video_emb.shape[0]
        h_bar = video_emb.detach().reshape(B, K, P, -1).float().mean(dim=2)    # pooled features [B, K, 2048]
        # the chunk as seen by the policy: pose increments normalized to [-1, 1], gripper command in {0, 1}
        chunk = torch.tensor(actions[:, :self.action_horizon, :], device=h_bar.device, dtype=torch.float32)
        cum_actions, a_tilde = cumulative_action_targets(chunk)               # [B, 2K, 7], [B, H, 7]
        obs = (h_bar - h_bar[:, :1] if self.frame_difference else h_bar) if "obs" in self.posterior_inputs else None
        u_post = self.skill_posterior(obs, cum_actions if "action" in self.posterior_inputs else None)   # [B, 1, 32]
        post_out = self.skill_rvq(u_post, update_codebook=True)              # q*, e*; the codebook is written here
        e_star = post_out["quantized"]
        terms = {"motion": _normalized_l1(self.skill_rec_decoder(e_star, h_bar[:, 0]), h_bar[:, 1:] - h_bar[:, :1]),
                 "action_rec": F.l1_loss(self.skill_action_decoder(e_star).float(), a_tilde)}

        # ---------------- prior branch (same path at training and inference) ----------------
        pri_out, terms["ce"], ce_acc = self._prior_branch(z_la, post_out)
        # the prior branch is optimized jointly only after the posterior codes have stabilized: like L_CE, its
        # commitment loss is zero for the first `ce_warmup_forwards` iterations
        pri_commit = pri_out["commit"] * (0.0 if self.skill_step.item() < self.ce_warmup else 1.0)
        terms["commit"] = post_out["commit"] + pri_commit
        terms["diversity"] = post_out["diversity"]

        # ---------------- latent world model conditioned on the skill tokens ----------------
        terms["wm"] = (self._wm_loss(video_emb, K, P, z_la) if self.loss_weights["wm"] > 0
                       else video_emb.new_zeros((), dtype=torch.float32))

        self._end_training_forward(post_out, pri_out, ce_acc, terms)
        return terms, z_act

    def _inference_condition(self, z_la, z_act):
        _, codes = self.skill_rvq.infer_quantized(self.skill_head(z_la), restrict_seen=False)
        return z_act, codes
