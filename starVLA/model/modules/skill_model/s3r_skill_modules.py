# S3R-VLA skill modules (shared by all versions).
#
#   ResidualSkillQuantizer     shared residual codebook: straight-through estimator, EMA update driven by the posterior
#                              branch, dead-code restart, diversity regularization, adaptive temperature (v1, v2)
#   DriftAwareSkillQuantizer   + kernel-weighted EMA = drift-aware skill quantization, DASQ (v3)
#   PosteriorQFormer           posterior branch: observation-change tokens (+ action tokens) -> skill vector(s)
#   PriorQFormer               prior branch: latent action / skill token outputs -> skill vector(s)
#   SkillProjector             quantized skill vectors -> backbone hidden size (v1, v2)
#   MotionDecoder              skill (+ optional condition) -> future observation changes, MLP (v1; v3 with condition)
#   ConditionalMotionDecoder   skill + current time step -> future observation changes, two input projections (v2)
#   TransformerActionDecoder   skill -> action chunk with per-step queries (v1, v2)
#   MLPActionDecoder           skill -> action chunk, MLP (v3)
#   PosteriorEMA               EMA copy of the posterior Q-Former, not used by any loss (v1, v2 checkpoint layout)
#   cumulative_action_targets  cumulative actions A_i and the action-decoder target a_tilde (v3)
#   coarse_to_fine_ce          level-wise cross-entropy of the prior logits against the posterior codes
#
# With `concat_queries=True` the Q-Formers concatenate their S query outputs into one vector, so each window has one skill
# code (v3); otherwise every query output is a separately quantized skill vector (v1, v2).
# Future observations and actions are only used as training targets; the posterior branch is discarded at inference.
import copy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _all_reduce_(t: torch.Tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """MLP with three linear layers."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, out_dim))


# ======================================================================================================== codebooks
class ResidualSkillQuantizer(nn.Module):
    """D-level residual quantizer whose codebook is shared by the posterior and prior branches.

    * Straight-through estimator. The codebook is updated by EMA, driven only by the posterior forward pass
      (update_codebook=True); the prior branch only reads it.
    * Dead-code restart: codes whose recent hard-assignment count falls below `revive_threshold` are reset, every
      `revive_every` codebook updates, to the posterior residuals with the largest quantization errors in the batch,
      and their EMA statistics are reset accordingly.
    * Diversity regularization: div = (log V - H(p_bar)) / log V in [0, 1], where p_bar is the batch (cross-rank)
      code-usage histogram, with a straight-through estimate of the soft assignments.
    * Adaptive temperature: logits = -d^2 / tau, where tau tracks the moving average of the gap between the
      second-nearest and nearest squared distances; it is updated only in the posterior pass.
    The codebook and its statistics are kept in float32 buffers.

    Note: the EMA update and the dead-code restart write the codebook buffers in place. The trainer runs the forward pass
    under torch.autocast("cuda", dtype=torch.bfloat16), so the distance matmul saves a bf16 copy for backward.
    Backpropagating through an update_codebook=True output in pure fp32 (without autocast) triggers PyTorch's in-place
    version check.
    """

    def __init__(self, num_levels: int = 2, codebook_size: int = 32, code_dim: int = 32,
                 ema_decay: float = 0.99, commit_weight: float = 1.0, eps: float = 1e-5,
                 revive_every: int = 100, revive_threshold: float = 1.0,
                 revive_decay: float = 0.99, revive_noise: float = 0.01,
                 tau_ema: float = 0.99, tau_target_gap: float = 1.0, tau_min: float = 1e-6,
                 diversity_eps: float = 1e-8):
        super().__init__()
        self.D, self.K, self.dim = num_levels, codebook_size, code_dim
        self.ema_decay, self.commit_weight, self.eps = ema_decay, commit_weight, eps
        cb = torch.randn(num_levels, codebook_size, code_dim) * 0.1
        self.register_buffer("codebook", cb)                                    # [D,K,dim]
        self.register_buffer("ema_cluster", torch.zeros(num_levels, codebook_size))
        self.register_buffer("ema_embed_sum", cb.clone())
        self.register_buffer("usage_count", torch.zeros(num_levels, codebook_size))  # cumulative posterior usage
        self.register_buffer("quant_err", torch.zeros(()))                      # moving average, for logging
        self._dist_synced = False                                               # broadcast buffers from rank 0 once

        # dead-code restart
        self.revive_every = int(revive_every)
        self.revive_threshold = float(revive_threshold)
        self.revive_decay = float(revive_decay)
        self.revive_noise = float(revive_noise)
        self.register_buffer("recent_count", torch.zeros(self.D, self.K))     # decayed hard-assignment count
        self.register_buffer("update_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("revive_total", torch.zeros(()))                  # total restarted codes, for logging

        # adaptive temperature (tau = 1.0 until the first codebook update)
        self.tau = 1.0
        self.tau_ema, self.tau_target_gap, self.tau_min = float(tau_ema), float(tau_target_gap), float(tau_min)
        self.register_buffer("tau_run", torch.zeros(()))                       # moving d^2 gap; 0 = not initialized

        self.diversity_eps = float(diversity_eps)

    # ------------------------------------------------------------------ helpers
    def _sync_once(self):
        # DeepSpeed/DDP broadcast parameters at initialization but not necessarily buffers; codebooks that are randomly
        # initialized differently on each rank would make the EMA diverge, so broadcast once from rank 0.
        if self._dist_synced:
            return
        self._dist_synced = True
        if dist.is_available() and dist.is_initialized():
            for name in ("codebook", "ema_cluster", "ema_embed_sum", "usage_count"):
                dist.broadcast(getattr(self, name).data, src=0)

    def _fp32_buffers(self):
        if self.codebook.dtype != torch.float32:
            for name in ("codebook", "ema_cluster", "ema_embed_sum", "usage_count", "quant_err"):
                buf = getattr(self, name)
                setattr(self, name, buf.float())
        if self.recent_count.dtype != torch.float32:
            self.recent_count = self.recent_count.float()

    @staticmethod
    def _dist2(x, cb):
        # x: [N,dim], cb: [K,dim] -> [N,K]
        return (x.pow(2).sum(-1, keepdim=True) - 2 * x @ cb.t() + cb.pow(2).sum(-1)[None, :])

    def _ema_update(self, level: int, x: torch.Tensor, onehot: torch.Tensor):
        # x: [N,dim] fp32, onehot: [N,K] fp32; summed over ranks under DDP
        cluster = _all_reduce_(onehot.sum(0))                       # [K]
        embed_sum = _all_reduce_(onehot.t() @ x)                    # [K,dim]
        d = self.ema_decay
        self.ema_cluster[level].mul_(d).add_(cluster, alpha=1 - d)
        self.ema_embed_sum[level].mul_(d).add_(embed_sum, alpha=1 - d)
        n = self.ema_cluster[level].sum()
        w = (self.ema_cluster[level] + self.eps) / (n + self.K * self.eps) * n  # Laplace smoothing
        self.codebook[level] = self.ema_embed_sum[level] / w.unsqueeze(-1)

    @torch.no_grad()
    def _update_tau(self, dist2: torch.Tensor):
        """dist2: [N,K]. Calibrate the temperature with the second-nearest minus nearest d^2 gap."""
        if dist2.shape[-1] < 2:
            return
        two = dist2.topk(2, dim=-1, largest=False).values
        gap = (two[:, 1] - two[:, 0]).mean()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(gap); gap = gap / dist.get_world_size()
        if float(self.tau_run.item()) <= 0:
            self.tau_run.fill_(float(gap))
        else:
            self.tau_run.mul_(self.tau_ema).add_(gap, alpha=1 - self.tau_ema)
        self.tau = max(float(self.tau_run.item()) / self.tau_target_gap, self.tau_min)

    @torch.no_grad()
    def _revive_dead_codes(self, level: int, residual: torch.Tensor, dist2: torch.Tensor):
        """Reset recently unused codes to the residuals with the largest quantization errors in the batch.
        residual: [N,dim] (fp32, detached), dist2: [N,K]"""
        dead = (self.recent_count[level] < self.revive_threshold)
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        err = dist2.min(dim=-1).values                                  # [N] d^2 to the nearest code
        n_take = min(n_dead, residual.shape[0])
        cand = residual[err.topk(n_take).indices]                       # [n_take,dim]
        if n_take < n_dead:                                             # resample if the batch is too small
            rep = cand[torch.randint(0, n_take, (n_dead - n_take,), device=cand.device)]
            cand = torch.cat([cand, rep], dim=0)
        cand = cand + self.revive_noise * torch.randn_like(cand)
        idx = dead.nonzero(as_tuple=True)[0]
        self.codebook[level][idx] = cand
        # reset the EMA statistics as if the code had just been used once; otherwise the next EMA step pulls it back
        self.ema_cluster[level][idx] = 1.0
        self.ema_embed_sum[level][idx] = cand
        self.recent_count[level][idx] = self.revive_threshold           # grace period
        self.revive_total += n_dead
        return n_dead

    @torch.no_grad()
    def _broadcast_codebook(self):
        if dist.is_available() and dist.is_initialized():
            for name in ("codebook", "ema_cluster", "ema_embed_sum", "recent_count"):
                dist.broadcast(getattr(self, name).data, src=0)

    def _diversity_from_logits(self, logits_list):
        """logits_list: D tensors [B,S,K] (temperature included). Returns (div_loss, mean_entropy_bits).
        Straight-through estimate: the forward value uses hard assignments (the actual code-usage histogram), the
        gradient flows through the soft assignments."""
        logK = math.log(self.K)
        divs, ents = [], []
        for lg in logits_list:
            p = F.softmax(lg.reshape(-1, self.K).float(), dim=-1)     # [N,K] soft assignment (gradient)
            hard = F.one_hot(p.argmax(-1), self.K).float()            # [N,K] hard assignment (value)
            p_st = hard + p - p.detach()
            pbar = p_st.mean(0)                                       # [K] batch code-usage histogram
            if dist.is_available() and dist.is_initialized():
                pbar = pbar / dist.get_world_size()
                dist.all_reduce(pbar)                                 # global average over ranks
            H = -(pbar * (pbar + self.diversity_eps).log()).sum()
            divs.append((logK - H) / logK)                            # in [0,1]; 1 = all samples share one code
            ents.append(H / math.log(2.0))                            # bits
        return torch.stack(divs).mean(), torch.stack(ents).mean().detach()

    @torch.no_grad()
    def batch_code_entropy(self, codes: torch.Tensor) -> List[float]:
        """codes: [B,S,D] -> per-level entropy (bits) of the code distribution, for collapse monitoring."""
        out = []
        for d_lv in range(codes.shape[-1]):
            idx = codes[..., d_lv].reshape(-1)
            p = torch.bincount(idx, minlength=self.K).float()
            p = p / p.sum().clamp_min(1)
            nz = p[p > 0]
            out.append(float(-(nz * nz.log2()).sum().item()))
        return out

    @torch.no_grad()
    def active_codes(self) -> str:
        """Codes per level whose recent hard-assignment count is at least the restart threshold, e.g. '16/16,15/16'."""
        return ",".join(f"{int((self.recent_count[lv] >= self.revive_threshold).sum().item())}/{self.K}"
                        for lv in range(self.D))

    # ------------------------------------------------------------------ forward
    def forward(self, z: torch.Tensor, update_codebook: bool = False,
                ce_labels: Optional[torch.Tensor] = None):
        """z: [B,S,dim]. Returns a dict with
        codes [B,S,D] (long) / quantized [B,S,dim] (straight-through) / commit (scalar) /
        logits: D tensors [B,S,K] (with ce_labels, levels >= 2 use teacher-forced posterior residuals) /
        n_revived / diversity / soft_entropy_bits
        """
        if update_codebook and torch.is_grad_enabled():
            # tau is updated only in the posterior pass and shared by the prior pass
            self._fp32_buffers()
            with torch.no_grad():
                self._update_tau(self._dist2(z.reshape(-1, self.dim).float(), self.codebook[0]))

        self._fp32_buffers()
        self._sync_once()
        in_dtype = z.dtype
        B, S, _ = z.shape
        x = z.reshape(-1, self.dim).float()
        residual = x
        quant = torch.zeros_like(x)
        codes, logits = [], []
        commit = x.new_zeros(())
        do_update = update_codebook and torch.is_grad_enabled()
        n_revived = 0

        for d_lv in range(self.D):
            cb = self.codebook[d_lv]
            dist2 = self._dist2(residual, cb)
            logits.append((-dist2 / self.tau).reshape(B, S, self.K))
            idx = dist2.argmin(dim=-1)
            q = cb[idx]
            commit = commit + F.mse_loss(residual, q.detach())
            if do_update:
                onehot = F.one_hot(idx, self.K).float()
                with torch.no_grad():
                    self._ema_update(d_lv, residual.detach(), onehot)
                    self.usage_count[d_lv] += onehot.sum(0)
                    cnt = onehot.sum(0)                                 # recent count (summed over ranks)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(cnt)
                    self.recent_count[d_lv].mul_(self.revive_decay).add_(cnt)
            codes.append(idx.reshape(B, S))
            quant = quant + q
            if ce_labels is not None and d_lv + 1 < self.D:
                tf_q = self.codebook[d_lv][ce_labels[..., d_lv].reshape(-1)]
                residual = residual - tf_q
            else:
                residual = residual - q

        # dead-code restart, periodically, after the EMA update of all levels
        if do_update:
            with torch.no_grad():
                self.update_step += 1
                if self.revive_every > 0 and int(self.update_step.item()) % self.revive_every == 0:
                    res = x
                    for d_lv in range(self.D):
                        cb = self.codebook[d_lv]
                        d2 = self._dist2(res, cb)
                        n_revived += self._revive_dead_codes(d_lv, res.detach(), d2)
                        res = res - self.codebook[d_lv][d2.argmin(dim=-1)]
                    self._broadcast_codebook()                          # keep ranks consistent

        if ce_labels is not None and self.D >= 2:
            # logits of levels >= 2 are computed from residuals teacher-forced with the posterior codes
            tf_res = z.reshape(-1, self.dim).float()
            tf_logits = [logits[0]]
            for d_lv in range(1, self.D):
                tf_res = tf_res - self.codebook[d_lv - 1][ce_labels[..., d_lv - 1].reshape(-1)]
                tf_logits.append((-self._dist2(tf_res, self.codebook[d_lv]) / self.tau).reshape(B, S, self.K))
            logits = tf_logits

        with torch.no_grad():
            self.quant_err.mul_(0.99).add_(F.mse_loss(x, quant), alpha=0.01)
        quantized = x + (quant - x).detach()                            # straight-through estimator
        out = {
            "codes": torch.stack(codes, dim=-1),                        # [B,S,D]
            "quantized": quantized.reshape(B, S, self.dim).to(in_dtype),
            "commit": self.commit_weight * commit,
            "logits": logits,
            "n_revived": n_revived,
        }
        # level-0 logits are never teacher-forced, so they are the same quantity in both passes
        div, soft_ent = self._diversity_from_logits(out["logits"])
        out["diversity"] = div
        out["soft_entropy_bits"] = soft_ent
        return out

    @torch.no_grad()
    def infer_quantized(self, z: torch.Tensor, restrict_seen: bool = True):
        """Inference: level-wise nearest code, optionally restricted to codes used by the posterior branch.
        z: [B,S,dim] -> quantized [B,S,dim], codes [B,S,D]"""
        self._fp32_buffers()
        B, S, _ = z.shape
        residual = z.reshape(-1, self.dim).float()
        quant = torch.zeros_like(residual)
        codes = []
        for d_lv in range(self.D):
            dist2 = self._dist2(residual, self.codebook[d_lv])
            if restrict_seen and self.usage_count[d_lv].sum() > 0:
                unseen = self.usage_count[d_lv] <= 0
                dist2 = dist2.masked_fill(unseen[None, :], float("inf"))
            idx = dist2.argmin(-1)
            q = self.codebook[d_lv][idx]
            quant = quant + q
            residual = residual - q
            codes.append(idx.reshape(B, S))
        return quant.reshape(B, S, self.dim).to(z.dtype), torch.stack(codes, dim=-1)


class DriftAwareSkillQuantizer(ResidualSkillQuantizer):
    """Residual quantizer with drift-aware skill quantization (DASQ).

    Kernel-weighted EMA: the EMA statistics of level d use the assignment weights
        w_nj = 1                                     if q_n = j
        w_nj = mu * exp(-d_nj / (zeta * delta_bar))  otherwise,
    where d_nj is the squared distance from the posterior residual u_n to code c_j and delta_bar is the EMA (coefficient
    `kernel_ema`) of the batch-average quantization error; both levels share one delta_bar, updated with the quantization
    error of a level whenever that level is updated. mu = 0 recovers the hard-assignment EMA. Diversity regularization,
    the temperature kappa, the straight-through estimator, and teacher-forced prior logits are inherited.

    Codes are c_j = m_j / N_j, as in Eq. (4) of the paper; a code whose statistics have decayed to zero keeps its vector.
    The EMA statistics start as N_j = 1 and m_j = c_j (the same state a restarted code receives), so the randomly
    initialized codes move smoothly towards the posterior outputs.

    Dead-code restart counts only hard assignments and resets the dead codes to the posterior residuals with the largest
    quantization errors of the whole batch (all ranks), not of the local batch of one rank.
    """

    count_floor = 1e-12     # N_j below this value: the statistics of the code have fully decayed

    def __init__(self, *args, kernel_weight: float = 0.1, kernel_scale: float = 4.0,
                 kernel_ema: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_cluster.fill_(1.0)                   # N_j = 1; m_j = c_j is set by the parent (ema_embed_sum = codebook)
        self.kernel_weight = float(kernel_weight)     # mu
        self.kernel_scale = float(kernel_scale)       # zeta
        self.kernel_ema = float(kernel_ema)
        self.register_buffer("delta_bar", torch.zeros(()))     # EMA of the batch-average quantization error; 0 = unset
        self.register_buffer("kernel_width", torch.zeros(()))  # zeta * delta_bar, for logging
        self.register_buffer("kernel_mass", torch.zeros(()))   # mean total weight given to non-selected codes, for logging

    def _fp32_buffers(self):
        super()._fp32_buffers()
        for name in ("delta_bar", "kernel_width", "kernel_mass"):
            b = getattr(self, name)
            if b.dtype != torch.float32:
                setattr(self, name, b.float())

    @torch.no_grad()
    def _update_delta_bar(self, d2: torch.Tensor) -> float:
        """d2: [N,K]. Update delta_bar with the batch-average quantization error (averaged over ranks)."""
        delta = d2.min(dim=-1).values.mean()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(delta); delta = delta / dist.get_world_size()
        if float(self.delta_bar.item()) <= 0:
            self.delta_bar.fill_(float(delta))
        else:
            self.delta_bar.mul_(self.kernel_ema).add_(delta, alpha=1 - self.kernel_ema)
        width = max(self.kernel_scale * float(self.delta_bar.item()), 1e-8)
        self.kernel_width.fill_(width)
        return width

    def _ema_update(self, level: int, x: torch.Tensor, onehot: torch.Tensor):
        """EMA update of Eq. (4) with the assignment weights of Eq. (7); mu = 0 gives the hard-assignment weights."""
        w = onehot
        if self.kernel_weight > 0:
            d2 = self._dist2(x, self.codebook[level])                  # [N,K]
            width = self._update_delta_bar(d2)
            w_soft = self.kernel_weight * torch.exp(-d2 / width) * (1.0 - onehot)
            self.kernel_mass.fill_(float(w_soft.sum(dim=-1).mean().item()))
            w = onehot + w_soft
        cluster = _all_reduce_(w.sum(0))                               # [K]
        embed_sum = _all_reduce_(w.t() @ x)                            # [K,dim]
        d = self.ema_decay
        self.ema_cluster[level].mul_(d).add_(cluster, alpha=1 - d)     # N_j
        self.ema_embed_sum[level].mul_(d).add_(embed_sum, alpha=1 - d)  # m_j
        n = self.ema_cluster[level]
        updated = self.ema_embed_sum[level] / n.clamp_min(self.count_floor).unsqueeze(-1)   # c_j = m_j / N_j
        self.codebook[level] = torch.where((n > self.count_floor).unsqueeze(-1), updated, self.codebook[level])

    @torch.no_grad()
    def _revive_dead_codes(self, level: int, residual: torch.Tensor, dist2: torch.Tensor):
        """Reset the dead codes to the posterior residuals with the largest quantization errors of the whole batch.
        Every rank contributes its local candidates with the largest errors (at most one per dead code), and the
        candidates with the largest errors among all ranks are chosen, identically on every rank; the parent then
        broadcasts the codebook of rank 0, so the added noise is the same everywhere.
        residual: [N,dim] (fp32, detached), dist2: [N,K]"""
        dead = (self.recent_count[level] < self.revive_threshold)
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        err = dist2.min(dim=-1).values.float()                          # [N] d^2 to the nearest code
        n_local = min(n_dead, residual.shape[0])
        top = err.topk(n_local)
        cand = residual.new_zeros(n_dead, self.dim)                     # fixed shapes on every rank
        cand_err = err.new_full((n_dead,), float("-inf"))
        cand[:n_local] = residual[top.indices]
        cand_err[:n_local] = top.values
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world = dist.get_world_size()
            cands = [torch.zeros_like(cand) for _ in range(world)]
            errs = [torch.zeros_like(cand_err) for _ in range(world)]
            dist.all_gather(cands, cand)
            dist.all_gather(errs, cand_err)
            cand, cand_err = torch.cat(cands, dim=0), torch.cat(errs, dim=0)
        n_take = min(n_dead, int(torch.isfinite(cand_err).sum().item()))
        cand = cand[cand_err.topk(n_take).indices]                      # [n_take,dim] largest errors of the batch
        if n_take < n_dead:                                             # resample if the batch is too small
            rep = cand[torch.randint(0, n_take, (n_dead - n_take,), device=cand.device)]
            cand = torch.cat([cand, rep], dim=0)
        cand = cand + self.revive_noise * torch.randn_like(cand)
        idx = dead.nonzero(as_tuple=True)[0]
        self.codebook[level][idx] = cand
        self.ema_cluster[level][idx] = 1.0                              # N_j = 1
        self.ema_embed_sum[level][idx] = cand                           # m_j = c_j
        self.recent_count[level][idx] = self.revive_threshold           # N~_j = N_min
        self.revive_total += n_dead
        return n_dead


# ======================================================================================================== Q-Formers
class PosteriorQFormer(nn.Module):
    """Posterior Q-Former (training only).

    Sequence: S learnable queries, the observation tokens (one per time step, linearly projected, plus the positional
    embedding of the step) and, if `action_dim > 0`, the action tokens alpha_k = W_alpha [A_{2k}; A_{2k+1}] (plus a type
    embedding and the positional embedding of step k). After self-attention, each query output is projected to
    `code_dim`; with `concat_queries=True` the S outputs are concatenated into one vector [B, 1, S * code_dim],
    otherwise they are returned as S vectors [B, S, code_dim].
    """

    def __init__(self, feat_dim: int = 1024, hidden: int = 256, num_frames: int = 4, num_skill_tokens: int = 4,
                 code_dim: int = 32, num_layers: int = 2, num_heads: int = 8,
                 use_obs: bool = True, action_dim: int = 0, concat_queries: bool = False):
        super().__init__()
        assert use_obs or action_dim > 0, "the posterior needs observation changes and/or actions"
        self.num_frames, self.S, self.concat_queries = num_frames, num_skill_tokens, concat_queries
        if use_obs:
            self.in_proj = nn.Linear(feat_dim, hidden)
        self.frame_pos = nn.Parameter(torch.zeros(1, num_frames, hidden))
        self.skill_queries = nn.Parameter(torch.zeros(1, num_skill_tokens, hidden))
        nn.init.trunc_normal_(self.frame_pos, std=0.02)
        nn.init.trunc_normal_(self.skill_queries, std=0.02)
        if action_dim > 0:
            self.act_proj = nn.Linear(2 * action_dim, hidden)            # W_alpha
            self.act_type = nn.Parameter(torch.zeros(1, 1, hidden))
            nn.init.trunc_normal_(self.act_type, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, num_heads, hidden * 4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.out_proj = nn.Linear(hidden, code_dim)

    def forward(self, obs: Optional[torch.Tensor] = None, cum_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """obs: [B, K, feat_dim] observation-change tokens; cum_actions: [B, 2K, action_dim] cumulative actions"""
        x = obs if obs is not None else cum_actions
        B = x.shape[0]
        tokens = [self.skill_queries.expand(B, -1, -1)]
        if obs is not None:
            tokens.append(self.in_proj(obs.to(self.in_proj.weight.dtype)) + self.frame_pos)
        if cum_actions is not None:
            pairs = cum_actions.reshape(B, self.num_frames, -1)        # [A_{2k}; A_{2k+1}]
            tokens.append(self.act_proj(pairs.to(self.act_proj.weight.dtype)) + self.act_type + self.frame_pos)
        h = self.encoder(torch.cat(tokens, dim=1))
        out = self.out_proj(h[:, : self.S])
        return out.reshape(B, 1, -1) if self.concat_queries else out


class PriorQFormer(nn.Module):
    """Prior Q-Former (training and inference): S learnable queries cross-attend to the latent action / skill token
    outputs [B, N, vlm_dim]; each query output is projected to `code_dim` ([B, S, code_dim]), or concatenated into one
    vector [B, 1, S * code_dim] with `concat_queries=True`."""

    def __init__(self, vlm_dim: int = 2048, hidden: int = 256, num_skill_tokens: int = 4,
                 code_dim: int = 32, num_heads: int = 8, concat_queries: bool = False):
        super().__init__()
        self.concat_queries = concat_queries
        self.in_proj = nn.Linear(vlm_dim, hidden)
        self.queries = nn.Parameter(torch.zeros(1, num_skill_tokens, hidden))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, code_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        kv = self.in_proj(tokens.to(self.in_proj.weight.dtype))
        q = self.queries.expand(B, -1, -1)
        h, _ = self.attn(q, kv, kv)
        h = self.norm1(h + q)
        h = self.norm2(h + self.ffn(h))
        out = self.out_proj(h)
        return out.reshape(B, 1, -1) if self.concat_queries else out


# ======================================================================================================== decoders
class SkillProjector(nn.Module):
    """Quantized skill vectors (code_dim) -> backbone hidden size (2048)."""

    def __init__(self, code_dim: int = 32, hidden: int = 256, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(code_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, q):
        return self.net(q.to(self.net[0].weight.dtype))


class MotionDecoder(nn.Module):
    """MLP: [flattened skill | optional condition (cond_dim)] -> observation changes of the following (T-1) time steps
    [B, num_future, feat_dim]."""

    def __init__(self, code_dim: int = 32, num_skill_tokens: int = 4, num_future: int = 3, feat_dim: int = 1024,
                 hidden: int = 512, cond_dim: int = 0):
        super().__init__()
        self.num_future, self.feat_dim = num_future, feat_dim
        self.net = _mlp(code_dim * num_skill_tokens + cond_dim, hidden, num_future * feat_dim)

    def forward(self, skill: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = skill.shape[0]
        dt = self.net[0].weight.dtype
        x = skill.to(dt).reshape(B, -1)
        if cond is not None:
            x = torch.cat([x, cond.to(dt)], dim=-1)
        return self.net(x).reshape(B, self.num_future, self.feat_dim)


class ConditionalMotionDecoder(nn.Module):
    """[skill | pooled current time step] -> observation changes of the following (T-1) time steps, with separate input
    projections of the skill and the condition (v2 parametrization)."""

    def __init__(self, code_dim: int = 32, num_skill_tokens: int = 2, num_future: int = 3,
                 feat_dim: int = 1024, hidden: int = 512):
        super().__init__()
        self.num_future, self.feat_dim = num_future, feat_dim
        self.cond_proj = nn.Linear(feat_dim, hidden)
        self.code_proj = nn.Linear(code_dim * num_skill_tokens, hidden)
        self.net = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, num_future * feat_dim))

    def forward(self, skill: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = skill.shape[0]
        dt = self.code_proj.weight.dtype
        h = self.code_proj(skill.to(dt).reshape(B, -1)) + self.cond_proj(cond.to(dt))
        return self.net(h).reshape(B, self.num_future, self.feat_dim)


class TransformerActionDecoder(nn.Module):
    """Skill [B,S,code_dim] -> action chunk [B,n_steps,action_dim], without observation input: fixed per-step queries
    cross-attend to the skill vectors."""

    def __init__(self, code_dim: int = 32, hidden: int = 256, n_steps: int = 8,
                 action_dim: int = 7, num_layers: int = 2, num_heads: int = 4,
                 max_skill_tokens: int = 16):
        super().__init__()
        self.n_steps = n_steps
        self.in_proj = nn.Linear(code_dim, hidden)
        self.skill_pos = nn.Parameter(torch.zeros(1, max_skill_tokens, hidden))
        self.step_queries = nn.Parameter(torch.zeros(1, n_steps, hidden))
        nn.init.trunc_normal_(self.skill_pos, std=0.02)
        nn.init.trunc_normal_(self.step_queries, std=0.02)
        layer = nn.TransformerDecoderLayer(hidden, num_heads, hidden * 4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.out_proj = nn.Linear(hidden, action_dim)

    def forward(self, quantized: torch.Tensor) -> torch.Tensor:
        B, S, _ = quantized.shape
        mem = self.in_proj(quantized.to(self.in_proj.weight.dtype)) + self.skill_pos[:, :S]
        q = self.step_queries.expand(B, -1, -1)
        h = self.decoder(tgt=q, memory=mem)
        return self.out_proj(h)


class MLPActionDecoder(nn.Module):
    """Skill -> action chunk [B, horizon, action_dim], MLP without observation input."""

    def __init__(self, code_dim: int = 32, horizon: int = 7, action_dim: int = 7, hidden: int = 512):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.net = _mlp(code_dim, hidden, horizon * action_dim)

    def forward(self, skill: torch.Tensor) -> torch.Tensor:
        B = skill.shape[0]
        return self.net(skill.reshape(B, -1).to(self.net[0].weight.dtype)).reshape(B, self.horizon, self.action_dim)


class PosteriorEMA(nn.Module):
    """EMA copy of the posterior Q-Former (the codebook is shared, not copied). It is not used by any loss and is kept so
    that the parameter layout matches the v1 / v2 checkpoints."""

    def __init__(self, posterior: PosteriorQFormer, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow = copy.deepcopy(posterior)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, posterior: PosteriorQFormer):
        for ps, p in zip(self.shadow.parameters(), posterior.parameters()):
            ps.mul_(self.decay).add_(p.detach().float().to(ps.dtype), alpha=1 - self.decay)
        for bs, b in zip(self.shadow.buffers(), posterior.buffers()):
            bs.copy_(b)


# ======================================================================================================== targets / losses
def cumulative_action_targets(actions: torch.Tensor, gripper_dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """actions: [B, H, A] chunk a_t..a_{t+H-1} (pose increments + gripper command), frames 0..H of the window;
    a_{t+i} is the command issued at frame i and executed between frames i and i+1.

    Returns
      cum:     [B, H+1, A] cumulative action A_i of frame i: pose = sum_{i'<i} a_{t+i'} (increments executed before
               frame i, 0 for frame 0); gripper = change of the command issued at frame i from the command issued at
               frame 0 (frame H issues no new command and keeps the last one), in {-1, 0, 1}
      a_tilde: [B, H, A]   the chunk with each gripper command replaced by its change from the initial command,
               i.e. the same gripper representation as A_0..A_{H-1}
    """
    B, H, A = actions.shape
    g = gripper_dim % A
    pose_idx = [i for i in range(A) if i != g]
    pose = actions[..., pose_idx]                                     # [B,H,A-1]
    grip = actions[..., g]                                            # [B,H]
    rel = grip - grip[:, :1]                                          # change from the initial command
    cum_pose = torch.cat([pose.new_zeros(B, 1, A - 1), pose.cumsum(dim=1)], dim=1)   # [B,H+1,A-1]
    cum_grip = torch.cat([rel, rel[:, -1:]], dim=1)                   # [B,H+1]

    def _assemble(p, gr):
        out = p.new_empty(*p.shape[:-1], A)
        out[..., pose_idx] = p
        out[..., g] = gr
        return out

    return _assemble(cum_pose, cum_grip), _assemble(pose, rel)


def coarse_to_fine_ce(logits: List[torch.Tensor], labels: torch.Tensor,
                      level_weights: Optional[List[float]] = None):
    """logits: D tensors [B,S,K]; labels: [B,S,D] -> (loss, top-1 accuracy of level 1)"""
    D = labels.shape[-1]
    if level_weights is None:
        level_weights = [1.0] * D
    loss = logits[0].new_zeros(())
    for d_lv in range(D):
        lg = logits[d_lv].reshape(-1, logits[d_lv].shape[-1]).float()
        loss = loss + level_weights[d_lv] * F.cross_entropy(lg, labels[..., d_lv].reshape(-1))
    loss = loss / sum(level_weights[:D])
    with torch.no_grad():
        acc = (logits[0].argmax(-1) == labels[..., 0]).float().mean()
    return loss, acc
