# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

torch.set_printoptions(threshold=10_000)


class LabelSmoothedCrossEntropy(torch.nn.Module):
    """This is a version for the RLST class itself and RLSTCriterion"""
    def __init__(self, label_smoothing=0.0, ignore_index=None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def label_smoothed_nll_loss(self, lprobs, target, label_smoothing, reduce=True):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        pad_count = 0
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            pad_count = pad_mask.sum()
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)

        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            non_pad_count = lprobs.size()[0] - pad_count
            nll_loss = nll_loss.sum() / non_pad_count
            smooth_loss = smooth_loss.sum() / non_pad_count
        lab_sm_i = label_smoothing / (lprobs.size(-1) - 1)
        loss = (1.0 - label_smoothing - lab_sm_i) * nll_loss + lab_sm_i * smooth_loss
        return loss, nll_loss

    def forward(self, net_output, target, reduce):
        lprobs = torch.nn.functional.log_softmax(net_output, dim=-1)
        loss, nll_loss = self.label_smoothed_nll_loss(lprobs, target, self.label_smoothing, reduce=reduce)
        return loss, nll_loss


@dataclass
class RLSTCriterionConfig(FairseqDataclass):
    N: int = field(
        default=100_000,
        metadata={"help": "N"}
    )
    smoothing: float = field(
        default=0.0,
        metadata={"help": "label smoothing for cross entropy mistranslation loss"},
    )
    epsilon_max: float = field(
        default=0.4,
        metadata={"help": "starting epsilon value"}
    )
    epsilon_min: float = field(
        default=0.4,
        metadata={"help": "ending epsilon value"}
    )
    teacher_forcing_max: float = field(
        default=1.0,
        metadata={"help": "starting teacher force"}
    )
    teacher_forcing_min: float = field(
        default=1.0,
        metadata={"help": "final teacher force"}
    )
    eta_min: float = field(
        default=0.02,
        metadata={"help": "minimum value of policy multiplier"}
    )
    eta_max: float = field(
        default=0.2,
        metadata={"help": "maximum value of policy multiplier"}
    )
    rho: float = field(
        default=0.99,
        metadata={"help": "rho"}
    )
    rtf_delta: float = field(
        default=5.0,
        metadata={"help": "rtf_delta"}
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion(
    "rlst_criterion", dataclass=RLSTCriterionConfig
)
class RLSTCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, N, smoothing, eta_min, eta_max, epsilon_max, epsilon_min, teacher_forcing_max, teacher_forcing_min, rho, rtf_delta):
        super().__init__(task)
        if sentence_avg:
            raise NotImplementedError("RLST does not support sentence-avg")
        self.RHO = rho
        self.rho_to_n = 1  # n is minibatch index
        self.mistranslation_loss_weight = 0
        self.policy_loss_weight = 0
        self.mistranslation_criterion = LabelSmoothedCrossEntropy(ignore_index=self.padding_idx, label_smoothing=smoothing)
        self.policy_criterion = nn.MSELoss(reduction="sum")
        self.n = 0
        self.N = N
        self.rtf_delta = rtf_delta
        self.eta = None
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.epsilon = None
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.teacher_forcing = None
        self.teacher_forcing_max = teacher_forcing_max
        self.teacher_forcing_min = teacher_forcing_min

    def forward(self, model, sample, reduce=True):
        src_tokens = sample["net_input"]["src_tokens"]
        trg_tokens = sample["target"]
        self.teacher_forcing = self.teacher_forcing_min + (self.teacher_forcing_max - self.teacher_forcing_min) * math.e ** ((-3) * self.n / self.N)
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.e ** ((-3) * self.n / self.N)
        rtf_prob = math.e ** (-self.n/self.N)
        word_outputs, Q_used, Q_target, is_read, is_write = model(src_tokens, trg_tokens, self.epsilon, self.teacher_forcing, self.rtf_delta, rtf_prob)
        weighted_loss, mistranslation_loss, nll_loss, policy_loss = self.compute_loss(word_outputs, trg_tokens, Q_used, Q_target)
        self.n += 1
        total_time_steps = int((is_write + is_read).sum())
        ntokens = sample["ntokens"]
        read_lead, read_lead_relative = self.compute_read_lead_metrics(is_read, is_write)

        logging_output = {
            "ntokens": ntokens,
            "nsentences": sample["target"].size(0),
            "mistranslation_loss": ntokens * mistranslation_loss.data,
            "nll_loss": ntokens * nll_loss.data,
            "policy_loss": total_time_steps * policy_loss.data,
            "total_time_steps": total_time_steps,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "tf": self.teacher_forcing,
            "rtf_prob": rtf_prob,
            "workers_num": 1,
            "total_reads": int(is_read.sum()),
            "read_lead": float(read_lead.sum()),
            "read_lead_relative": float(read_lead_relative.sum())
        }

        return weighted_loss * ntokens, ntokens, logging_output

    def compute_loss(self, word_outputs, trg, Q_used, Q_target):

        word_outputs = word_outputs.reshape(-1, word_outputs.shape[-1])
        trg = trg.view(-1)

        mistranslation_loss, nll_loss = self.mistranslation_criterion(word_outputs, trg, reduce=True)
        self.eta = self.eta_max - (self.eta_max - self.eta_min) * math.e ** ((-3) * self.n / self.N)
        policy_loss = self.policy_criterion(Q_used, Q_target)/torch.count_nonzero(Q_target)
        self.rho_to_n *= self.RHO
        w_k = (self.RHO - self.rho_to_n) / (1 - self.rho_to_n)
        self.mistranslation_loss_weight = w_k * self.mistranslation_loss_weight + (1 - w_k) * float(mistranslation_loss)
        self.policy_loss_weight = w_k * self.policy_loss_weight + (1 - w_k) * float(policy_loss)
        weighted_loss = self.eta * policy_loss / self.policy_loss_weight + mistranslation_loss / self.mistranslation_loss_weight
        return weighted_loss, mistranslation_loss, nll_loss, policy_loss

    @staticmethod
    def compute_read_lead_metrics(is_read, is_write):
        """This metric measures how much write action is ahead of read action in time."""
        with torch.no_grad():
            action_weights = torch.cumsum(is_read + is_write, dim=1)
            episode_length = action_weights[:, -1]
            read_weights = action_weights - 1
            read_weights[~is_read] = 0
            write_weights = action_weights - 1
            write_weights[~is_write] = 0
            mean_read = torch.sum(read_weights, dim=1) / (is_read.sum(dim=1) + 1e-8)
            mean_write = torch.sum(write_weights, dim=1) / (is_write.sum(dim=1) + 1e-8)
            read_lead = mean_write - mean_read
            return read_lead, read_lead / episode_length

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        mistranslation_loss = sum(log.get("mistranslation_loss", 0) for log in logging_outputs)
        nll_loss = sum(log.get("nll_loss", 0) for log in logging_outputs)
        policy_loss = sum(log.get("policy_loss", 0) for log in logging_outputs)
        eta = sum(log.get("eta", 0) for log in logging_outputs)
        epsilon = sum(log.get("epsilon", 0) for log in logging_outputs)
        teacher_forcing = sum(log.get("tf", 0) for log in logging_outputs)
        rtf_prob = sum(log.get("rtf_prob", 0) for log in logging_outputs)
        workers = sum(log.get("workers_num", 0) for log in logging_outputs)
        total_time_steps = sum(log.get("total_time_steps", 0) for log in logging_outputs)

        total_reads = sum(log.get("total_reads", 0) for log in logging_outputs)
        read_relative_frequency = total_reads / total_time_steps
        read_lead = sum(log.get("read_lead", 0) for log in logging_outputs)
        read_lead_relative = sum(log.get("read_lead_relative", 0) for log in logging_outputs)

        metrics.log_scalar("loss", mistranslation_loss / ntokens / math.log(2), ntokens, round=2, priority=1)
        metrics.log_scalar("nll_loss", nll_loss / ntokens / math.log(2), ntokens, round=2, priority=2)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg, round=2, base=2), priority=3)
        metrics.log_scalar("policy_loss", policy_loss / total_time_steps, total_time_steps, round=2, priority=4)

        metrics.log_scalar("eta", eta / workers, 0, round=2, priority=26)
        metrics.log_scalar("epsilon", epsilon / workers, 0, round=2, priority=27)
        metrics.log_scalar("tf", teacher_forcing / workers, 0, round=2, priority=28)
        metrics.log_scalar("rtf", rtf_prob / workers, 0, round=2, priority=29)

        metrics.log_scalar("read_rf", read_relative_frequency, ntokens, round=2, priority=36)
        metrics.log_scalar("read_lead", read_lead / nsentences, nsentences, round=2, priority=37)
        metrics.log_scalar("read_lead_r", read_lead_relative / nsentences, nsentences, round=2, priority=38)
        metrics.log_scalar("ntokens", ntokens, round=2, priority=39)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
