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


@dataclass
class RLSTCriterionConfig(FairseqDataclass):
    N: int = field(
        default=100_000,
        metadata={"help": "N"},
    )
    epsilon: float = field(
        default=0.15,
        metadata={"help": "epsilon"},
    )
    teacher_forcing: float = field(
        default=1.0,
        metadata={"help": "teacher force"},
    )
    rho: float = field(
        default=0.99,
        metadata={"help": "rho"},
    )
    eta_min: float = field(
        default=1/30,
        metadata={"help": "minimum value of policy multiplier"},
    )
    eta_max: float = field(
        default=0.2,
        metadata={"help": "maximum value of policy multiplier"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion(
    "rlst_criterion", dataclass=RLSTCriterionConfig
)
class RLSTCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, N, epsilon, teacher_forcing, rho, eta_min, eta_max):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.RHO = rho
        self.rho_to_n = 1  # n is minibatch index
        self.mistranslation_loss_weight = 0
        self.policy_loss_weight = 0
        self.mistranslation_criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        self.policy_criterion = nn.MSELoss(reduction="sum")
        self.policy_multiplier = None
        self.n = -1
        self.N = N
        self.epsilon = epsilon
        self.teacher_forcing = teacher_forcing
        self.eta_min = eta_min
        self.eta_max = eta_max

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens = sample["net_input"]["src_tokens"]
        trg_tokens = sample["target"]
        src_tokens = src_tokens.T.contiguous()
        trg_tokens = trg_tokens.T.contiguous()
        word_outputs, Q_used, Q_target, actions = model(src_tokens, trg_tokens, self.epsilon, self.teacher_forcing)
        loss, mistranslation_loss, policy_loss, _ = self.compute_loss(word_outputs, trg_tokens, Q_used, Q_target)
        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        logging_output = {
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "actions": actions,
            "mistranslation_loss": mistranslation_loss,
            "policy_loss": policy_loss,
            "policy_multiplier": self.policy_multiplier,
            "epsilon": self.epsilon
        }

        return sample_size * loss, sample_size, logging_output

    def compute_loss(self, word_outputs, trg, Q_used, Q_target):
        if not self.training:
            word_outputs = word_outputs[:trg.size()[0], :, :]

        word_outputs = word_outputs.view(-1, word_outputs.shape[-1])
        trg = trg.view(-1)

        mistranslation_loss = self.mistranslation_criterion(word_outputs, trg)
        if self.training:
            self.n += 1
            self.policy_multiplier = self.eta_max - (self.eta_max - self.eta_min) * math.e ** ((-3) * self.n / self.N)
            policy_loss = self.policy_criterion(Q_used, Q_target)/torch.count_nonzero(Q_target)
            self.rho_to_n *= self.RHO
            w_k = (self.RHO - self.rho_to_n) / (1 - self.rho_to_n)
            self.mistranslation_loss_weight = w_k * self.mistranslation_loss_weight + (1 - w_k) * float(mistranslation_loss)
            self.policy_loss_weight = w_k * self.policy_loss_weight + (1 - w_k) * float(policy_loss)
            loss = policy_loss * self.policy_multiplier / self.policy_loss_weight + mistranslation_loss / self.mistranslation_loss_weight
            return loss, mistranslation_loss, policy_loss, self.policy_multiplier
        else:
            return -1.0, mistranslation_loss, -1.0, self.policy_multiplier

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        mistranslation_loss = sum(log.get("mistranslation_loss", 0) for log in logging_outputs)
        policy_loss = sum(log.get("policy_loss", 0) for log in logging_outputs)
        policy_multiplier = sum(log.get("policy_multiplier", 0) for log in logging_outputs)
        epsilon = sum(log.get("epsilon", 0) for log in logging_outputs)/len(logging_outputs)

        total_actions = sum(log.get("actions", 0) for log in logging_outputs)
        actions_ratio = RLSTCriterion.actions_ratio(total_actions.squeeze(1).tolist())

        metrics.log_scalar("loss", mistranslation_loss, sample_size, round=3, priority=0)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg, round=2, base=math.e), priority=1)
        metrics.log_scalar("policy_loss", policy_loss, sample_size, round=2, priority=2)
        metrics.log_scalar("plm", policy_multiplier, round=2, priority=3)
        metrics.log_scalar("eps", epsilon, round=2, priority=4)
        metrics.log_scalar("reads", actions_ratio[0], round=2, priority=5)
        metrics.log_scalar("writes", actions_ratio[1], round=2, priority=6)
        metrics.log_scalar("boths", actions_ratio[2], round=2, priority=7)
        metrics.log_scalar("sample_size", sample_size, round=3, priority=99)

    @staticmethod
    def actions_ratio(actions):
        s = sum(actions)
        a = [actions[0] / s, actions[1] / s, actions[2] / s]
        return [round(action, 2) for action in a]

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
