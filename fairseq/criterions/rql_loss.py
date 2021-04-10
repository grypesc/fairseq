# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class RQLCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("rql_criterion", dataclass=RQLCriterionConfig)
class RQLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.ro = 0.99
        self.ro_to_k = 1
        self._mistranslation_loss_weight = 0
        self._policy_loss_weight = 0
        self.mistranslation_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_index)
        self.policy_criterion = nn.MSELoss()
        self.mistranslation_loss_multiplier = 30

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens = sample["net_input"]["src_tokens"]
        trg_tokens = sample["target"]
        word_outputs, Q_used, Q_target, actions = model(src_tokens, trg_tokens, self.epsilon, self.teacher_forcing)
        loss, mistranslation_loss = self.compute_loss(word_outputs, trg_tokens, Q_used, Q_target)
        sample_size = (sample["target"].size(0))
        logging_output = {
            "loss": loss.data,
            "mistranslation_loss": mistranslation_loss,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, word_outputs, trg, Q_used, Q_target):
        _mistranslation_loss = self.mistranslation_criterion(word_outputs, trg)

        if self.training:
            _policy_loss = self.policy_criterion(Q_used, Q_target)
            self.ro_to_k *= self.ro
            w_k = (self.ro - self.ro_to_k) / (1 - self.ro_to_k)
            self._mistranslation_loss_weight = w_k * self._mistranslation_loss_weight + (1 - w_k) * float(_mistranslation_loss)
            self._policy_loss_weight = w_k * self._policy_loss_weight + (1 - w_k) * float(_policy_loss)
            loss = _policy_loss / self._policy_loss_weight + self.mistranslation_loss_multiplier * _mistranslation_loss / self._mistranslation_loss_weight
        else:
            loss = -1

        return loss, _mistranslation_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        mistranslation_loss_sum = sum(log.get("mistranslation_loss", 0) for log in logging_outputs)
        sample_size_sum = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum, round=3
        )
        metrics.log_scalar(
            "mistranslation_loss", mistranslation_loss_sum, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size_sum, round=3
        )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
