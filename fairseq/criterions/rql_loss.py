# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

torch.set_printoptions(threshold=10_000)

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
        self.mistranslation_criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        self.policy_criterion = nn.MSELoss()
        self.mistranslation_loss_multiplier = 30
        self.epsilon = 0.5
        self.teacher_forcing = 0.5

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
        loss, mistranslation_loss = self.compute_loss(word_outputs, trg_tokens, Q_used, Q_target)
        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        logging_output = {
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "loss": loss.data,
            "actions": actions,
            "mistranslation_loss": mistranslation_loss,
            "epsilon": self.epsilon
        }

        if self.training:
            self.epsilon = max(0.05, self.epsilon - 0.001)
        return loss, sample_size, logging_output

    def compute_loss(self, word_outputs, trg, Q_used, Q_target):
        if not self.training:
            word_outputs = word_outputs[:trg.size()[0], :, :]

        word_outputs = word_outputs.view(-1, word_outputs.shape[-1])
        trg = trg.view(-1)
        _mistranslation_loss = self.mistranslation_criterion(word_outputs, trg)

        if self.training:
            _policy_loss = self.policy_criterion(Q_used, Q_target)
            self.ro_to_k *= self.ro
            w_k = (self.ro - self.ro_to_k) / (1 - self.ro_to_k)
            self._mistranslation_loss_weight = w_k * self._mistranslation_loss_weight + (1 - w_k) * float(_mistranslation_loss)
            self._policy_loss_weight = w_k * self._policy_loss_weight + (1 - w_k) * float(_policy_loss)
            loss = _policy_loss / self._policy_loss_weight + self.mistranslation_loss_multiplier * _mistranslation_loss / self._mistranslation_loss_weight
        else:
            loss = _mistranslation_loss

        return loss, _mistranslation_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        mistranslation_loss = sum(log.get("mistranslation_loss", 0) for log in logging_outputs)
        epsilon = sum(log.get("epsilon", 0) for log in logging_outputs)/len(logging_outputs)

        total_actions = sum(log.get("actions", 0) for log in logging_outputs)
        actions_ratio = RQLCriterion.actions_ratio(total_actions.squeeze(1).tolist())

        metrics.log_scalar("loss", mistranslation_loss, sample_size, round=3, priority=0)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg, round=2, base=math.e), priority=1)
        metrics.log_scalar("reads", actions_ratio[0], round=2, priority=2)
        metrics.log_scalar("writes", actions_ratio[1], round=2, priority=3)
        metrics.log_scalar("boths", actions_ratio[2], round=2, priority=4)
        metrics.log_scalar("eps", epsilon, round=2, priority=5)

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
