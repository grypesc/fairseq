import copy
import random
import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.criterions.rlst_loss import LabelSmoothedCrossEntropy
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, FairseqEncoderDecoderModel


class LeakyNet(nn.Module):
    def __init__(self,
                 src_vocab_len,
                 trg_vocab_len,
                 rnn_hid_dim,
                 rnn_dropout,
                 rnn_num_layers,
                 src_embed_dim=256,
                 trg_embed_dim=256,
                 embedding_dropout=0.0,
                 ):
        super().__init__()

        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_num_layers = rnn_num_layers

        self.src_embedding = nn.Embedding(src_vocab_len, src_embed_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_len, trg_embed_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.rnn = nn.GRU(src_embed_dim + trg_embed_dim, rnn_hid_dim, num_layers=rnn_num_layers, dropout=0.0)
        self.linear = nn.Linear(rnn_hid_dim, rnn_hid_dim)
        self.activation = nn.LeakyReLU()
        self.output = nn.Linear(rnn_hid_dim, trg_vocab_len + 2)

    def forward(self, src, previous_output, rnn_state):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))
        rnn_input = torch.cat((src_embedded, trg_embedded), dim=2)
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        leaky_out = self.activation(self.linear(rnn_output))
        leaky_out = self.rnn_dropout(leaky_out)
        outputs = self.output(leaky_out)
        return outputs, rnn_state


class LeakyResidualApproximator(nn.Module):
    """Residual approximator for RLST. 'Simply the best.' - Tina Turner"""
    def __init__(self,
                 src_vocab_len,
                 trg_vocab_len,
                 src_embed_dim,
                 trg_embed_dim,
                 rnn_hid_dim,
                 rnn_dropout,
                 embedding_dropout,
                 rnn_num_layers):
        super().__init__()

        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_num_layers = rnn_num_layers
        self.src_embedding = nn.Embedding(src_vocab_len, src_embed_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_len, trg_embed_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.embedding_linear = nn.Linear(src_embed_dim + trg_embed_dim, rnn_hid_dim)
        self.rnns = nn.ModuleList([nn.GRU(rnn_hid_dim, rnn_hid_dim, batch_first=True) for _ in range(rnn_num_layers)])
        self.linear = nn.Linear(rnn_hid_dim, rnn_hid_dim)
        self.activation = nn.LeakyReLU()
        self.output = nn.Linear(rnn_hid_dim, trg_vocab_len + 2)

    def forward(self, src, previous_output, rnn_states):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))

        rnn_input = self.activation(self.embedding_linear(torch.cat((src_embedded, trg_embedded), dim=2)))
        rnn_input = self.embedding_dropout(rnn_input)
        rnn_new_states = torch.zeros(rnn_states.size(), device=src_embedded.device)
        for i, rnn in enumerate(self.rnns):
            rnn_out, rnn_new_states[i, :] = rnn(rnn_input, rnn_states[i:i + 1])
            rnn_input = rnn_out + rnn_input

        leaky_output = self.rnn_dropout(self.activation(self.linear(rnn_input)))
        outputs = self.output(leaky_output)
        return outputs, rnn_new_states

    def init_state(self, batch_size, device):
        return torch.zeros((self.rnn_num_layers, batch_size, self.rnn_hid_dim), device=device)

    def update_state(self, state, new_state, agents_ignored):
        state[:, ~agents_ignored, :] = new_state[:, ~agents_ignored, :]
        return state

    def reorder_state(self, state, new_order):
        return state.index_select(1, new_order)


class LeakyLSTM(nn.Module):
    def __init__(self,
                 src_vocab_len,
                 trg_vocab_len,
                 rnn_hid_dim,
                 rnn_dropout,
                 rnn_num_layers,
                 src_embed_dim=256,
                 trg_embed_dim=256,
                 embedding_dropout=0.0,
                 ):
        super().__init__()

        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_num_layers = rnn_num_layers

        self.src_embedding = nn.Embedding(src_vocab_len, src_embed_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_len, trg_embed_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.activation = nn.LeakyReLU()
        self.linear_embedding = nn.Linear(src_embed_dim + trg_embed_dim, rnn_hid_dim)
        self.rnn = nn.LSTM(src_embed_dim + trg_embed_dim, rnn_hid_dim, num_layers=rnn_num_layers, batch_first=True, dropout=0.0)
        self.linear = nn.Linear(rnn_hid_dim, rnn_hid_dim)
        self.output = nn.Linear(rnn_hid_dim, trg_vocab_len + 2)

    def forward(self, src, previous_output, rnn_state):
        src_embedded = self.src_embedding(src)
        trg_embedded = self.trg_embedding(previous_output)
        leaky_input = self.embedding_dropout(torch.cat((src_embedded, trg_embedded), dim=2))
        rnn_input = self.activation(self.embedding_dropout(self.linear_embedding(leaky_input)))
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        leaky_out = self.rnn_dropout(self.activation(self.linear(rnn_output)))
        outputs = self.output(leaky_out)
        return outputs, rnn_state

    def init_state(self, batch_size, device):
        return (
                torch.zeros((self.rnn_num_layers, batch_size, self.rnn_hid_dim), device=device),
                torch.zeros((self.rnn_num_layers, batch_size, self.rnn_hid_dim), device=device)
        )

    def update_state(self, state, new_state, agents_ignored):
        h_0, c_0 = state
        h_0_new, c_0_new = new_state
        h_0[:, ~agents_ignored, :] = h_0_new[:, ~agents_ignored, :]
        c_0[:, ~agents_ignored, :] = c_0_new[:, ~agents_ignored, :]
        return h_0, c_0

    def reorder_state(self, state, new_order):
        h_0, c_0 = state
        return h_0.index_select(1, new_order), c_0.index_select(1, new_order)


class DoubleHead(nn.Module):
    def __init__(self,
                 src_vocab_len,
                 trg_vocab_len,
                 rnn_hid_dim,
                 rnn_dropout,
                 rnn_num_layers,
                 src_embed_dim=256,
                 trg_embed_dim=256,
                 embedding_dropout=0.0,
                 ):
        super().__init__()

        class SharedEmbedding(nn.Module):
            def __init__(self,
                         src_vocab_len,
                         trg_vocab_len,
                         rnn_hid_dim,
                         src_embed_dim=256,
                         trg_embed_dim=256,
                         dropout=0.0,
                         ):
                super().__init__()
                self.src_embedding = nn.Embedding(src_vocab_len, src_embed_dim)
                self.trg_embedding = nn.Embedding(trg_vocab_len, trg_embed_dim)
                self.linear = nn.Linear(src_embed_dim + trg_embed_dim, rnn_hid_dim)
                self.activation = nn.LeakyReLU()
                self.dropout = nn.Dropout(dropout)
                self.rnn = nn.LSTM(rnn_hid_dim, rnn_hid_dim, num_layers=1, batch_first=True, dropout=0.0)
                self.conv1 = nn.Conv1d(5, 64, 1, stride=1, padding=0)
                self.conv2 = nn.Conv1d(64, 1, 1, stride=1, padding=0)

            def forward(self, src, prev_out, rnn_state):
                src_embedded = self.src_embedding(src)
                conv1_out = self.conv1(src_embedded)
                conv2_out = self.activation(self.conv2(conv1_out))
                trg_embedded = self.trg_embedding(prev_out)
                linear_in = self.dropout(torch.cat((conv2_out, trg_embedded), dim=2))
                rnn_in = self.dropout(self.activation(self.linear(linear_in)))
                rnn_out, new_rnn_state = self.rnn(rnn_in, rnn_state)
                return rnn_out + rnn_in, new_rnn_state

        class Head(nn.Module):
            def __init__(self,
                         input_dim,
                         rnn_hid_dim,
                         out_dim,
                         dropout=0.0,
                         ):
                super().__init__()
                self.linear1 = nn.Linear(input_dim, rnn_hid_dim)
                self.activation = nn.LeakyReLU()
                self.dropout = nn.Dropout(dropout)
                self.rnn = nn.LSTM(rnn_hid_dim, rnn_hid_dim, num_layers=1, batch_first=True, dropout=0.0)
                self.linear2 = nn.Linear(rnn_hid_dim, rnn_hid_dim)
                self.output = nn.Linear(rnn_hid_dim, out_dim)

            def forward(self, shared_in, rnn_state):
                rnn_in = self.dropout(self.activation(self.linear1(shared_in)))
                rnn_out, rnn_new_state = self.rnn(rnn_in, rnn_state)
                out_in = self.dropout(self.activation(self.linear2(rnn_out + rnn_in)))
                return self.output(out_in), rnn_new_state

        self.rnn_hid_dim = rnn_hid_dim
        self.shared_embedding = SharedEmbedding(src_vocab_len, trg_vocab_len, rnn_hid_dim, src_embed_dim, trg_embed_dim, embedding_dropout)
        self.token_head = Head(rnn_hid_dim, rnn_hid_dim, trg_vocab_len, rnn_dropout)
        self.policy_head = Head(rnn_hid_dim, rnn_hid_dim, 2, rnn_dropout)

    def forward(self, src, previous_output, rnn_state):
        shared_out, rnn_state["embed"] = self.shared_embedding(src, previous_output, rnn_state["embed"])
        token_out, rnn_state["token_head"] = self.token_head(shared_out, rnn_state["token_head"])
        policy_out, rnn_state["policy_head"] = self.policy_head(shared_out, rnn_state["policy_head"])
        return torch.cat((token_out, policy_out), dim=2), rnn_state

    def init_state(self, batch_size, device):
        return {
            "embed":
                (torch.zeros((1, batch_size, self.rnn_hid_dim), device=device),
                 torch.zeros((1, batch_size, self.rnn_hid_dim), device=device)),
            "token_head":
                (torch.zeros((1, batch_size, self.rnn_hid_dim), device=device),
                 torch.zeros((1, batch_size, self.rnn_hid_dim), device=device)),
            "policy_head":
                (torch.zeros((1, batch_size, self.rnn_hid_dim), device=device),
                 torch.zeros((1, batch_size, self.rnn_hid_dim), device=device))
        }

    def update_state(self, state, new_state, agents_ignored):
        for key in state.keys():
            h_0, c_0 = state[key]
            h_0_new, c_0_new = new_state[key]
            h_0[:, ~agents_ignored, :] = h_0_new[:, ~agents_ignored, :]
            c_0[:, ~agents_ignored, :] = c_0_new[:, ~agents_ignored, :]
        return state

    def reorder_state(self, state, new_order):
        for key in state.keys():
            h_0, c_0 = state[key]
            state[key] = h_0.index_select(1, new_order), c_0.index_select(1, new_order)
        return state


@register_model('rlst')
class RLST(FairseqEncoderDecoderModel):
    """
    This class implements RLST algorithm presented in the paper. Given batch size of n, it creates n partially observable
    training or testing environments in which n interpreter agents operate in order to transform source sequences into the target ones.
    At time t each agent can be at different indices in input and output sequences, this indices are vectors i and j.
    At t=0 each agent is fed with first source token. Then in a time loop it performs actions based on its observations
    and approximator state until all agents are terminated or the time is up.
    """
    def __init__(self, approximator, trg_vocab_len, discount, m, mistranslation_loss,
                 src_eos_index, src_null_index, src_pad_index, trg_eos_index, trg_null_index, trg_pad_index,
                 incremental_encoder, incremental_decoder):
        super().__init__(incremental_encoder, incremental_decoder)
        self.approximator = approximator
        self.trg_vocab_len = trg_vocab_len
        self.DISCOUNT = discount
        self.M = m  # Read after eos punishment
        self.mistranslation_loss = mistranslation_loss

        self.SRC_EOS = src_eos_index
        self.SRC_NULL = src_null_index
        self.SRC_PAD = src_pad_index
        self.TRG_EOS = trg_eos_index
        self.TRG_NULL = trg_null_index
        self.TRG_PAD = trg_pad_index

        self.encoder = incremental_encoder
        self.decoder = incremental_decoder

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        parser.add_argument(
            '--rnn-hid-dim', type=int, metavar='N',
            help='dimensionality of the rnn hiddens state',
        )
        parser.add_argument(
            '--rnn-num-layers', type=int, metavar='N',
            help='number of rnn layers',
        )
        parser.add_argument(
            '--src-embed-dim', type=int, metavar='N',
            help='dimension of embedding layer for source tokens',
        )
        parser.add_argument(
            '--trg-embed-dim', type=int, metavar='N',
            help='dimension of embedding layer for target tokens',
        )
        parser.add_argument(
            '--max-testing-time', type=int, metavar='N',
            help='maximum duration of a testing episode, e.g. during generation. If too low, agents will not be'
                 'able to process the whole sequence. By default it is adjusted based on length of source sentence'
        )
        parser.add_argument(
            '--discount', type=float, metavar='N',
            help='number of rnn layers',
        )
        parser.add_argument(
            '--m', type=float, metavar='N',
            help='read after source eos punishment',
        )
        parser.add_argument(
            '--rnn-dropout', type=float, metavar='N',
            help='dropout between rnn layers',
        )
        parser.add_argument(
            '--embedding-dropout', type=float, metavar='N',
            help='dropout after embedding layers',
        )

    @classmethod
    def build_model(cls, args, task):
        source_vocab = task.source_dictionary
        target_vocab = task.target_dictionary
        TESTING_EPISODE_MAX_TIME = args.max_testing_time

        approximator = DoubleHead(
            src_vocab_len=len(source_vocab.symbols),
            trg_vocab_len=len(target_vocab.symbols),
            src_embed_dim=args.src_embed_dim,
            trg_embed_dim=args.trg_embed_dim,
            rnn_hid_dim=args.rnn_hid_dim,
            rnn_dropout=args.rnn_dropout,
            embedding_dropout=args.embedding_dropout,
            rnn_num_layers=args.rnn_num_layers)

        mistranslation_loss = LabelSmoothedCrossEntropy(label_smoothing=args.smoothing)
        incremental_encoder = RLSTIncrementalEncoder(task.source_dictionary)
        incremental_decoder = RLSTIncrementalDecoder(task.target_dictionary,
                                                     approximator,
                                                     TESTING_EPISODE_MAX_TIME,
                                                     source_vocab.eos_index,
                                                     source_vocab.bos_index,
                                                     target_vocab.eos_index,
                                                     target_vocab.bos_index
                                                     )

        model = RLST(approximator, len(target_vocab), args.discount, args.m,
                     mistranslation_loss,
                     source_vocab.eos_index,
                     source_vocab.bos_index,
                     source_vocab.pad_index,
                     target_vocab.eos_index,
                     target_vocab.bos_index,
                     target_vocab.pad_index,
                     incremental_encoder, incremental_decoder)
        return model

    def forward(self, src, trg, epsilon, teacher_forcing, rtf_delta, rtf_prob):
        """
        :param src: Tensor of shape batch size x src seq length
        :param trg: Tensor of shape batch size x trg seq length
        :param epsilon: Probability of random action in epsilon greedy strategy
        :param teacher_forcing: Probability of output being ground truth at each time step
        :return: token_probs: Tensor of shape batch size x trg seq len x number of features e.g. target vocab length
        :return: Q_used: Tensor of shape batch size x time . Containes Q values of actions taken by agents
        :return: Q_target: Tensor of shape batch size x time. Containes best Q values in next time step w.r.t Q_used
        :return: logging_is_read: Bool tensor of shape batch size x time. Data about taken read actions
        :return: logging_is_write: Bool tensor of shape batch size x time. Data about taken write actions
        """

        device = src.device
        batch_size = src.size()[0]
        src_seq_len = src.size()[1]
        trg_seq_len = trg.size()[1]
        word_output = torch.full((batch_size, 1), self.TRG_NULL, device=device)
        rnn_state = self.approximator.init_state(batch_size, device)

        token_probs = torch.zeros((batch_size, trg_seq_len, self.trg_vocab_len), device=device)
        Q_used = torch.zeros((batch_size, src_seq_len + trg_seq_len - 1), device=device)
        Q_target = torch.zeros((batch_size, src_seq_len + trg_seq_len - 1), device=device)

        terminated_agents = torch.full((batch_size, 1), False, device=device)
        channels = 5
        lin_space = torch.linspace(0, channels-1, channels, dtype=torch.long, device=device)
        SRC_NULL_VECTOR = torch.full((1, channels), self.SRC_NULL, device=device)

        i = lin_space.repeat(batch_size, 1)  # input indices
        j = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=device)  # output indices

        input = torch.gather(src, 1, i)
        output, rnn_state = self.approximator(input, word_output, rnn_state)
        action = torch.max(output[:, :, -2:], 2)[1]

        logging_is_read = torch.full((batch_size, src_seq_len + trg_seq_len - 1), False, dtype=torch.bool, device=device)
        logging_is_write = torch.full((batch_size, src_seq_len + trg_seq_len - 1), False, dtype=torch.bool, device=device)

        for t in range(src_seq_len + trg_seq_len - 1):
            _, word_output = torch.max(output[:, :, :-2], dim=2)
            random_action_agents = torch.rand((batch_size, 1), device=device) < epsilon
            random_action = torch.randint(low=0, high=2, size=(batch_size, 1), device=device)
            action[random_action_agents] = random_action[random_action_agents]

            # forced_to_read_agents = torch.rand((batch_size, 1), device=device) < rtf_prob
            # should_read_agents = (i - rtf_delta) / src_seq_len < j / trg_seq_len
            # action[forced_to_read_agents * should_read_agents] = 0

            Q_used[:, t] = torch.gather(output[:, 0, -2:], 1, action).squeeze_(1)
            Q_used[terminated_agents.squeeze(1), t] = 0

            with torch.no_grad():
                reading_agents = ~terminated_agents * (action == 0)
                writing_agents = ~terminated_agents * (action == 1)

                logging_is_read[:, t] = reading_agents.squeeze(1)
                logging_is_write[:, t] = writing_agents.squeeze(1)

                just_terminated_agents = writing_agents * (torch.gather(trg, 1, j) == self.TRG_EOS)
                naughty_agents = reading_agents * (torch.gather(src, 1, i[:, -1:]) == self.SRC_EOS)
                i = i + ~naughty_agents * reading_agents
                old_j = j
                j = j + writing_agents * ~just_terminated_agents
                terminated_agents = terminated_agents + just_terminated_agents

                if random.random() < teacher_forcing:
                    word_output = torch.gather(trg, 1, old_j)
                word_output[reading_agents] = self.TRG_NULL

                reward = (-1) * self.mistranslation_loss(output[:, 0, :-2], torch.gather(trg, 1, old_j)[:, 0], reduce=False)[0]

            token_probs[writing_agents.squeeze(1), old_j[writing_agents], :] = output[writing_agents.squeeze(1), 0, :-2]

            input = torch.gather(src, 1, i)
            input[writing_agents.squeeze(1)] = SRC_NULL_VECTOR
            # input[naughty_agents, :-1] = self.SRC_EOS
            output, rnn_state = self.approximator(input, word_output, rnn_state)
            next_best_action_value, action = torch.max(output[:, :, -2:], 2)
            next_best_action_value = next_best_action_value.squeeze_(1)

            with torch.no_grad():
                Q_target[:, t] = reward + self.DISCOUNT * next_best_action_value
                Q_target[terminated_agents.squeeze(1), t] = 0
                Q_target[reading_agents.squeeze(1), t] = next_best_action_value[reading_agents.squeeze(1)]
                Q_target[naughty_agents.squeeze(1), t] = self.DISCOUNT * next_best_action_value[reading_agents.squeeze(1) * naughty_agents.squeeze(1)]
                Q_target[just_terminated_agents.squeeze(1), t] = reward[just_terminated_agents.squeeze(1)]
                Q_target[naughty_agents.squeeze(1), t] -= self.M

                if torch.all(terminated_agents):
                    return token_probs, Q_used, Q_target.detach_(), logging_is_read, logging_is_write

        return token_probs, Q_used, Q_target.detach_(), logging_is_read, logging_is_write


class RLSTIncrementalEncoder(FairseqEncoder):
    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(self, src_tokens, src_lengths):
        return src_tokens

    def reorder_encoder_out(self, encoder_out, new_order):
        return encoder_out.index_select(0, new_order)


class RLSTIncrementalDecoder(FairseqIncrementalDecoder):
    def __init__(self, dictionary, approximator, testing_episode_max_time,
                 src_eos_index, src_null_index, trg_eos_index, trg_null_index):

        super().__init__(dictionary)
        self.TESTING_EPISODE_MAX_TIME = testing_episode_max_time
        self.approximator = approximator
        self.SRC_EOS = src_eos_index
        self.SRC_NULL = src_null_index
        self.TRG_EOS = trg_eos_index
        self.TRG_NULL = trg_null_index

        self.trg_vocab_len = len(dictionary)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        src = encoder_out
        batch_size = src.size()[0]
        src_seq_len = src.size()[1]
        testing_episode_max_time = self.TESTING_EPISODE_MAX_TIME
        if not testing_episode_max_time:
            testing_episode_max_time = 3 * src_seq_len + 5
        device = src.device

        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            i = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=device)
            t = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=device)
            rnn_state = self.approximator.init_state(batch_size, device)
            input = src[:, :1]
            word_output = torch.full((batch_size, 1), self.TRG_NULL, device=device)
        else:
            input = torch.full((batch_size, 1), self.SRC_NULL, device=device)
            i = cached_state["i"]
            t = cached_state["t"]
            rnn_state = cached_state["rnn_state"]
            word_output = prev_output_tokens[:, -1:].clone().detach()

        frozen_agents = torch.full((batch_size, 1), False, device=device)
        token_probs = torch.zeros((batch_size, 1, self.trg_vocab_len), device=device)

        while True:
            output, new_rnn_state = self.approximator(input, word_output, copy.deepcopy(rnn_state))
            rnn_state = self.approximator.update_state(copy.deepcopy(rnn_state), new_rnn_state, frozen_agents.squeeze(1))

            failed_agents = t > testing_episode_max_time

            action = torch.max(output[:, :, -2:], 2)[1]
            reading_agents = (action == 0) * (~frozen_agents) * (~failed_agents)
            writing_agents = (action == 1) * (~frozen_agents) + failed_agents
            frozen_agents[writing_agents] = True

            token_probs[writing_agents.squeeze(1), 0, :] = output[writing_agents.squeeze(1), 0, :-2]

            naughty_agents = reading_agents * (torch.gather(src, 1, i) == self.SRC_EOS)
            i = i + ~naughty_agents * reading_agents
            i[i >= src_seq_len] = src_seq_len - 1

            input = torch.gather(src, 1, i)
            word_output[reading_agents] = self.TRG_NULL
            t[reading_agents + writing_agents] += 1

            if torch.all(frozen_agents):
                cached_state_new = {
                    "i": i,
                    "t": t,
                    "rnn_state": rnn_state
                }
                utils.set_incremental_state(self, incremental_state, 'cached_state', cached_state_new)
                return token_probs, None

    def reorder_incremental_state(self, incremental_state, new_order):
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        i = cached_state["i"].index_select(0, new_order)
        t = cached_state["t"].index_select(0, new_order)
        rnn_state = self.approximator.reorder_state(cached_state["rnn_state"], new_order)

        cached_state_new = {
                "i": i,
                "t": t,
                "rnn_state": rnn_state
            }

        utils.set_incremental_state(self, incremental_state, 'cached_state', cached_state_new)


@register_model_architecture('rlst', 'rlst')
def rlst(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.rnn_hid_dim = getattr(args, 'rnn_hid_dim', 512)
    args.rnn_num_layers = getattr(args, 'rnn_num_layers', 2)
    args.rnn_dropout = getattr(args, 'rnn_dropout', 0.0)
    args.embedding_dropout = getattr(args, 'embedding_dropout', 0.0)
    args.src_embed_dim = getattr(args, 'src_embed_dim', 256)
    args.trg_embed_dim = getattr(args, 'trg_embed_dim', 256)
    args.discount = getattr(args, 'discount', 0.9)
    args.m = getattr(args, 'm', 7.0)
    args.max_testing_time = getattr(args, 'max_testing_time', None)

