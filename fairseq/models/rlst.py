import random
import torch
import torch.nn as nn

from fairseq.models import BaseFairseqModel, register_model, register_model_architecture


class Net(nn.Module):
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
        self.src_embedding = nn.Embedding(src_vocab_len, src_embed_dim, scale_grad_by_freq=True)
        self.trg_embedding = nn.Embedding(trg_vocab_len, trg_embed_dim, scale_grad_by_freq=True)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.rnn = nn.GRU(src_embed_dim + trg_embed_dim, rnn_hid_dim, num_layers=rnn_num_layers, bidirectional=False, dropout=rnn_dropout)
        self.output = nn.Linear(rnn_hid_dim, trg_vocab_len + 3)

    def forward(self, src, previous_output, rnn_state):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))
        rnn_input = torch.cat((src_embedded, trg_embedded), dim=2)
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        outputs = self.output(rnn_output)
        return outputs, rnn_state


@register_model('rlst')
class RLST(BaseFairseqModel):
    def __init__(self, net, device, testing_episode_max_time, trg_vocab_len, discount, m,
                 src_eos_index, src_null_index, src_pad_index, trg_eos_index, trg_null_index, trg_pad_index):
        super().__init__()
        self.net = net
        self.device = device
        self.testing_episode_max_time = testing_episode_max_time
        self.trg_vocab_len = trg_vocab_len
        self.DISCOUNT = discount
        self.M = m  # Read after eos punishment

        self.SRC_EOS = torch.tensor([src_eos_index], device=device)
        self.SRC_NULL = torch.tensor([src_null_index], device=device)
        self.SRC_PAD = torch.tensor([src_pad_index], device=device)
        self.TRG_EOS = torch.tensor([trg_eos_index], device=device)
        self.TRG_NULL = torch.tensor([trg_null_index], device=device)
        self.TRG_PAD = torch.tensor([trg_pad_index], device=device)

        self.mistranslation_loss_per_word = nn.CrossEntropyLoss(ignore_index=int(self.TRG_PAD), reduction='none')

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        parser.add_argument(
            '--rnn_hid_dim', type=int, metavar='N',
            help='dimensionality of the rnn hiddens state',
        )
        parser.add_argument(
            '--rnn_num_layers', type=int, metavar='N',
            help='number of rnn layers',
        )
        parser.add_argument(
            '--src_embed_dim', type=int, metavar='N',
            help='dimension of embedding layer for source tokens',
        )
        parser.add_argument(
            '--trg_embed_dim', type=int, metavar='N',
            help='dimension of embedding layer for target tokens',
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
            '--rnn_dropout', type=float, metavar='N',
            help='dropout between rnn layers',
        )
        parser.add_argument(
            '--embedding_dropout', type=float, metavar='N',
            help='dropout after embedding layers',
        )

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.
        source_vocab = task.source_dictionary
        target_vocab = task.target_dictionary
        TESTING_EPISODE_MAX_TIME = 350
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net = Net(
            src_vocab_len=len(source_vocab.symbols),
            trg_vocab_len=len(target_vocab.symbols),
            src_embed_dim=args.src_embed_dim,
            trg_embed_dim=args.trg_embed_dim,
            rnn_hid_dim=args.rnn_hid_dim,
            rnn_dropout=args.rnn_dropout,
            embedding_dropout=args.embedding_dropout,
            rnn_num_layers=args.rnn_num_layers).to(device)

        model = RLST(net, device, TESTING_EPISODE_MAX_TIME, len(target_vocab), args.discount, args.m,
                     source_vocab.eos_index,
                     source_vocab.bos_index,
                     source_vocab.pad_index,
                     target_vocab.eos_index,
                     target_vocab.bos_index,
                     target_vocab.pad_index).to(device)

        return model

    def forward(self, src, trg=None, epsilon=None, teacher_forcing=None):
        if self.training:
            return self._training_episode(src, trg, epsilon, teacher_forcing)
        return self._testing_episode(src)

    def _training_episode(self, src, trg, epsilon, teacher_forcing):
        device = self.device
        batch_size = src.size()[1]
        src_seq_len = src.size()[0]
        trg_seq_len = trg.size()[0]
        word_output = torch.full((1, batch_size), int(self.TRG_NULL), device=device)
        rnn_state = torch.zeros((self.net.rnn_num_layers, batch_size, self.net.rnn_hid_dim), device=device)

        word_outputs = torch.zeros((trg_seq_len, batch_size, self.trg_vocab_len), device=device)
        Q_used = torch.zeros((src_seq_len + trg_seq_len, batch_size), device=device)
        Q_target = torch.zeros((src_seq_len + trg_seq_len, batch_size), device=device)

        writing_agents = torch.full((1, batch_size), False, device=device, requires_grad=False)
        naughty_agents = torch.full((1, batch_size,), False, device=device, requires_grad=False)  # Want more input after input eos
        terminated_agents = torch.full((1, batch_size,), False, device=device, requires_grad=False)

        i = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device, requires_grad=False)  # input indices
        j = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device, requires_grad=False)  # output indices
        t = 0  # time
        actions_count = torch.zeros(3, dtype=torch.long, device=device, requires_grad=False)

        while True:
            input = torch.gather(src, 0, i)
            input[writing_agents] = self.SRC_NULL
            input[naughty_agents] = self.SRC_EOS
            output, rnn_state = self.net(input, word_output, rnn_state)
            _, word_output = torch.max(output[:, :, :-3], dim=2)
            action = torch.max(output[:, :, -3:], 2)[1]

            random_action_agents = torch.rand((1, batch_size), device=device) < epsilon
            random_action = torch.randint(low=0, high=3, size=(1, batch_size), device=device)
            action[random_action_agents] = random_action[random_action_agents]

            Q_used[t, :] = torch.gather(output[0, :, -3:], 1, action.T).squeeze_(1)
            Q_used[t, terminated_agents.squeeze(0)] = 0

            with torch.no_grad():
                reading_agents = ~terminated_agents * (action == 0)
                writing_agents = ~terminated_agents * (action == 1)
                bothing_agents = ~terminated_agents * (action == 2)

                actions_count[0] += reading_agents.sum()
                actions_count[1] += writing_agents.sum()
                actions_count[2] += bothing_agents.sum()

            agents_outputting = writing_agents + bothing_agents
            word_outputs[j[agents_outputting], agents_outputting.squeeze(0), :] = output[0, agents_outputting.squeeze(0), :-3]

            just_terminated_agents = agents_outputting * (torch.gather(trg, 0, j) == self.TRG_EOS).squeeze_(0)
            naughty_agents = (reading_agents + bothing_agents) * (torch.gather(src, 0, i) == self.SRC_EOS).squeeze_(0)
            i = i + ~naughty_agents * (reading_agents + bothing_agents)
            old_j = j
            j = j + agents_outputting

            terminated_agents = terminated_agents + just_terminated_agents

            i[i >= src_seq_len] = src_seq_len - 1
            j[j >= trg_seq_len] = trg_seq_len - 1

            if random.random() < teacher_forcing:
                word_output = torch.gather(trg, 0, old_j)
            word_output[reading_agents] = self.TRG_NULL

            with torch.no_grad():
                _input = torch.gather(src, 0, i)
                _input[writing_agents] = self.SRC_NULL
                _input[naughty_agents] = self.SRC_EOS
                _output, _ = self.net(_input, word_output, rnn_state)
                next_best_action_value, _ = torch.max(_output[:, :, -3:], 2)

                reward = (-1) * self.mistranslation_loss_per_word(output[0, :, :-3], torch.gather(trg, 0, old_j)[0, :]).unsqueeze(0)
                Q_target[t, :] = reward + self.DISCOUNT * next_best_action_value
                Q_target[t, terminated_agents.squeeze(0)] = 0
                Q_target[t, reading_agents.squeeze(0)] = next_best_action_value[reading_agents]
                Q_target[t, (reading_agents * naughty_agents).squeeze(0)] = self.DISCOUNT * next_best_action_value[reading_agents * naughty_agents]
                Q_target[t, just_terminated_agents.squeeze(0)] = reward[just_terminated_agents]
                Q_target[t, naughty_agents.squeeze(0)] -= self.M

                if terminated_agents.all() or t >= src_seq_len + trg_seq_len - 1:
                    return word_outputs, Q_used, Q_target, actions_count.unsqueeze_(dim=1)
            t += 1

    def _testing_episode(self, src):
        device = self.device
        batch_size = src.size()[1]
        src_seq_len = src.size()[0]
        word_output = torch.full((1, batch_size), int(self.TRG_NULL), device=device)
        rnn_state = torch.zeros((self.net.rnn_num_layers, batch_size, self.net.rnn_hid_dim), device=device)

        word_outputs = torch.zeros((self.testing_episode_max_time, batch_size, self.trg_vocab_len), device=device)

        writing_agents = torch.full((1, batch_size), False, device=device, requires_grad=False)
        naughty_agents = torch.full((1, batch_size,), False, device=device, requires_grad=False)  # Want more input after input eos
        after_eos_agents = torch.full((1, batch_size,), False, device=device, requires_grad=False)  # Already outputted EOS

        i = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device, requires_grad=False)  # input indices
        j = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device, requires_grad=False)  # output indices
        t = 0  # time
        actions_count = torch.zeros(3, dtype=torch.long, device=device, requires_grad=False)

        while True:
            input = torch.gather(src, 0, i)
            input[writing_agents] = self.SRC_NULL
            input[naughty_agents] = self.SRC_EOS
            output, rnn_state = self.net(input, word_output, rnn_state)
            _, word_output = torch.max(output[:, :, :-3], dim=2)
            action = torch.max(output[:, :, -3:], 2)[1]

            reading_agents = (action == 0)
            writing_agents = (action == 1)
            bothing_agents = (action == 2)

            actions_count[0] += (~after_eos_agents * reading_agents).sum()
            actions_count[1] += (~after_eos_agents * writing_agents).sum()
            actions_count[2] += (~after_eos_agents * bothing_agents).sum()

            agents_outputting = writing_agents + bothing_agents
            word_outputs[j[agents_outputting], agents_outputting.squeeze(0), :] = output[0, agents_outputting.squeeze(0), :-3]

            after_eos_agents += (word_output == self.TRG_EOS)
            naughty_agents = (reading_agents + bothing_agents) * (torch.gather(src, 0, i) == self.SRC_EOS).squeeze_(0)
            i = i + ~naughty_agents * (reading_agents + bothing_agents)
            j = j + agents_outputting

            i[i >= src_seq_len] = src_seq_len - 1
            word_output[reading_agents] = self.TRG_NULL

            if t >= self.testing_episode_max_time - 1:
                return word_outputs, None, None, actions_count.unsqueeze_(dim=1)
            t += 1


@register_model_architecture('rlst', 'rlst')
def rlst(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.rnn_hid_dim = getattr(args, 'rnn_hid_dim', 256)
    args.rnn_num_layers = getattr(args, 'rnn_num_layers', 1)
    args.rnn_dropout = getattr(args, 'rnn_dropout', 0.20)
    args.embedding_dropout = getattr(args, 'embedding_dropout', 0.0)
    args.src_embed_dim = getattr(args, 'src_embed_dim', 128)
    args.trg_embed_dim = getattr(args, 'trg_embed_dim', 128)
    args.discount = getattr(args, 'discount', 0.90)
    args.m = getattr(args, 'm', 10.0)

