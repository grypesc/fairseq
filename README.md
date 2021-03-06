<p align="center">
  <img src="docs/fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://github.com/pytorch/fairseq/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
</p>

--------------------------------------------------------------------------------
# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install this version of fairseq** go to repository root level and run:

``` bash
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

In order to repeat results from the Reinforcement Learning for on-line Sequence Transformation paper:  
Download and preprocess the data:
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en

fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```
This preprocesses data for German-English translation. To train and evaluate English-German translation models, switch `--source-lang` and `--target-lang` values in `fairseq-preprocess` command. 

Training RLST on WMT15:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train data-bin/wmt15_de_en --arch rlst --criterion rlst_criterion --no-epoch-checkpoints \
--eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --eval-bleu-args '{"beam": 1}' --maximize-best-checkpoint-metric \
--rnn-hid-dim 512 --rnn-num-layers 2 --rnn-dropout 0.0 --src-embed-dim 256 --trg-embed-dim 256  --embedding-dropout 0.0 \
--max-tokens 4096 --max-epoch 100 --optimizer adam --clip-norm 10.0 --lr 1e-3  --weight-decay 1e-5 --left-pad-source --rho 0.99 \
--epsilon-min 0.2 --epsilon-max 0.2 --rtf-delta 1.0 --N 200000 --m 7.0 --discount 0.90 --eta-min 0.02 --eta-max 0.2 \
--save-dir checkpoints/rlst
```
Training the transformer model:
```shell
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 10.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \ 
--dropout 0.3 --weight-decay 0.0001 --max-tokens 4096 --eval-bleu --eval-bleu-args '{"beam": 1}' --eval-bleu-detok moses \ 
--eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --no-epoch-checkpoints \ 
--max-epoch 100 --save-dir checkpoints/transformer
```
Training the encoder decoder LSTM model:
```shell
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en --optimizer adam --lr 1e-3 --clip-norm 10.0  --max-tokens 4096 \
--save-dir checkpoints/lstm/ --arch lstm_wiseman_iwslt_de_en --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe \
--eval-bleu-args '{"beam": 1}' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --no-epoch-checkpoints \ 
--max-epoch 100
```

Trained LSTM and transformer models can be evaluated on the test set using the `fairseq-generate` command:
```shell
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt14.tokenized.de-en/ --path <path to model checkpoints directory>/checkpoint_best.pt --beam 1 --remove-bpe --quiet
```
To test RLST you also need to provide ```--left-pad-source``` flag:
```shell
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt14.tokenized.de-en/ --path <path to model checkpoints directory>/checkpoint_best.pt --beam 1 --remove-bpe --quiet --left-pad-source
```