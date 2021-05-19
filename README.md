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
Prepare data:
First download and preprocess the data:
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
    --workers 20 --nwordssrc=12000 --nwordstrg=12000
```

Running RLST:
```shell
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en --arch rlst --rnn_hid_dim 768 --rnn_num_layers 2 --rnn_dropout 0.0 \
--src_embed_dim 256 --trg_embed_dim 256  --embedding_dropout 0.2 --max-tokens 4096 --max-epoch 50 \
--optimizer adam --clip-norm 10.0 --lr 1e-3  --weight-decay 1e-5 --left-pad-source --warmup-updates 100 \
--criterion rlst_criterion --teacher-forcing 0.5 --epsilon 0.15 --N 100000 --m 7.0 --discount 0.90 --eta-min 0.02 --eta-max 0.2 \
--eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --eval-bleu-args '{"beam": 1}' --maximize-best-checkpoint-metric \
--save-dir checkpoints/rlst
```
```shell
CUDA_VISIBLE_DEVICES=3 fairseq-train data-bin/iwslt14.tokenized.de-en --optimizer adam --lr 1e-3 --clip-norm 1.0  --max-tokens 4096 \
 --save-dir checkpoints/lstm/ --arch lstm_wiseman_iwslt_de_en --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe \
 --eval-bleu-args '{"beam": 1}' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```
