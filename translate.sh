#!/usr/bin/env bash
#dataset=dev
model="/cs/natlang-expts/aalineja/dl4mt-simul-trans-prediction/models/my-pretrained-uni-en-de/wmt15_bpe2k_uni_en-de.npz"
data="/cs/natlang-expts/aalineja/dl4mt-simul-trans-prediction/data"
dict="${data}/all_de-en.en.tok.bpe.pkl"
dict_rev="${data}/all_de-en.de.tok.bpe.pkl"
source="${data}/newstest2013.en.tok.bpe"
saveto="./my_pretrained-uni-en-de.out"
reference="${data}/newstest2013.de.tok.bpe"


# pyenv local anaconda-2.4.0

THEANO_FLAGS="floatX=float32, device=cuda" python translate.py --p 4 --k 5 -model $model -dictionary $dict -dictionary_target $dict_rev -source $source -saveto $saveto

./data/multi-bleu.perl $reference < $saveto
