#!/usr/bin/env bash
#dataset=dev
model="/cs/natlang-expts/aalineja/dl4mt-simul-trans/models/pretrained_adadelta2/model_de-en.npz"
data="/cs/natlang-expts/aalineja/dl4mt-simul-trans/data" 
dict="${data}/all_de-en.de.tok.bpe.pkl"
dict_rev="${data}/all_de-en.en.tok.bpe.pkl"
source="${data}/newstest2016.de.tok.bpe"
saveto="./deengreedy.out"
reference="${data}/newstest2016.en.tok.bpe"

# pyenv local anaconda-2.4.0
THEANO_FLAGS="floatX=float32, device=cpu" python translate_uni.py -p 8 -k 1 $model $dict $dict_rev $source $saveto

# ./data/multi-bleu.perl $reference < $saveto
