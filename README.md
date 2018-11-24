# simultaneous-nmt
Simultaneous neural machine translation that uses prediction on the source side.
This code is a Theano implementation of EMNLP18 paper [Prediction Improves Simultaneous Neural Machine Translation](http://aclweb.org/anthology/D18-1337). Our implementations are based on [dl4mt-simul-trans](https://github.com/nyu-dl/dl4mt-simul-trans) repository developed by Gu et al.


# Dataset

- We have used WMT'15 corpora as our dataset for pretraining and training our agent parameters.
- Newstest 2013 for validation and testset.
- The data should be tokenized and Byte Pair Encoded.

# Pretraining
The first step of training the model starts with pretraining Environment. The parameters of the uni-directional LSTM can be changed using the function `pretrain_config()` in `config.py`. After setting up configuration, pretrain can be started:
```bash
$ export THEANO_FLAGS=device=gpu,floatX=float32
$ python pretrain_uni.py
```

# Training the Agent
Like pretraining the settings of the Model can be configured in `config.py`. Then training the Agent can be started using `sh run_train.sh`.


