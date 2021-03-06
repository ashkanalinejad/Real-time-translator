
from collections import OrderedDict
import pickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from .utils.data_iterator import TextIterator, WindowIterator

from .lstm import *

def predict(f_pred, words):

	if len(words) >= 4:
		words = words[-3:]
	words = numpy.expand_dims(words, axis=1)
	#print(words)        

	x = numpy.zeros((3, 1)).astype('int64')
	mask = numpy.ones((3, 1)).astype('float32')

	x = words
	#print(x)
	#print(mask)
	preds = f_pred(x, mask)
	return preds

if __name__ == '__main__':

    dictionaries = ['/Users/alinejad/Desktop/SFU/SNMT-Prediction/dl4mt-simul-trans/data/all_de-en.en.tok.bpe.pkl']

    worddicts = [None]
    worddicts_r = [None]
    with open(dictionaries[0], 'rb') as f:
        worddicts = pkl.load(f)
    worddicts_r = dict()
    for kk, vv in list(worddicts.items()):
        worddicts_r[vv] = kk

    model = sys.argv[1]

    words = [worddicts[sys.argv[i]] for i in range(2, len(sys.argv))]
    print(words)

    model_options = OrderedDict()
    model_options['dim_proj'] = 1024
    model_options['patience'] = 10
    model_options['max_epochs'] = 5000
    model_options['dispFreq'] = 10
    model_options['decay_c'] = 0.
    model_options['lrate'] = 0.00002
    model_options['n_words'] = 20000
    model_options['encoder'] = 'lstm'
    model_options['validFreq'] = 1000
    model_options['saveFreq'] = 1000
    model_options['maxlen'] = 100
    model_options['batch_size'] = 64
    model_options['valid_batch_size'] = 64
    model_options['noise_std'] = 0.
    model_options['use_dropout'] = True
    model_options['reload_model'] = None
    model_options['test_size'] = -1
    model_options['ydim'] = 20000

    params = OrderedDict()
    params = init_params(model_options)
    load_params(model, params)

    tparams = init_tparams(params)
    model_options['dim_proj'] = 1024

    (use_noise, x, mask,
     y, y_mask, f_pred_prob, f_pred, log_probs, f_u_shape) = build_model(tparams, model_options)

    use_noise.set_value(0.)
    pp = predict(f_pred, words)
    print(worddicts_r[pp[0]])

