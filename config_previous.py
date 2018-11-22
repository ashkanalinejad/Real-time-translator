from collections import OrderedDict

# data_home  = '/home/thoma/scratch/un16/'
# model_home = '/home/thoma/scratch/simul/'
# data_home  = '/mnt/scratch/un16/'
# model_home = '/mnt/scratch/simul/'

data_home   = '/cs/natlang-expts/aalineja/dl4mt-simul-trans/data/'
model_home  = '/cs/natlang-expts/aalineja/dl4mt-simul-trans/models/'


def pretrain_config():

    """Configuration for pretraining underlining NMT model."""

    config = dict()

    # training set (source, target)
    config['datasets'] = [data_home + 'all_de-en.de.tok.bpe',
                          data_home + 'all_de-en.en.tok.bpe']

    # validation set (source, target)
    config['valid_datasets'] = [data_home + 'newstest2011.de.tok.bpe',
                                data_home + 'newstest2011.en.tok.bpe']

    # vocabulary (source, target)
    config['dictionaries']   = [data_home + 'all_de-en.de.tok.bpe.pkl',
                                data_home + 'all_de-en.en.tok.bpe.pkl']

    # save the model to
    config['saveto']      = model_home + 'pretrained_adadelta_birnn/model_de-en.npz'
    config['reload_']     = True

    # model details
    config['dim_word']    = 1028
    config['dim']         = 1028
    config['n_words']     = 20000
    config['n_words_src'] = 20000

    # learning details
    config['decay_c']     = 0
    config['clip_c']      = 1.
    config['use_dropout'] = False
    config['lrate']       = 0.0001
    config['optimizer']   = 'adadelta'
    config['patience']    = 1000
    config['maxlen']      = 50
    config['batch_size']  = 32
    config['valid_batch_size'] =  32
    config['validFreq']   = 5000
    config['dispFreq']    = 20
    config['saveFreq']    = 5000
    config['sampleFreq']  = 500
    config['birnn']  = True

    return config


def rl_config():
    """Configuration for training the agent using REINFORCE algorithm."""

    config = OrderedDict()  # general configuration

    # work-space
    config['workspace'] = model_home

    # training set (source, target); or leave it None, agent will use the same corpus saved in the model
    config['datasets'] = [data_home + 'all_de-en.de.tok.bpe',
                          data_home + 'all_de-en.en.tok.bpe']

    # validation set (source, target); or leave it None, agent will use the same corpus saved in the model
    config['valid_datasets'] = [data_home + 'newstest2011.de.tok',
                                data_home + 'newstest2011.en.tok']

    # vocabulary (source, target); or leave it None, agent will use the same dictionary saved in the model
    config['dictionaries']   = [data_home + 'all_de-en.de.tok.bpe.pkl',
                                data_home + 'all_de-en.en.tok.bpe.pkl']

    # pretrained model
    config['model']  = model_home + '.pretrained/model_de-en.npz'
    config['option'] = model_home + '.pretrained/model_de-en.npz.pkl'

    # critical training parameters.
    config['sample']    = 10
    config['batchsize'] = 10
    config['rl_maxlen'] = 40
    config['target_ap'] = 0.8   # 0.75  # target delay if using AP as reward.
    config['target_cw'] = 8     # if cw > 0 use cw mode

    # under-construction
    config['predict']    = False

    # learning rate
    config['lr_policy'] = 0.0002
    config['lr_model']  = 0.00002

    # policy parameters
    config['prop']      = 0.5   # leave it default
    config['recurrent'] = True  # use a recurrent agent
    config['layernorm'] = False # layer normalalization for the GRU agent.
    config['updater']   = 'REINFORCE'  # 'TRPO' not work well.
    config['act_mask']  = True  # leave it default

    # old model parameters (maybe useless, leave them default)
    config['step']     = 1
    config['peek']     = 1
    config['s0']       = 1
    config['gamma']    = 1
    config['Rtype']    = 10
    config['maxsrc']   = 10
    config['pre']      = False
    config['coverage'] = False
    config['upper']    = False

    config['finetune'] = True
    config['train_gt'] = False   # when training with GT, fix the random agent??
    config['full_att'] = True
    config['predict']  = False

    return config

