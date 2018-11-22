import theano
import argparse

import numpy
import pickle as pkl

from .nmt_uni import (build_model, build_sampler, gen_sample, load_params,
                 init_params, init_tparams, prepare_data)

from multiprocessing import Process, Queue


def translate_model(queue, rqueue, pid, model, options, k, normalize, kp, sigma):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost, f_cost = \
        build_model(tparams, options)
    inps = [x, x_mask, y, y_mask]

    f_log_probs = theano.function(inps, cost)

    # word index
    f_init, f_next = build_sampler(tparams, options, trng)

    def _translate(idx, seq):
        #sample = []
        #all_scores = []
        #all_c = []
        #score = 0
        #for kidx in xrange(kp):
        #    if kidx == 0:
        #        ss = -1.
        #    else:
        #        ss = sigma
            # sample given an input sequence and obtain scores
        sample, score = gen_sample(tparams, f_init, f_next,
                                       numpy.array(seq).reshape([len(seq), 1]),
                                       options, trng=trng, k=1, maxlen=30,
                                       stochastic=True, argmax=True)

            # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        print(idx, " Score is : ",  score)
        print(idx, " Sample is : ", sample)
            #sidx = numpy.argmin(score)
            #all_samples.append(sample)
            #all_scores.append(score)
            # all_c.append(c[0])

        #source_list = seq
        #x, x_mask, y, y_mask = prepare_data(source_list, sample, maxlen=None)
        #all_scores = f_log_probs(x, x_mask, y, y_mask)
        #if normalize:
        #    lengths = numpy.array([len(s) for s in sample])
        #    all_scores = all_scores / lengths

        #print idx, scores
        try:
            sidx = score.argmin()
        except:
            sidx = score
        return sample #, all_c[sidx]

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x = req[0], req[1]
        print(pid, '-', idx)
        seq = _translate(idx, x)
        print("Seq is : ", seq)
        rqueue.put((idx, seq))

    return


def main(model, dictionary, dictionary_target, source_file, saveto, k=5,
         normalize=False, n_process=5, chr_level=False,
         options_file=None, sigma=-1., kp=1):

    # load model model_options
    if options_file is not None:
        with open(options_file, 'rb') as f:
            options = pkl.load(f)
    else:
        with open('%s.pkl' % model, 'rb') as f:
            options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.items():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.items():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in range(n_process):
        processes[midx] = Process(
            target=translate_model,
            args=(queue, rqueue, midx, model, options, k, normalize, kp, sigma))
        processes[midx].start()

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname):
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = [word_dict[w] if w in word_dict else 1 for w in words]
                x = [ii if ii < options['n_words'] else 1 for ii in x]
                x += [0]
                #print "x is : ", x
                queue.put((idx, x))
        return idx+1

    def _finish_processes():
        for midx in range(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        c     = [None] * n_samples
        counter = 0
        for idx in range(n_samples):
            resp = rqueue.get(block=True)
            print("%%%%%%%%%%%%% Resp ", idx, " is :", resp)
            trans[resp[0]] = resp[1]
            try:
                c[resp[0]] = resp[0]
                print("----> ", idx, "===", resp[0])
            except:
                print("******************** idx : ", idx, "********************")
                counter += 1
                pass
            #if numpy.mod(idx, 10) == 0:
            print('Sample ', (idx+1), '/', n_samples, ' Done')
        print("Counter is : ", counter)
        return trans, c

    print('Translating ', source_file, '...')
    n_samples = _send_jobs(source_file)
    print("jobs sent")
    trans, c = _retrieve_jobs(n_samples)
    print("Jobs recieved")
    trans = _seqs2words(trans)
    print("Done with translation")
    _finish_processes()
    print("Start saving translation")
    with open(saveto, 'w') as f:
        print('\n'.join(trans), file=f)
        print('{}\n'.format(c), file=f)
    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-kp', type=int, default=1)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-c', action="store_true", default=False)
    parser.add_argument('-o', type=str, default=None)
    parser.add_argument('-s', type=float, default=-1.)
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.dictionary_target, args.source,
         args.saveto, k=args.k, normalize=args.n, n_process=args.p,
         chr_level=args.c, options_file=args.o, kp=args.kp, sigma=args.s)
