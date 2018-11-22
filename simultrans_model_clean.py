"""
Simultaneous Machine Translateion with prediction.

"""
import numpy as np
from nmt_uni import *
from reward  import return_reward
from prediction.sample import predict

import time
import sys

timer = time.time


# utility functions
def _seqs2words(caps, idict, actions=None, target=0):
    capsw = []
    colors = ['cyan', 'green', 'yellow', 'red', 'magenta']

    for kk, cc in enumerate(caps):
        ww   = []
        pos  = 0
        iss  = 0
        flag = False

        for w in cc:
            if w == 0:
                break

            word = idict[w]
            if actions is not None:
                while True:
                    if iss == len(actions[kk]):
                        word = word # clr(word, 'white')
                        break

                    if actions[kk][iss] == target:
                        word = word # clr(word, colors[pos % len(colors)])
                        iss += 1
                        flag = True
                        break
                    else:
                        if flag:
                            pos  += 1
                            flag = False
                        iss += 1

            ww.append(word)

        capsw.append(' '.join(ww))
    return capsw


def _bpe2words(capsw):
    capw   = []
    for cc in capsw:
        words = cc.replace('@@ ', '')


        capw += [words]
    return capw


def _action2delay(src, actions):
    delays = []
    X = len(src)
    for act in actions:
        A = numpy.array(act, dtype='float32')
        Y = numpy.sum(act)
        S = numpy.sum(numpy.cumsum(1 - A) * A)

        assert (X > 0) and (Y > 0), 'avoid NAN {}, {}'.format(X, Y)

        tau = S / (Y * X)
        delays.append([tau, X, Y, S])

    return delays


# padding for computing policy gradient
def _padding(arrays, shape, dtype='float32', return_mask=False, sidx=0):
    B = numpy.zeros(shape, dtype=dtype)

    if return_mask:
        M = numpy.zeros((shape[0], shape[1]), dtype='float32')

    for it, arr in enumerate(arrays):
        arr = numpy.asarray(arr, dtype=dtype)
        # print arr.shape

        steps = arr.shape[0]
        # print 'arr is : ', str(arr.shape)
        if arr.ndim < 2:
            B[sidx: steps + sidx, it] = arr
        else:
            steps2 = arr.shape[1]
            B[sidx: steps + sidx, it, : steps2] = arr

        if return_mask:
            M[sidx: steps + sidx, it] = 1.

    if return_mask:
        return B, M
    return B


# ==============================================================
# Simultaneous Translation in Batch-mode
# ==============================================================
def simultaneous_decoding(funcs, agent, options,
                          srcs, trgs, f_pred, idict=None,
                          t_idict=None, samples=None, greedy=False,
                          train=False, forget_left=True):
    start_time = time.clock()
    # --- unzip functions
    f_sim_ctx     = funcs[0]
    f_sim_init    = funcs[1]
    f_sim_next    = funcs[2]
    f_cost        = funcs[3]

    if options['finetune']:
        ff_init   = funcs[4]
        ff_cost   = funcs[5]
        ff_update = funcs[6]

    n_sentences   = len(srcs)
    n_out         = 3 if options['predict'] else 2
    n_samples     = options['sample'] if not samples else samples
    sidx          = options['s0']
    maxlen        = options['rl_maxlen']

    _probs        = numpy.zeros((n_out, ))
    _total        = 0

    live_k        = n_samples * n_sentences
    live_all      = live_k

    # ============================================================================ #
    # Initialization Before Generating Trajectories
    # ============================================================================ #

    # Critical! add the <eos> ------------------
    srcs = [src + [0] for src in srcs]
    trgs = [trg + [0] for trg in trgs]

    src_max   = max([len(src) for src in srcs])
    if src_max < sidx:
        sidx  = src_max
    trg_max   = max([len(trg) for trg in trgs])

    x0, y0, ctx0, z0, seq_info0 = [], [], [], [], []

    # data initialization
    for id, (src, trg) in enumerate(zip(srcs, trgs)):

        _x          = numpy.array(src, dtype='int64')[:, None]
        _y          = numpy.array(trg, dtype='int64')[:, None]
        _, _ctx0, _ = f_sim_ctx(_x)
        _z0         = f_sim_init(_ctx0[:sidx, :])

        x0.append(_x[:, 0])
        y0.append(_y[:, 0])
        ctx0.append(_ctx0[:, 0, :])
        z0.append(_z0.flatten())
        seq_info0.append([id, len(src), 0])  # word id / source length / correctness

    # pad the results
    x0, x_mask = _padding(x0, (src_max, n_sentences), dtype='int64', return_mask=True)
    y0, y_mask = _padding(y0, (trg_max, n_sentences), dtype='int64', return_mask=True)
    ctx       = _padding(ctx0, (src_max, n_sentences, ctx0[0].shape[-1]))
    z0        = numpy.asarray(z0)

    mask      = numpy.asarray([1.] * sidx + [0.] * (src_max - sidx), dtype='float32')[:, None]
    one       = numpy.asarray([1.] * src_max, dtype='float32')[:, None]

    # hidden states
    hidden0   = agent.init_hidden()

    # if we have multiple samples for one input sentence
    mask      = numpy.tile(mask, [1, live_k])
    z         = numpy.tile(z0,   [n_samples, 1])
    ctx       = numpy.tile(ctx,  [1, n_samples, 1])
    x         = numpy.tile(x0,   [1, n_samples])
    y         = numpy.tile(y0,   [1, n_samples])
    hidden    = numpy.tile(hidden0, [live_k, 1])

    seq_info  = []
    for _ in range(int(live_k / n_sentences)):
        seq_info += copy.deepcopy(seq_info0)

    # =========================================================================== #
    # PIPE for message passing
    # =========================================================================== #
    pipe   = OrderedDict()
    h_pipe = OrderedDict()

    # initialize pipes
    for key in ['sample', 'score', 'action',
                'obs', 'attentions', 'old_attend',
                'coverage', 'forgotten',
                'seq_info',
                'cmask', 'source', 'i_mask']:
        pipe[key] = []

    # initialize h-pipe
    for key in ['sample', 'obs', 'attentions',
                'hidden', 'old_attend', 'cmask']:
        h_pipe[key] = [[] for _ in range(live_k)]

    h_pipe['score']     = numpy.zeros(live_k).astype('float32')
    h_pipe['action']    = [[0]  * sidx for _ in range(live_k)]
    h_pipe['forgotten'] = [[-1] * sidx for _ in range(live_k)]
    h_pipe['coverage']  = numpy.zeros((live_k, ctx.shape[0])).astype('float32')

    h_pipe['mask']      = mask   # Backup mask
    h_pipe['maskp']     = mask   # Main Mask
    h_pipe['ctx']       = ctx    # Backup contexts
    h_pipe['ctxp']      = ctx    # Main contexts
    h_pipe['source']    = x      # source words
    h_pipe['seq_info']  = seq_info
    h_pipe['heads']     = numpy.asarray([[sidx, 0, 0]] * live_k)  # W C F
    h_pipe['headsp']    = h_pipe['heads']
    h_pipe['i_mask']    = mask

    h_pipe['prev_w']    = -1 * numpy.ones((live_k, )).astype('int64')
    h_pipe['prev_z']    = z
    h_pipe['prev_hid']  = hidden
    h_pipe['prev_wsrc'] = numpy.zeros((live_k, 3)).astype('int64') # used for prediction
    
    # these are inputs that needs to be updated
    step       = 0 
    timeread = []
    timewrite = []
    timepredict0 = []
    timepredict1 = []
    #
    # =======================================================================
    # ROLLOUT: Iteration until all the samples over.
    # Action space:
    # 0: READ,
    # 1: WRITE,
    # 2: PREDICT (optional)
    # =======================================================================
    while live_k > 0:
        step += 1

        # ------------------------------------------------------------------
        # Run one-step translation
        # ------------------------------------------------------------------
        inps  = [h_pipe[v] for v in ['prev_w', 'ctxp', 'mask', 'prev_z']]
        next_p, next_w, next_z, next_o, next_a, cur_emb = f_sim_next(*inps)

        if options['full_att']:
            old_mask = numpy.tile(one,  [1, live_k])
            inps2    = inps
            inps2[2] = old_mask
            _, _, _, _, next_fa, _ = f_sim_next(*inps2)

        # -------------------------------------------------------------------
        # obtain the candidate and the accumulated score.
        if (not greedy) and (options['finetune']):
            if options['train_gt']:  # ground-truth words
                _cand = [y0[h_pipe['headsp'][idx, 1], h_pipe['seq_info'][idx][0]]
                        for idx in range(live_k)]
            else:
                _cand = next_w       # sampling
        else:
            _cand    = next_p.argmax(axis=-1)  # live_k, candidate words

        _score       = numpy.log(next_p[list(range(live_k)), _cand] + 1e-8)
        # -------------------------------------------------------------------

        # new place-holders for temporal results: new-hyp-message
        n_pipe = OrderedDict()

        for key in ['sample', 'score', 'headsp', 'heads', 'attentions',
                    'old_attend', 'coverage', 'maskp', 'mask', 'prev_wsrc',
                    'ctxp', 'ctx', 'seq_info', 'cmask', 'obs',
                    'prev_z', 'source', 'i_mask', 'action', 'forgotten']:

            n_pipe[key] = copy.copy(h_pipe[key])
        n_pipe['hidden'] = []

        # first_ctx = n_pipe['ctx']
        
        cov    = n_pipe['coverage'] * n_pipe['mask'].T + next_a  # clean that has been forgotten
        cid    = cov.argmax(axis=-1)

        # ------------------------------------------------------------------
        # Run one-step agent action.
        # ------------------------------------------------------------------
        _actions, _aprop, _hidden, _z = agent.action(next_o, h_pipe['prev_hid'])  # input the current observation

        if greedy:
            _actions = _aprop.argmax(-1)

        _total += _aprop.shape[0]
        _probs += _aprop.sum(axis=0)

        # ------------------------------------------------------------------
        # Evaluate the action
        # ------------------------------------------------------------------
        for idx, wi in enumerate(_cand):

            # action
            a      = _actions[idx]
            c_mask = n_pipe['mask'][:, idx]

            if options.get('upper', False):
                a = 0  # testing upper bound: only wait
            if greedy and (n_pipe['heads'][idx, 0] >= n_pipe['seq_info'][idx][1]):
                a = 1  # in greedy mode. must end.
            if greedy and (n_pipe['heads'][idx, 2] >= n_pipe['heads'][idx, 0]):
                a = 1  # in greedy mode. must end.
            if ((n_pipe['action'][idx][-1] == 2) and (a == 0)):
                a = 2  # READ after PREDICT
            if (n_pipe['headsp'][idx, 0] >= n_pipe['seq_info'][idx][1]) and (a == 2):
                a = 1  # If we've reached to the end of source sentence, predict shouldn't be an option.
            if (n_pipe['action'][idx][-1] == 1) and (a == 2):
                a = 0  # PREDICT after  WRITE

            # must read the whole sentence
            # pass             

            # message appending
            n_pipe['obs'][idx].append(next_o[idx])
            n_pipe['action'][idx].append(a)
            n_pipe['hidden'].append(_hidden[idx])

            # change the behavior of NMT model
            # n_pipe['seq_info'] = length of each sample. E.g. 38.

            # For READ
            if a == 0:
                startr = time.clock()
                # read-head move on one step
                if n_pipe['heads'][idx, 0] < n_pipe['seq_info'][idx][1]:  # Shoulb be heads. Not headsp
                    n_pipe['mask'][n_pipe['heads'][idx, 0], idx] = 1
                    n_pipe['maskp'] = n_pipe['mask']
                    n_pipe['heads'][idx, 0] += 1
                    n_pipe['headsp'][idx, 0] = n_pipe['heads'][idx, 0]

                    # For prediction:
                    _x_ = np.array([i for i in np.array(x[:, idx]) if i.any()])
                    _x_ = np.append(_x_, [0]*(n_pipe['seq_info'][idx][1]-len(_x_)))
                    #if _x_.shape == 2:
                    #    print(_x_)
                    #    _x_.append([0])
                    if n_pipe['heads'][idx, 0] == len(_x_):
                    	pass
	                    # print(_x_)
	                    # print(_x_[n_pipe['heads'][idx, 0]-2:n_pipe['heads'][idx, 0]+1])
	                    # print(n_pipe['heads'][idx, 0]-2, n_pipe['heads'][idx, 0]+1)
                    else:
                        n_pipe['prev_wsrc'][idx, :] = _x_[n_pipe['heads'][idx, 0]-2:n_pipe['heads'][idx, 0]+1]

                n_pipe['forgotten'][idx].append(-1)

                # if the first word is still waiting for decoding
                if numpy.sum(n_pipe['action'][idx]) == 0:
                    temp_sidx = n_pipe['heads'][idx, 0]
                    _ctx0     = ctx0[n_pipe['seq_info'][idx][0]][:, None, :]
                    _z0       = f_sim_init(_ctx0[:temp_sidx])  # initializer
                    n_pipe['prev_z'][idx] = _z0
                    n_pipe['i_mask'][temp_sidx-1, idx] = 1
                endr = time.clock()
                timeread.append(endr-startr)

            # For WRITE:
            elif a == 1:
                startw = time.clock()
                n_pipe['sample'][idx].append(wi)
                n_pipe['cmask'][idx].append(c_mask)
                n_pipe['score'][idx] += _score[idx]
                n_pipe['attentions'][idx].append(next_a[idx])
                n_pipe['forgotten'][idx].append(-1)

                if options['full_att']:
                    n_pipe['old_attend'][idx].append(next_fa[idx])

                n_pipe['prev_z'][idx]    = next_z[idx]  # update the decoder's hidden states
                n_pipe['heads'][idx, 1] += 1
                n_pipe['headsp'][idx, 1] = n_pipe['heads'][idx, 1]
                n_pipe['coverage'][idx]  = cov[idx]
                endw = time.clock()
                timewrite.append(endw-startw)

            # For PREDICT:
            elif a == 2:
                startp0 = time.clock()
                ctxp = np.zeros((src_max, 1028))            
                _x_ = np.array([i for i in np.array(x[:, idx]) if i.any()])
                _x_ = np.append(_x_, [0]*(n_pipe['seq_info'][idx][1]-len(_x_)))
                
                # print(n_pipe['prev_wsrc'].shape)
                # print(n_pipe['prev_wsrc'][idx, :])
                preds = predict(f_pred, n_pipe['prev_wsrc'][idx, :])
                # currentp = modelp.sample(session, wordsp, vocabp, 1, str(PW0), 2, 1, 4)
                startp1 = time.clock()
                _x_[n_pipe['headsp'][idx, 0]] = int(preds)
                _x_ = np.array(np.expand_dims(_x_, axis=1), dtype='int64')
                _, ctxp0, _ = f_sim_ctx(_x_)
                ctxp0 = np.array(ctxp0[:, 0, :], dtype="float32")
                ctxp[:ctxp0.shape[0], :] = ctxp0           # Padding with zeros
                endp1 = time.clock()

                n_pipe['ctxp'][:, idx, :] = ctxp # Double check dimensions
                n_pipe['maskp'][n_pipe['headsp'][idx, 0], idx] = 1
                n_pipe['headsp'][idx, 0] += 1
                n_pipe['prev_wsrc'][idx, :-1] = n_pipe['prev_wsrc'][idx, 1:]
                n_pipe['prev_wsrc'][idx, -1]  = int(preds)
                endp0 = time.clock()
                timepredict1.append(endp1-startp1)
                timepredict0.append(endp0-startp0)
            
            else:
                raise NotImplementedError

        # ------------------------------------------------------------------
        # Check the correctness!
        # ------------------------------------------------------------------
        for idx in range(live_k):
            if n_pipe['heads'][idx, 0] >= n_pipe['seq_info'][idx][1]:
                # the read head already reached the end.
                n_pipe['seq_info'][idx][2] = -1

        # ------------------------------------------------------------------
        # Collect the trajectories
        # ------------------------------------------------------------------
        #  kill the completed samples, so I need to build new hyp-messages
        h_pipe = OrderedDict()

        for key in ['sample', 'score', 'headsp', 'heads', 'prev_wsrc',
                    'maskp', 'mask', 'prev_z', 'coverage',
                    'forgotten', 'action', 'obs', 'ctxp',
                    'ctx', 'seq_info', 'attentions', 'hidden',
                    'old_attend', 'cmask', 'source', 'i_mask']:
            h_pipe[key] = []
 
        for idx in range(len(n_pipe['sample'])):

            if (len(n_pipe['sample'][idx]) > 0) and \
                  ((n_pipe['sample'][idx][-1] == 0)         # translate over
                   or (n_pipe['heads'][idx][1] >= maxlen)   # exceed the maximum length
                   or (step > (1.5 * maxlen))):

                for key in ['sample', 'score', 'action', 'obs',
                            'attentions', 'old_attend', 'coverage',
                            'forgotten', 'cmask', 'seq_info']:
                    pipe[key].append(n_pipe[key][idx])

                pipe['i_mask'].append(n_pipe['i_mask'][:, idx])
                pipe['source'].append(n_pipe['source'][:, idx])
                live_k -= 1

            else:

                for key in ['sample', 'score', 'headsp', 'heads',
                            'prev_z', 'action', 'prev_wsrc',
                            'obs', 'attentions', 'hidden',
                            'old_attend', 'coverage',
                            'forgotten', 'cmask', 'seq_info']:
                    h_pipe[key].append(n_pipe[key][idx])

                h_pipe['maskp'].append(n_pipe['maskp'][:, idx])
                h_pipe['mask'].append(n_pipe['mask'][:, idx])
                h_pipe['ctxp'].append(n_pipe['ctxp'][:, idx])
                h_pipe['ctx'].append(n_pipe['ctx'][:, idx])
                h_pipe['i_mask'].append(n_pipe['i_mask'][:, idx])
                h_pipe['source'].append(n_pipe['source'][:, idx])

        # make it numpy array
        for key in ['headsp', 'heads', 'score', 'coverage',
                    'maskp', 'mask', 'ctx', 'prev_z',
                    'hidden', 'source', 'i_mask']:
            h_pipe[key] = numpy.asarray(h_pipe[key])
        h_pipe['ctxp'] = numpy.asarray(h_pipe['ctxp'], dtype='float32')
        h_pipe['maskp']   = h_pipe['maskp'].T
        h_pipe['mask']   = h_pipe['mask'].T
        h_pipe['source'] = h_pipe['source'].T
        h_pipe['i_mask'] = h_pipe['i_mask'].T

        #print h_pipe['ctx'].shape
        if h_pipe['ctx'].ndim == 3:
            h_pipe['ctx']    = h_pipe['ctx'].transpose(1, 0, 2)

        if h_pipe['ctxp'].ndim == 3:
            h_pipe['ctxp']    = h_pipe['ctxp'].transpose(1, 0, 2)

        elif h_pipe['ctx'].ndim == 2:
            h_pipe['ctx']  = h_pipe['ctx'][:, None, :]

        elif h_pipe['ctxp'].ndim == 2:
            h_pipe['ctxp']  = h_pipe['ctxp'][:, None, :]

        #print h_pipe['ctx'].shape
        h_pipe['prev_hid'] = h_pipe['hidden']
        h_pipe['prev_w']   = numpy.array([w[-1] if len(w) > 0
                                          else -1 for w in h_pipe['sample']], dtype='int64')
        h_pipe['prev_wsrc'] = np.array(h_pipe['prev_wsrc'])

    #session.close()
    #tf.reset_default_graph()

    # =======================================================================
    # Collecting Rewards.
    # =======================================================================
    R     = []
    track = []

    Ref   = []
    Sys   = []

    Words = []   # Sample Sentence
    SWord = []   # Source Sentence
    TWord = []   # Target Sentence

    max_steps   = -1    # Maximum number of number of actions per sample
    max_w_steps = -1    # Maximum number of words per sample.

    for k in range(live_all):
        sp, sc, act, sec_info = [pipe[key][k] for key in ['sample', 'score', 'action', 'seq_info']]
        reference   = [_bpe2words(_seqs2words([trgs[sec_info[0]]], t_idict))[0].split()]
        y_sample    = numpy.asarray(sp,  dtype='int64')[:, None]
        y_sample_mask = numpy.ones_like(y_sample, dtype='float32')

        steps       = len(act)
        w_steps     = len(sp)

        # turn back to sentence level
        words       = _seqs2words([sp], t_idict)[0]
        decoded     = _bpe2words([words])[0].split()

        Ref        += [reference]
        Sys        += [decoded]
        Words      += [words]
        SWord      += [srcs[sec_info[0]]]
        TWord      += [trgs[sec_info[0]]]

        # ----------------------------------------------------------------
        # reward keys
        # ----------------------------------------------------------------
        keys = {"steps": steps,
                "y": y_sample, "y_mask": y_sample_mask,
                "x_mask": x_mask,
                "act": act, "src_max": src_max,
                "ctx0": ctx0, "sidx": sidx,
                "f_cost": f_cost, "alpha": 0.5,
                "sample": decoded,
                "reference": reference,
                "words": words,
                "source_len": sec_info[1],

                'target': options['target_ap'],
                'cw': options['target_cw'],
                'gamma': options['gamma'],
                'Rtype': options['Rtype'],
                'maxsrc': options['maxsrc']}

        ret = return_reward(**keys)
        Rk, quality, delay, instant_reward, pred = ret
        reward = numpy.mean(instant_reward)  # the terminal reward

        if steps > max_steps:
            max_steps = steps

        if w_steps > max_w_steps:
            max_w_steps = w_steps

        R     += [Rk]
        track += [(quality, delay, reward)]

    pipe['R']     = R
    pipe['track'] = track
    pipe['Ref']   = Ref
    pipe['Sys']   = Sys

    pipe['Words'] = Words
    pipe['SWord'] = SWord
    pipe['TWord'] = TWord

    # If not train, End here
    if not train:
        return pipe

    info = OrderedDict()
    p_r  = _padding(pipe['R'], shape=(max_steps, live_all))
    p_obs, p_mask = _padding(pipe['obs'],
                             shape=(max_steps, live_all, agent.n_in),
                             return_mask=True, sidx=sidx)
    p_act         = _padding(pipe['action'],
                             shape=(max_steps, live_all), dtype='int64')

    # ================================================================= #
    # Policy Gradient over Trajectories for the Agent
    # ================================================================= #
    if not options['train_gt']:

        # learning
        info_t   = agent.get_learner()(
                [p_obs, p_mask], p_act, p_r,
                lr=options['lr_policy'])
        info.update(info_t)
        p_adv         = info['advantages']

    else:
        p_adv         = p_r

    # ================================================================ #
    # Policy Gradient for the underlying NMT model
    # ================================================================ #
    if options['finetune']:
        p_y, p_y_mask = _padding(pipe['sample'],
                                 shape=(max_w_steps, live_all),
                                 return_mask=True, dtype='int64')
        p_x      = numpy.asarray(pipe['source']).T
        p_i_mask = numpy.asarray(pipe['i_mask']).T
        p_c_mask = _padding(pipe['cmask'],
                            shape=(max_w_steps, live_all, p_x.shape[0]))

        new_adv = [p_adv[p_act[:, s] == 1, s] for s in range(p_adv.shape[1])]
        new_adv, one_reward = _padding(new_adv, shape=(max_w_steps, live_all), return_mask=True)

        if not options['train_gt']:
            a_cost, _ = ff_cost(p_x, p_i_mask, p_y, p_y_mask,
                                p_c_mask.transpose(0, 2, 1), new_adv)
        else:
            a_cost, _ = ff_cost(p_x, p_i_mask, p_y, p_y_mask,
                                p_c_mask.transpose(0, 2, 1), one_reward)
        ff_update(options['lr_model'])

        info['a_cost'] = a_cost

    # add the reward statistics
    q, d, r = list(zip(*pipe['track']))
    info['Quality']   = numpy.mean(q)
    info['Delay']     = numpy.mean(d)
    info['StartR']    = numpy.mean(r)

    _probs     /= float(_total)
    info['p(READ)']   = _probs[0]
    info['p(WRITE)'] = _probs[1]

    if options['predict']:
        info['F']   = _probs[2]

    end_time = time.clock()
    
    print("Average time for READ is : ", str(numpy.mean(numpy.array(timeread))))
    print("Average time for WRITE is : ", str(numpy.mean(numpy.array(timewrite))))
    print("Average time for PREDICT is : ", str(numpy.mean(numpy.array(timepredict0))))
    print("Average time for Source-side Predict is : ", str(numpy.mean(numpy.array(timepredict1))))

    print("time is : ", str(end_time-start_time), " seconds")
    return pipe, info

