import numpy

import pickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 cache=10,
                 eos=False):

        self.source = fopen(source, 'r')
        self.target = fopen(target, 'w+')

        self.window = 3

        print('scan the dataset.')
        for si, sent in enumerate(self.source):
            tarsent = []
            for word in sent.split():
                tarsent.append(word + " ")
            del tarsent[0]
            tarsent.append('eos')
            tarsent = ''.join(tarsent)
            self.target.write(tarsent + '\n') 
            pass

        self.target.close()
        self.target = fopen(target, 'r')
        for ti, _ in enumerate(self.target):
            pass

        self.source.close()
        self.target.close()

        #assert si == ti, 'the number of the source and target document must the same'
        print(('scanned {} lines'.format(si)))

        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')

        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
            self.target_dict = self.source_dict

        self.num = si
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * cache

        self.end_of_data = False




    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break

                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            # sort by target buffer
            #tlen = numpy.array([len(t) for t in self.target_buffer])
            #tidx = tlen.argsort()

            #_sbuf = [self.source_buffer[i] for i in tidx]
            #_tbuf = [self.target_buffer[i] for i in tidx]

            #self.source_buffer = _sbuf
            #self.target_buffer = _tbuf

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target


class WindowIterator:

    def __init__(self, source, size, batch_size=16, padded=True ):
        self.source = source
        self.size = size
        self.padded = padded
        self.batch_size = batch_size

        if self.padded:
            new_source = numpy.zeros(( len(self.source) + (2*self.size) )).astype('int64')
            new_source[self.size:len(self.source)+self.size] = self.source
            self.source = new_source

        self.pointer = 1
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.pointer = 1

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        try:
            while True:
                ssent = []
                tsent = []
                if self.pointer+self.size < len(self.source):
                    # print self.pointer, len(self.source)
                    ssent = self.source[self.pointer:self.pointer+self.size]
                    tsent = [self.source[self.pointer+self.size]]
                    self.pointer += 1

                source.append(ssent)
                target.append(tsent)

                if (len(source) >= self.batch_size) or (len(target) >= self.batch_size):
                    break

            if ssent == [] or tsent == []:
                raise IOError

        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target


def iterate(fname, word_dict, n_words):
    with open(fname, 'r') as f:
        for line in f:
            words = line.strip().split()
            x = [word_dict[w] if w in word_dict else 1 for w in words]
            x = [ii if ii < n_words else 1 for ii in x]
            x += [0]
            yield x


def check_length(fname):
    f = open(fname, 'r')
    count = 0
    for _ in f:
        count += 1
    f.close()
    return count


