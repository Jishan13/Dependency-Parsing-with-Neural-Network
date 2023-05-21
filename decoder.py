from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras
from extract_training_data import FeatureExtractor, State
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class Parser(object):

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            features = self.extractor.get_input_representation(words, pos, state)
            features = features.reshape((1, features.size))
            predict = self.model.predict(features)
            idx = np.argsort(-predict)[0]
            for i in idx:
                transition, POS = self.output_labels[i]
                buff_len = len(state.buffer)
                stack_len = len(state.stack)
                if transition == 'shift':
                    if buff_len == 1 and stack_len == 0:
                        state.shift()
                        break
                    elif buff_len > 1:
                        state.shift()
                        break
                elif transition == 'left_arc':
                    if stack_len > 0 and POS != 'root':
                        state.left_arc(POS)
                        break
                elif transition == 'right_arc':
                    if stack_len > 0:
                        state.right_arc(POS)
                        break
        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, 'r')
        pos_vocab_f = open(POS_VOCAB_FILE, 'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2], 'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
