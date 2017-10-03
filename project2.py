from collections import defaultdict
import operator
import random
import numpy as np


def get_training_corpus(file):
    corpus = []
    f = open(file, 'r')
    i = 1
    for line in f:
        if i%3 == 1:
            new_line = ['<s>']
            for word in line.split():
                new_line.append(word)
            corpus.append(new_line)
        elif i%3 == 2:
            new_line = ['<p>']
            for word in line.split():
                new_line.append(word)
            corpus.append(new_line)
        else:
            new_line = ['<e>']
            for word in line.split():
                new_line.append(word)
            corpus.append(new_line)
        i=i+1
    return corpus

training_data = 'train.txt'
t_corpus = get_training_corpus(training_data)

def get_test_corpus(file):
    corpus = []
    f = open(file, 'r')
    i = 1
    for line in f:
        if i%3 == 1:
            new_line = ['<s>']
            for word in line.split():
                new_line.append(word)
            corpus.append(new_line)
        elif i%3 == 2:
            new_line = ['<p>']
            for word in line.split():
                new_line.append(word)
            corpus.append(new_line)
        elif i%3 == 0:
            new_line = ['<i>']
            for word in line.split():
                new_line.append(word)
            corpus.append(new_line)
        i=i+1
    return corpus
    
test_data = 'test.txt'
test_corpus = get_test_corpus(test_data)

#lexicon <token, label> = #occurrence
#the most basic baseline system
def generate_baseline_NE_lexicon(corpus):
    lexicon = defaultdict(float)
    c_sentences = len(corpus)/3;
    for j in range(c_sentences):
        c_words = len(corpus[3*j+2])
        for i in range(c_words):
            if corpus[3*j+2][i] != 'O':
                lexicon[corpus[3*j][i], corpus[3*j+2][i]] += 1
    return lexicon

baseline_NE_lexicon = generate_baseline_NE_lexicon(t_corpus)

#calculate P(t_i)
def get_trasition_unigram_freqs(corpus):
    transition = defaultdict(float)
    c_transitions = 0
    for sentence in corpus:
        if sentence[0] == '<e>':
            for label in sentence:
                transition[label] += 1
                c_transitions += 1
    for key in transition:
        transition[key] /= (1.0 * c_transitions)
    return transition

transition_unigram_freqs = get_trasition_unigram_freqs(t_corpus)

#calculate P(t_i+1 | t_i)
def get_transition_bigram_freqs(corpus, unigram):
    transition = defaultdict(float)
    c_transitions = 0
    for sentence in corpus:
        if sentence[0] == '<e>':
            for i in range(len(sentence) - 1):
                transition[sentence[i], sentence[i+1]] += 1
                c_transitions += 1
    for key in transition:
        transition[key] = (transition[key] * 1.0 / c_transitions) / unigram[key[0]]
    return transition
transition_bigram_freqs = get_transition_bigram_freqs(t_corpus, transition_unigram_freqs)

#calculate P(w_i | t_i)
#<label, token>
def get_observation_freqs(corpus, unigram):
    observation = defaultdict(float)
    c_pairs = 0
    for j in range(len(corpus)):
        if corpus[j][0] == '<s>':
            for i in range(len(corpus[j])):
                observation[corpus[j+2][i], corpus[j][i]] += 1
                c_pairs += 1
    for key in observation:
        observation[key] = (observation[key] * 1.0 / c_pairs) / unigram[key[0]]
    return observation
observation_freqs = get_observation_freqs(t_corpus, transition_unigram_freqs)

#simple states B-ORG,I-ORG,B-LOC,I-LOC,B-PER,I-PER,B-MISC,I-MISC,O
def get_all_states(unigram):
    t = []
    for key in unigram:
        if key != '<e>':
            t.append(key)
    return t
T = get_all_states(transition_unigram_freqs)

#merge indices with label B-xxx with indices with label I-xxx into xxx,indices lists
def merge(b, inner):
    result = []
    c_starter = len(b)
    b = map(eval, b)
    inner = map(eval, inner)
    i = 0
    for j in range(c_starter):
        result.append(str(b[j]))
        if b[j]+1 in inner:
            k = 1
            for current in range(i,len(inner)):
                if inner[current] == b[j] + k:
                    result[j] += '-' + str(b[j] + k)
                    k += 1
                else:
                    i = current
                    break
        else:
            result[j] += '-' + str(b[j])
    return result

#basic viterbi_algorithm use 
#P(label_i+1 | label_i) as transition probabilities
#P(w_i | label_i) as lexical generation probabilities
def viterbi_algorithm(corpus, transition, observation, T):
    lexical_categories = defaultdict(list)
    c = len(T)
    for lines in range(len(corpus)):
        if corpus[lines][0] == '<s>':
            n = len(corpus[lines])
            tags = np.zeros((n), dtype=np.int16)
            single_line_category = defaultdict(list)
            score = np.zeros((c,n), dtype=np.float64)
            bptr = np.zeros((c,n), dtype=np.int16)
            #initialize
            for i in range(c):
                score[i][1] = transition['<e>', T[i]] * observation[T[i], corpus[lines][1]]
                bptr[i][1] = 0
            #iteration
            for t in range(2, n):
                for i in range(c):
                    a = np.zeros((c), dtype=np.float64)
                    for k in range(c):
                        num = score[k][t-1]*transition[T[k], T[i]]
                        a[k] = num
                    k = np.argmax(a)
                    score[i][t] = score[k][t-1] * transition[T[k], T[i]] * observation[T[i], corpus[lines][t]]
                    bptr[i][t] = k
            #finally token w_i has label T[tags[i]] 
            b = np.zeros((c), dtype=np.float64)
            for k in range(c):
                b[k] = score[k][n-1]
            tags[n-1] = np.argmax(b)
            for i in range(n-2,0,-1):
                tags[i] = bptr[tags[i+1]][i+1]            
            #finish to get <label> = index lists
            for i in range(1,n):
                single_line_category[T[tags[i]]].append(corpus[lines+2][i])
            r1 = merge(single_line_category['B-PER'], single_line_category['I-PER'])
            for m in range(len(r1)):
                lexical_categories['PER'].append(r1[m])
            r2 = merge(single_line_category['B-LOC'], single_line_category['I-LOC'])
            for m in range(len(r2)):
                lexical_categories['LOC'].append(r2[m])
            r3 = merge(single_line_category['B-ORG'], single_line_category['I-ORG'])
            for m in range(len(r3)):
                lexical_categories['ORG'].append(r3[m])
            r4 = merge(single_line_category['B-MISC'], single_line_category['I-MISC'])
            for m in range(len(r4)):
                lexical_categories['MISC'].append(r4[m])
    return lexical_categories

lexical_categories = viterbi_algorithm(test_corpus, transition_bigram_freqs, observation_freqs, T)

#normalize output 
#Type,Prediction
#PER,......
#LOC,......
#ORG,......
#MISC,.....
def output_test_result(result):
    f = open('testresult.csv', 'w')
    f.writelines('Type,Prediction\n')
    str1 = 'PER,'
    for index in range(len(result['PER'])):
        if result['PER'][index] != '':
            str1 += str(result['PER'][index]) + ' '
    str1 += '\n'
    f.writelines(str1)
    str2 = 'LOC,'
    for index in range(len(result['LOC'])):
        if result['LOC'][index] != '':
            str2 += str(result['LOC'][index]) + ' '
    str2 += '\n'
    f.writelines(str2)
    str3 = 'ORG,'
    for index in range(len(result['ORG'])):
        if result['ORG'][index] != '':
            str3 += str(result['ORG'][index]) + ' '
    str3 += '\n'
    f.writelines(str3)
    str4 = 'MISC,'
    for index in range(len(result['MISC'])):
        if result['MISC'][index] != '':
            str4 += str(result['MISC'][index]) + ' '
    str4 += '\n'
    f.writelines(str4)
    f.close()

output_test_result(lexical_categories)
