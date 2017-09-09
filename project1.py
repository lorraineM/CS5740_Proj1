
# coding: utf-8

# In[46]:

from collections import defaultdict
import operator
import random

def get_corpus(file):
    corpus = []
    f = open(file, 'r')
    for line in f:
        new_line = []
        for word in line.split():
            new_line.append(word.lower())
        corpus.append(new_line)
    return corpus

def get_corpus_set(corpus):
    return set([word for word_list in corpus for word in word_list])

def get_random_word(corpus):
    return random.sample(get_corpus_set(corpus), 1)

# the frequency of each word in the corpus, eg P(word = A)
def get_unigram_freqs(corpus):
    w_count = defaultdict(float)
    total_count = 0
    for line in corpus:
        for word in line:
            w_count[word.lower()] += 1
            total_count += 1
    for key in w_count:
        w_count[key] /= (1.0 * total_count)  
    return w_count

# the frequency of each bigram in the corpus, eg P(B = key[1] | A = key[0])
def get_bigram_freqs(corpus, unigram_freqs):
    bi_count = defaultdict(float)
    total_count = 0
    for line in corpus:
        for i in range(len(line) - 1):
            bi_count[(line[i].lower(), line[i + 1].lower())] += 1
            total_count += 1
            
    for key in bi_count:
        bi_count[key] = (bi_count[key] * 1.0 / total_count) / unigram_freqs[key[0]]
    return bi_count

### TO BE MODIFIED LATER
def get_next_word_unigram(sentence, corpus_set):
    return get_random_word(corpus_set)

### TO BE MODIFIED LATER
def get_next_word_bigram(sentence, corpus):
    last_word = sentence[len(sentence) - 1]
    candidates = defaultdict(float)
    total_prob = 0.0
    for key in corpus:
        if key[0] == last_word:
            candidates[key[1]] = corpus[key]
            total_prob += corpus[key]
    cumulative_prob = 0.0
    for key in candidates:
        term_prob = candidates[key] / total_prob
        cumulative_prob += term_prob
        candidates[key] = cumulative_prob
    
    rand = random.uniform(0.0, 1.0)
    prev = 0.0
    for key in sorted(candidates, key=candidates.get):
        if rand > prev and rand <= candidates[key]:
            return key
        else:
            prev = candidates[key]

    return max(candidates.items(), key=operator.itemgetter(1))[0]

pos_file = 'SentimentDataset/Train/pos.txt'
pos_corpus = get_corpus(pos_file)


# In[47]:

pos_unigram_freqs = get_unigram_freqs(pos_corpus)
pos_unigram_freqs


# In[48]:

pos_bigram_freqs = get_bigram_freqs(pos_corpus, pos_unigram_freqs)
pos_bigram_freqs


# In[41]:

get_next_word_bigram(['sum'], pos_bigram_freqs)


# In[164]:

start = get_random_word(pos_all_words)
start
for i in range(0, 20):
    nxt = get_next_word_unigram(start, pos_unigrams)
    start.append(nxt)
    if nxt == '.':
        break
    print(start)


# In[57]:

bi_start = get_random_word(pos_corpus)
for i in range(0, 5):
    nxt = get_next_word_bigram(bi_start, pos_bigram_freqs)
    bi_start.append(nxt)
    if nxt == '.':
        break
    print(bi_start)


# In[ ]:



