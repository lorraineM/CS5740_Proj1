
# coding: utf-8

# In[2]:

from collections import defaultdict
import operator
import random
import numpy as np

def get_corpus(file):
    corpus = []
    f = open(file, 'r')
    for line in f:
        # append start word
        new_line = ['<s>']
        for word in line.split():
            new_line.append(word.lower())
        # append stop word
        new_line.append('</s>')
        corpus.append(new_line)
    return corpus

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

# generate the next random word using unigram model
def get_next_word_unigram(corpus):
    keys = [key for key in sorted(corpus, key=corpus.get)]
    probs = [corpus[key] for key in sorted(corpus, key=corpus.get)]
    return np.random.choice(keys, 1, probs)[0]
    
# generate the next random word using bigram model
def get_next_word_bigram(sentence, corpus):
    last_word = sentence[len(sentence) - 1]
    candidates = defaultdict(float)
    total_prob = 0.0
    for key in corpus:
        if key[0] == last_word:
            candidates[key[1]] = corpus[key]
            total_prob += corpus[key]
    
    for key in candidates:
        candidates[key] /= total_prob
    
    keys = [key for key in sorted(candidates, key=candidates.get)]
    probs = [candidates[key] for key in sorted(candidates, key=candidates.get)]
    return np.random.choice(keys, 1, probs)[0]

pos_file = 'SentimentDataset/Train/pos.txt'
neg_file = 'SentimentDataset/Train/neg.txt'
pos_corpus = get_corpus(pos_file)
neg_corpus = get_corpus(neg_file)


#generate unigram models for postive and negative files.
pos_unigram_freqs = get_unigram_freqs(pos_corpus)
neg_unigram_freqs = get_unigram_freqs(neg_corpus)

#generate bigram models for positive and negative files
pos_bigram_freqs = get_bigram_freqs(pos_corpus, pos_unigram_freqs)
neg_bigram_freqs = get_bigram_freqs(neg_corpus, neg_unigram_freqs)

# Generate a sentence using unigram model
def generate_unigram_sentence(length, model):
    e = pos_unigram_freqs
    if model == 0:
        e = neg_unigram_freqs
    start = ['<s>']
    for i in range(0, length):
        nxt = get_next_word_unigram(e)
        if nxt == '</s>' or nxt == '.':
            break
        start.append(nxt)
        
    print(' '.join(start[1:]))

print("Generate Unigram Sentence using positive sentiment file:")
generate_unigram_sentence(20,1)

# Generate a sentence using bigram model
def generate_bigram_sentence(length, model):
    e = pos_bigram_freqs
    if model == 0:
        e = neg_bigram_freqs
    bi_start = ['<s>']
    for i in range(0, length):
        nxt = get_next_word_bigram(bi_start, e)
        if nxt == '</s>' or nxt == '.':
            break
        bi_start.append(nxt)

    print(' '.join(bi_start[1:]))

print("Generate Bigram Sentence using positive sentiment file:")
generate_bigram_sentence(20,1)

print()
print("Generate Unigram Sentence using negative sentiment file:")
generate_unigram_sentence(20,0)
print("Generate Bigram Sentence using negative sentiment file:")
generate_unigram_sentence(20,0)


# In[4]:

# Here we added the unknown options and we assume the words that only appear once in the corpus as unknown
def get_corpus_with_unknown(corpus):
    w_count = defaultdict(float)
    for line in corpus:
        for word in line:
            w_count[word.lower()] += 1
    words_to_be_removed = set([key for key in w_count if w_count[key] == 1])
    corpus_filtered = [list(c) for c in corpus]
    for i in range(0, len(corpus_filtered)):
        for j in range(0, len(corpus_filtered[i])):
            if corpus_filtered[i][j] in words_to_be_removed:
                corpus_filtered[i][j] = '<unk>'
    return corpus_filtered

def get_bigram_corpus(corpus):
    bi_corpus = []
    for line in corpus:
        l = []
        for i in range(len(line) - 1):
            l.append((line[i].lower(), line[i + 1].lower()))
        bi_corpus.append(l)
    return bi_corpus


# In[5]:

pos_corpus_with_unknown = get_corpus_with_unknown(pos_corpus)
neg_corpus_with_unknown = get_corpus_with_unknown(neg_corpus)
pos_unigram_freqs_with_unknown = get_unigram_freqs(pos_corpus_with_unknown)
neg_unigram_freqs_with_unknown = get_unigram_freqs(neg_corpus_with_unknown)
pos_bigram_freqs_with_unknown = get_bigram_freqs(pos_corpus_with_unknown, pos_unigram_freqs_with_unknown)
neg_bigram_freqs_with_unknown = get_bigram_freqs(neg_corpus_with_unknown, neg_unigram_freqs_with_unknown)
pos_bigram_corpus_with_unknown = get_bigram_corpus(pos_corpus_with_unknown)
neg_bigram_corpus_with_unknown = get_bigram_corpus(neg_corpus_with_unknown)


# In[244]:




# In[224]:




# In[7]:

pos_bigram_freqs_with_unknown


# In[6]:

#def calculate_perplexity(model):
    #return np.exp(1.0 / len(model) * sum([-np.log(model[m]) for m in model]))
    
def calculate_perplexity_unigram(sentence, model):
    log_prob = []
    for uni in sentence:
        if uni in model:
            log_prob.append(-np.log(model[uni]))
        else:
            log_prob.append(-np.log(model['<unk>']))
    return np.exp(1.0 / len(model) * (sum(log_prob)))

def calculate_perplexity_bigram(sentence, model, start_prob):
    log_prob = [-np.log(start_prob)]
    for bi in sentence:
        if bi in model:
            log_prob.append(-np.log(model[bi]))
        elif (bi[0], '<unk>') in model:
            log_prob.append(-np.log(model[(bi[0], '<unk>')]))
        elif ('<unk>', bi[1]) in model:
            log_prob.append(-np.log(model[('<unk>', bi[1])]))
        else:
            log_prob.append(-np.log(model[('<unk>', '<unk>')]))
    return np.exp(1.0 / len(model) * (sum(log_prob)))
        
    #return np.exp(1.0 / len(n_gram_sentence) * (sum([-np.log(model[m]) for m in n_gram_sentence if m in model else -np.log(model['<unk>'])])))


# In[7]:

[w for w in pos_bigram_freqs_with_unknown if w[1] == '<unk>']
pos_bigram_freqs_with_unknown[('<unk>', '<unk>')]


# In[8]:

calculate_perplexity_unigram(pos_corpus_with_unknown[1], pos_unigram_freqs_with_unknown)
calculate_perplexity_bigram(pos_bigram_corpus_with_unknown[1], neg_bigram_freqs_with_unknown, neg_unigram_freqs_with_unknown['<s>'])

dev_pos_corpus = get_corpus('SentimentDataset/Dev/pos.txt')
dev_neg_corpus = get_corpus('SentimentDataset/Dev/neg.txt')
dev_pos_corpus_with_unknown = get_corpus_with_unknown(dev_pos_corpus)
dev_neg_corpus_with_unknown = get_corpus_with_unknown(dev_neg_corpus)
dev_pos_unigram_freqs_with_unknown = get_unigram_freqs(dev_pos_corpus_with_unknown)
dev_neg_unigram_freqs_with_unknown = get_unigram_freqs(dev_neg_corpus_with_unknown)
dev_pos_bigram_freqs_with_unknown = get_bigram_freqs(dev_pos_corpus_with_unknown, dev_pos_unigram_freqs_with_unknown)
dev_neg_bigram_freqs_with_unknown = get_bigram_freqs(dev_neg_corpus_with_unknown, dev_neg_unigram_freqs_with_unknown)
dev_pos_bigram_corpus_with_unknown = get_bigram_corpus(dev_pos_corpus_with_unknown)
dev_neg_bigram_corpus_with_unknown = get_bigram_corpus(dev_neg_corpus_with_unknown)


# In[9]:

def predict_with_unigram_perplexity(corpus, pos_model, neg_model):
    res = []
    for line in corpus:
        pos_pp = calculate_perplexity_unigram(line, pos_model)
        neg_pp = calculate_perplexity_unigram(line, neg_model)
        if pos_pp <= neg_pp:
            res.append(0)
        else:
            res.append(1)
    return res


def predict_with_bigram_perplexity(corpus, pos_model, neg_model):
    res = []
    for line in corpus:
        pos_pp = calculate_perplexity_bigram(line, pos_model, pos_unigram_freqs_with_unknown['<s>'])
        neg_pp = calculate_perplexity_bigram(line, neg_model, neg_unigram_freqs_with_unknown['<s>'])
        if pos_pp <= neg_pp:
            res.append(0)
        else:
            res.append(1)
    return res

def accuracy(prediction, actual):
    return 1 - sum([1 for i in range(len(prediction)) if prediction[i] != actual[i]]) * 1.0 / len(prediction)


# In[10]:

### perplexity does not produce good results
dev_pos_pred_uni_pp = predict_with_unigram_perplexity(dev_pos_corpus_with_unknown, pos_unigram_freqs_with_unknown, neg_unigram_freqs_with_unknown)
print(accuracy(dev_pos_pred_uni_pp, [0 for l in range(len(dev_pos_pred_uni_pp))]))

dev_neg_pred_uni_pp = predict_with_unigram_perplexity(dev_neg_corpus_with_unknown, pos_unigram_freqs_with_unknown, neg_unigram_freqs_with_unknown)
print(accuracy(dev_neg_pred_uni_pp, [0 for l in range(len(dev_neg_pred_uni_pp))]))


# In[11]:

dev_pos_pred_bi_pp = predict_with_bigram_perplexity(dev_pos_bigram_corpus_with_unknown, pos_bigram_freqs_with_unknown, neg_bigram_freqs_with_unknown)
print(accuracy(dev_pos_pred_bi_pp, [0 for l in range(len(dev_pos_pred_bi_pp))]))

dev_neg_pred_bi_pp = predict_with_bigram_perplexity(dev_neg_bigram_corpus_with_unknown, pos_bigram_freqs_with_unknown, neg_bigram_freqs_with_unknown)
print(accuracy(dev_neg_pred_bi_pp, [0 for l in range(len(dev_neg_pred_bi_pp))]))


# In[12]:

test_corpus = get_corpus('SentimentDataset/Test/test.txt')
test_corpus_with_unknown = get_corpus_with_unknown(test_corpus)
test_bigram_corpus_with_unknown = get_bigram_corpus(test_corpus_with_unknown)

test_pred_uni_pp = predict_with_unigram_perplexity(test_corpus_with_unknown, pos_unigram_freqs_with_unknown, neg_unigram_freqs_with_unknown)
test_pred_bi_pp = predict_with_bigram_perplexity(test_bigram_corpus_with_unknown, pos_bigram_freqs_with_unknown, neg_bigram_freqs_with_unknown)


# In[13]:

pred_file = open('section6.csv', 'w')
pred_file.write('Id,Prediction\n')
for i in range(len(test_pred_uni_pp)):
    pred_file.write(str(i + 1) + ',' + str(test_pred_bi_pp[i]) + '\n')
pred_file.close()


# In[14]:

def smooth(_lambda, unigram_model, bigram_model):
    new_model = defaultdict(float)
    for w in bigram_model:
        if len(w) == 2:
            new_model[w] = _lambda * bigram_model[w] + (1.0 - _lambda) * unigram_model[w[1]]
    return new_model


# In[15]:

pos_mixed_freqs = smooth(0.9, pos_unigram_freqs, pos_bigram_freqs)
print(calculate_perplexity(pos_mixed_freqs))

pos_mixed_freqs_with_unknown = smooth(0.9, pos_unigram_freqs_with_unknown, pos_bigram_freqs_with_unknown)
print(calculate_perplexity(pos_mixed_freqs_with_unknown))


# In[12]:

[w for w in pos_bigram_freqs if w[0] == "'s"]


# In[16]:

# the frequency of each word in the corpus, eg P(word = A) with Add-k smoothing
def get_unigram_freqs_add_k(corpus, k):
    w_count = defaultdict(float)
    total_count = 0
    for line in corpus:
        for word in line:
            w_count[word.lower()] += 1
            total_count += 1
    V = len(w_count)
    for key in w_count:
        w_count[key] = (w_count[key] + k) / (1.0 * total_count + k * V)  
    return w_count

# the frequency of each bigram in the corpus, eg P(B = key[1] | A = key[0]) with Add-k smoothing
def get_bigram_freqs_add_k(corpus, k):
    bi_count = defaultdict(float)
    for line in corpus:
        for i in range(len(line) - 1):
            bi_count[(line[i].lower(), line[i + 1].lower())] += 1
    
    w_count = defaultdict(float)
    for line in corpus:
        for word in line:
            w_count[word.lower()] += 1
            
    V = len(w_count)
    for key in bi_count:
        bi_count[key] = (bi_count[key] * 1.0 + k) / (w_count[key[0]] + k * V)
    return bi_count


# In[17]:

k = 0.0
pos_unigram_freqs_smoothed = get_unigram_freqs_add_k(pos_corpus, k)
pos_unigram_freqs_with_unknown_smoothed = get_unigram_freqs_add_k(pos_corpus_with_unknown, k)


# In[18]:

pos_bigram_freqs_smoothed = get_bigram_freqs_add_k(pos_corpus, k)
pos_bigram_freqs_with_unknown_smoothed = get_bigram_freqs_add_k(pos_corpus_with_unknown, k)


# In[19]:

print(calculate_perplexity(pos_unigram_freqs_smoothed))
print(calculate_perplexity(pos_unigram_freqs_with_unknown_smoothed))
print(calculate_perplexity(pos_bigram_freqs_smoothed))
print(calculate_perplexity(pos_bigram_freqs_with_unknown_smoothed))


# In[40]:




# In[20]:

def add_corpus(source_corpus, to_add):
    for line in to_add:
        for w in line:
            source_corpus.append(w)
    return source_corpus
    
def feature(original_corpus, training_corpus, training_corpusID):
    training_corpus_set = set(training_corpus)
    X = []
    for line in original_corpus:
        #print(line)
        feat = [0] * len(training_corpusID)
        for w in line:
            if w in training_corpus:
                #print(w, ' ', training_corpusID[w])
                feat[training_corpusID[w]] += 1
            else:
                feat[training_corpusID['<unk>']] += 1
        feat.append(1)
        X.append(feat)
    return X



# In[39]:

pos_corpus_with_unknown


# In[143]:

training_corpus = add_corpus([], pos_corpus_with_unknown)
training_corpus = add_corpus(training_corpus, neg_corpus_with_unknown)
training_corpus_set = set(training_corpus)
training_corpus = list(training_corpus_set)
training_corpus.sort()
training_corpusID = dict(zip(training_corpus, range(len(training_corpus))))


# In[57]:

def horizontal_concat(corpusA, corpusB):
    new_corpus = list()
    for i in range(len(corpusA)):
        new_corpus.append(corpusA[i] + corpusB[i])
    return new_corpus

mixed_training_pos_corpus = horizontal_concat(pos_corpus_with_unknown, pos_bigram_corpus_with_unknown)
mixed_training_neg_corpus = horizontal_concat(neg_corpus_with_unknown, neg_bigram_corpus_with_unknown)
mixed_training_corpus = add_corpus([], mixed_training_pos_corpus)
mixed_training_corpus = add_corpus(mixed_training_corpus, mixed_training_neg_corpus)
mixed_training_corpus_set = set(mixed_training_corpus)
mixed_training_corpus = list(mixed_training_corpus_set)
mixed_training_corpus.sort()
mixed_training_corpusID = dict(zip(mixed_training_corpus, range(len(mixed_training_corpus))))
len(mixed_training_corpus)


# In[131]:

pos_bigrams_selected = sorted(pos_bigram_freqs_with_unknown.items(), key=operator.itemgetter(1), reverse=True)[:2000]
neg_bigrams_selected = sorted(neg_bigram_freqs_with_unknown.items(), key=operator.itemgetter(1), reverse=True)[:2000]
neg_bigrams_selected


# In[132]:

unigram_training_corpus = add_corpus([], pos_corpus_with_unknown)
unigram_training_corpus = add_corpus(unigram_training_corpus, neg_corpus_with_unknown)
unigram_training_corpus = list(set(unigram_pos_training_corpus))
unigram_training_corpus.sort()
print(len(unigram_training_corpus))




#bigram_training_corpus = add_corpus([], pos_bigram_corpus_with_unknown)
#bigram_training_corpus = add_corpus(bigram_training_corpus, neg_bigram_corpus_with_unknown)
bigram_training_corpus = [w[0] for w in pos_bigrams_selected]
bigram_training_corpus = bigram_training_corpus + [w[0] for w in neg_bigrams_selected]
bigram_training_corpus = list(set(bigram_training_corpus))
bigram_training_corpus.sort()
print(len(bigram_training_corpus))

#mixed_training_corpus = horizontal_concat(unigram_training_corpus, bigram_training_corpus)
mixed_training_corpus = unigram_training_corpus + bigram_training_corpus
mixed_training_corpusID = dict(zip(mixed_training_corpus, range(len(mixed_training_corpus))))
print(len(mixed_training_corpus))
print(mixed_training_corpusID[('surface', 'flash')])

def feature_mixed(original_corpus, training_corpus, training_corpusID):
    training_corpus_set = set(training_corpus)
    X = []
    for line in original_corpus:
        #print(line)
        feat = [0] * len(training_corpusID)
        for w in line:
            if w in training_corpus:
                #print(w, ' ', training_corpusID[w])
                feat[training_corpusID[w]] += 1
            else:
                feat[training_corpusID['<unk>']] += 1
        
        for i in range(len(line) - 1):
            bi = (line[i].lower(), line[i + 1].lower())
            if bi in training_corpus:
                feat[training_corpusID[bi]] += 1      
        feat.append(1)
        X.append(feat)
    return X


# In[144]:

X_pos = feature(pos_corpus, training_corpus, training_corpusID)
y_pos = [0 for l in pos_corpus]
X_neg = feature(neg_corpus, training_corpus, training_corpusID)
y_neg = [1 for l in neg_corpus]


# In[133]:

X_pos = feature_mixed(pos_corpus, mixed_training_corpus, mixed_training_corpusID)
y_pos = [0 for l in pos_corpus]
X_neg = feature(neg_corpus, mixed_training_corpus, mixed_training_corpusID)
y_neg = [1 for l in neg_corpus]


# In[145]:

X = X_pos + X_neg
y = y_pos + y_neg


# In[146]:

len(X[0])


# In[27]:

import scipy.optimize
import nltk
from nltk.stem.porter import *
from sklearn import linear_model
'''
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_
predictions = clf.predict(X)'''


# In[163]:

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 6), random_state=1)
clf.fit(X, y)


# In[162]:

for w in range(5, 10):
    for d in range(5, 10):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(w, d), random_state=1)
        clf.fit(X, y)
        dev_pos_corpus_original = get_corpus('SentimentDataset/Dev/pos.txt')
        dev_neg_corpus_original = get_corpus('SentimentDataset/Dev/neg.txt')
        X_dev = feature(dev_pos_corpus_original + dev_neg_corpus_original, training_corpus, training_corpusID)
        y_dev = [0 for i in range(len(dev_pos_corpus_original))] + [1 for i in range(len(dev_neg_corpus_original))]
        pred_dev_res = clf.predict(X_dev)
        print('size: ', w, ' x ',d , ' with accuracy: ', accuracy(pred_dev_res, y_dev))


# In[31]:

pred_res = []
for p in predictions:
    if p >= 0.5:
        pred_res.append(1)
    else:
        pred_res.append(0)


# In[155]:

pred_res = clf.predict(X)


# In[156]:

pred_res


# In[157]:

error = sum([1 for i in range(len(pred_res)) if pred_res[i] != y[i]])


# In[158]:

error


# In[37]:

# Use development set to validate our training model
#dev_pos_corpus_original = get_corpus('SentimentDataset/Dev/pos.txt')
#dev_neg_corpus_original = get_corpus('SentimentDataset/Dev/neg.txt')
#X_dev = feature(dev_pos_corpus_original + dev_neg_corpus_original, training_corpus, training_corpusID)
#y_dev = [0 for i in range(len(dev_pos_corpus_original))] + [1 for i in range(len(dev_neg_corpus_original))]


# In[159]:

# Use development set to validate our training model
dev_pos_corpus_original = get_corpus('SentimentDataset/Dev/pos.txt')
dev_neg_corpus_original = get_corpus('SentimentDataset/Dev/neg.txt')
X_dev = feature(dev_pos_corpus_original + dev_neg_corpus_original, training_corpus, training_corpusID)
y_dev = [0 for i in range(len(dev_pos_corpus_original))] + [1 for i in range(len(dev_neg_corpus_original))]


# In[160]:

pred_dev_res = clf.predict(X_dev)

print(accuracy(pred_dev_res, y_dev))


# In[116]:

len(X_dev[0])


# In[140]:

test_corpus_original = get_corpus('SentimentDataset/Test/test.txt')


# In[164]:

test_corpus_original = get_corpus('SentimentDataset/Test/section8_test.txt')


# In[165]:

X_test = feature(test_corpus_original, training_corpus, training_corpusID)


# In[166]:

predictions = clf.predict(X_test)
pred_res = []
for p in predictions:
    if p >= 0.5:
        pred_res.append(1)
    else:
        pred_res.append(0)


# In[169]:

pred_res = clf.predict(X_test)


# In[170]:

pred_res


# In[ ]:




# In[171]:

pred_file = open('section8.csv', 'w')
pred_file.write('Id,Prediction\n')
for i in range(len(pred_res)):
    pred_file.write(str(i + 1) + ',' + str(pred_res[i]) + '\n')
pred_file.close()


# In[ ]:



