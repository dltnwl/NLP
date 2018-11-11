numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

import numpy as np
wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')

import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.5,variational_recurrent=True, dtype=tf.float32)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))


# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix
    
secondInputText = "Kodak found itself with $6.75 billion in debt"
secondInputMatrix = getSentenceMatrix(secondInputText)
    
predictedSentiment = sess.run(prediction, {input_data: secondInputMatrix})[0]
if (predictedSentiment[0] > predictedSentiment[1]):
    print("Positive Sentiment")
else:
    print("Negative Sentiment")
    
    
    
from sklearn import preprocessing
#preprocessing.scale(lis, 0)

lis=[]
for _ in range(20):
    predictedSentiment = sess.run(prediction, {input_data: secondInputMatrix})[0]
    lis.append(predictedSentiment)
    
positive=0
negative=0
for l in lis:
    if l[0]>l[1]:
        positive+=1
    else:
        negative+=1
print("positive prob:%d"%(positive/len(lis) ))




import pylab
fig=plt.figure()
fig.set_size_inches(10, 10)

ax=fig.add_subplot(1,1,1)
pylab.xlim([-5.5, 10])
pylab.ylim([-5.5, 10])


for i in range(10):
    ax.scatter(predictedSentiment[0], predictedSentiment[1])


x=np.arange(-5.5, 11, 1)
y=np.arange(-5.5, 11, 1)


ax.plot(x, y)




ax.annotate("Kodak found itself with $6.75 billion in debt", [predictedSentiment[0], predictedSentiment[1]], fontsize=15)
ax.annotate("Kodak couldn’t see the fundamental shift ", [predictedSentiment1[0], predictedSentiment1[1]], fontsize=15)
ax.annotate("Kodak failed  to grasp the significance of a technological transition", [predictedSentiment2[0], \
                                                                            predictedSentiment2[1]], fontsize=15)
ax.annotate("declining scale was also a big problem for Kodak", [predictedSentiment3[0], predictedSentiment3[1]], fontsize=15)
ax.annotate("Kodak gives up billions of dollars in profits", [predictedSentiment4[0], predictedSentiment4[1]], fontsize=15)

ax.annotate("Kodak's problem had to do with its ecosystem", [predictedSentiment5[0], predictedSentiment5[1]], fontsize=15)
ax.annotate("Kodak falls in the creative destruction of the digital age", [predictedSentiment6[0], predictedSentiment6[1]], fontsize=15)
ax.annotate("The moment it all went wrong for Kodak", [predictedSentiment7[0], predictedSentiment7[1]], fontsize=15)
ax.annotate("Kodak is on the verge of bankruptcy", [predictedSentiment8[0], predictedSentiment8[1]], fontsize=15)
ax.annotate("What Killed Kodak?", [predictedSentiment9[0], predictedSentiment9[1]], fontsize=15)



