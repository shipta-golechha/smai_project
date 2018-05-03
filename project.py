import numpy as np
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import operator
import string

# Margin proportional update

def update_margin(alphamatrix,R,S):
	tot_sum = 0 
	for row1 in R:
		for row2 in S:
			if row1[1]-row2[1] > 0:
				tot_sum = tot_sum + row1[1]-row2[1]


	for row1 in R:
		for row2 in S:
			if row1[1]-row2[1] > 0:
				alphamatrix[row1[0]-1][row2[0]-1]=float(row1[1]-row2[1])/float(tot_sum)

	return alphamatrix

# Max update function

def max_update(alphamatrix,R,S):
	R_dict = {}
	for row in R:
		R_dict[row[0]] = row[1]
	S_dict = {}
	for row in S:
		S_dict[row[0]] = row[1]
	sorted_r = sorted(R_dict.items(),key=operator.itemgetter(1))  # sorting all the w.x values (increasing order)
	sorted_s = sorted(S_dict.items(),key=operator.itemgetter(1),reverse=True)  # sorting all the w.x values (decreasing order)
	alphamatrix[sorted_r[0][0]-1][sorted_s[0][0]-1] = 1

	return alphamatrix


# Uniform update function

def Uniform_update(R,S,alphamatrix,E):

	for rows in R:
		for cols in S:
			if rows[1] <= cols[1]:
				alphamatrix[rows[0]-1][cols[0]-1] = 1.0 / len(E)

	return alphamatrix
						
# Randomized Update Function

def randomized_update(E,alphamatrix,R,S):

	L = np.random.normal(1.0, 0.005, len(E))/len(E)
	index = 0
	for rows in R:
		for cols in S:
			if rows[1] <= cols[1]:
				alphamatrix[rows[0]-1][cols[0]-1] = L[index]
				index = index + 1 

	return alphamatrix

# MaxF1 loss function

def maxF1(sorted_x,R):
	length = len(R)
	count = []
	local = 0
	index = 0
	for s in sorted_x:
		for rr in R:
			if s[0]==rr[0]:
				local=local+1
				break
		count.append(local)

	recall = []
	precision = []

	for re in range(len(sorted_x)):
		recall.append(float(count[re])/float(length))

	for pre in range(len(sorted_x)):
		precision.append(float(count[pre])/float(pre+1))
	
	max = []
	for ind in 	range(len(sorted_x)):
		if recall[ind]+precision[ind]!=0:
			max.append(float(2*recall[ind]*precision[ind])/float(recall[ind]+precision[ind]))

	cal_max = max[0]

	for element in max:
		if element > cal_max:
			cal_max = element

	return 1-cal_max

# One error loss function

def OneErrorLoss(sorted_x,R):
	flag = 0
	for relevant_topics in R:
		if sorted_x[0][0] == relevant_topics[0]:
			flag = 1
			break
	if flag == 1:
		return 0
	return 1

# IsErr loss function

def IsErrLoss(E):
	if len(E)>0:
		return 1
	else:
		return 0

# Error Set Size Loss Function

def ErrSetSizeLoss(E):
	return len(E)

# Average Loss Function

def AvgPLoss(relevant, ranking):
	x = 1
	rel_ranking_dict = {}
	for relranking in relevant:
		x = 1
		for ranks in ranking:
			if ranks[0] == relranking[0]:
				break
			else:
				x = x + 1
		rel_ranking_dict[relranking[0]] = x
	
	x = 0
	sum = 0    
	for topic in rel_ranking_dict.keys():
		x = 1

		for otherTopic in rel_ranking_dict.keys():
			if rel_ranking_dict[topic] >= rel_ranking_dict[otherTopic]:
				x = x + 1
			sum = sum + x /  rel_ranking_dict[topic]

	sum = sum / len(relevant)

	return (1 - sum)

########## Code for training ###########

def train(tfidfmatrix, x):
	print("Training...")
	n = len(tfidfmatrix[0])  #  19735, total number of unique words in all the documents
	w = np.zeros(shape=(k,n)) # shape of w is (90,19735) , 90 is total number of topics and 19735 is total number of unique words in all the documents
	i = 1

	for rows,docs in zip(tfidfmatrix,x): # rows represent tfidf corresponding each document, docs represent fileid of each document
		i = 1
		prototypes = {} # for storing all w
		ans = np.dot(w,rows.T)  # w.x
		
		for j in ans:
			prototypes[i] = j  # dictionary storing w.x values and an index corresponding to it as its keys
			i = i + 1
		sorted_x = sorted(prototypes.items(),key=operator.itemgetter(1),reverse=True)  # sorting all the w.x values (decreasing order)
		

		R = []  # for storing all y corresponding to a document
		S = []  # for storing all Y - y corresponding to a document
		E = []  # contains set of 
		for z in reuters.categories(docs):
			val = topics[z]
			for item in sorted_x:
				if item[0] == val:
					r = ()
					r = (val,item[1])
					R.append(r) #set of relevant topics
					
		alphamatrix = np.zeros(shape=(90,90))
		flag = 0
		for indexes in sorted_x:
			flag = 0
			for smally in R:
				if indexes[0] == smally[0]:
					flag = 1
					break
			if flag == 0:
				s = ()
				s = (indexes[0],indexes[1])
				S.append(s)  # set of non relevant topics
		for rows1 in R:
			for rows2 in S:
				if rows1[1] <= rows2[1]:
					e = ()
					e = (rows1[0],rows2[0])
					E.append(e)   # Error set containing (r,s) where w(r).x <= w(s).x
		

		########### Loss Functions ############	
		
		
		if ChoiceOfLossFunction == 1:
			loss = IsErrLoss(E)  #IsErr Loss function
		else:
			if ChoiceOfLossFunction == 2:
				loss = ErrSetSizeLoss(E)   # ErrSetSize Loss Function
			else:
				if ChoiceOfLossFunction == 3:
					loss = OneErrorLoss(sorted_x,R)   #OneEroor Loss function
				else:
					if ChoiceOfLossFunction == 4:
						loss = AvgPLoss(R,sorted_x)  # AvgP Loss Function
					else:
						if ChoiceOfLossFunction == 5:
							loss = maxF1(sorted_x,R)   # MaxF1 Loss Function

		
		########### Update Functions ##########

		   
		if ChoiceOfUpdateFunction == 1:
			alphamatrix = uniform_update(R,S,alphamatrix,E)   # Uniform Update
		else:
			if ChoiceOfUpdateFunction == 2:
				alphamatrix = max_update(alphamatrix,R,S)   # Max Update
			else:
				if ChoiceOfUpdateFunction == 3:
					alphamatrix = update_margin(alphamatrix,R,S)   # Margin proportional update
				else:
					if ChoiceOfUpdateFunction == 4:
						alphamatrix = randomized_update(E,alphamatrix,R,S)   # Randomized update
		
	
		tau = np.zeros(shape=(len(Y))) 


		# Calculation for tau array of size 90  

		for row in R:
			sum = 0 
			for cols in range(len(alphamatrix[0])):
				sum = sum + alphamatrix[row[0]-1][cols]
			tau[row[0]-1] = sum * loss
			
				
		for row in S:
			sum = 0
			for cols in range(len(alphamatrix[0])):
				sum = sum - alphamatrix[cols][row[0]-1]
			tau[row[0]-1] = sum * loss# to be multiplied by loss

		# updation of prototypes

		p = np.dot(tau.reshape(tau.shape[0],1),rows.reshape(1,rows.shape[0]))
		w = w + p
	
	return w


########## Code for testing ############

def test(w,tfidfvalues,fileids):

	print('Testing...')
	val_avg=0.0

	for row,fileid in zip(tfidfvalues,fileids):
	
		ans = np.dot(w,row.reshape(row.shape[0],1))
		count=0
		types = {}
		i=1
		actual = []
		for zz in reuters.categories(fileid):
			count=count+1
			actual.append(topics[zz])
		
		for j in ans:
			types[i] = j
			i = i + 1
		top = []
		sorted_x = sorted(types.items(),key=operator.itemgetter(1),reverse=True)
		for xx in range(count):
			top.append(sorted_x[xx])
		
		count_x=0
		for a in actual:
			flag=0
			for b in top:
				if b[0]==a:
					flag=1
					break
			if flag==1:
				count_x=count_x+1
		val_avg = val_avg + float(count_x)/float(count)        
	 
	val_avg = val_avg/len(fileids)
	
	return val_avg

#### Main Code begins ####

ChoiceOfUpdateFunction = 5

while ChoiceOfUpdateFunction <=0 or ChoiceOfUpdateFunction >= 5: 
	print('Enter Choice(1-4) of Update Function to be used from below Update Functions:')
	print('1. Uniform Update')
	print('2. Max Update')
	print('3. Margin Proportional Update')
	print('4. Randomized Update')
	ChoiceOfUpdateFunction = input()

ChoiceOfLossFunction = 6

while ChoiceOfLossFunction <=0 or ChoiceOfLossFunction >= 6:
	print('Enter Choice(1-5) of Loss Function to be used from below Loss Functions:')
	print('1. IsErr Loss')
	print('2. ErrSetSize Loss')
	print('3. OneErr Loss')
	print('4. AvgP Loss')
	print('5. maxF1 Loss')
	ChoiceOfLossFunction = input()


X = reuters.fileids(); #all the fileids, type = list
Y = reuters.categories(); #all the topics, i.e. 90, type = list
k =  len(Y); # k = 90

training_x = []
testing_x = []

# Creating dataset for training and testing 

for i in range(len(X)):
	if("train" in X[i]):
		training_x.append(X[i])
	else:
		testing_x.append(X[i])

#print(len(X), len(training_x), len(testing_x)) #(10788, 7769, 3019)

stop_words = set(stopwords.words('english'))   # set containing all stop words from english dictionary
ps = PorterStemmer()  
mainDict = {}
uniqueWordsDict = {}  # Dictionary containing all unique words as keys

# Performing preprocessing

for docs in training_x:
	training_words_list = reuters.words(docs)

	lower_training_words_list = []
	
	for text in training_words_list:
		lower = text.lower()
		if lower not in stop_words:
			if lower not in string.punctuation:
				lower_training_words_list.append(lower)

	#now we have words in lower_training_words_list after removing stopwords and punctuation

	training_words_dict ={}
	for word in lower_training_words_list:
		if ps.stem(word) in training_words_dict.keys():
			training_words_dict[ps.stem(word)] += 1
		else:
			training_words_dict[ps.stem(word)] = 1

		uniqueWordsDict[ps.stem(word)] = 1

	mainDict[docs] = training_words_dict

# creating of BagOfWords
# finding unique word 
# print(len(uniqueWordsDict)) #i.e. len = 19735

BagOfWords = np.zeros( shape=(len(training_x),len(uniqueWordsDict)) )

i = 0
for doc in training_x:
	j = 0
	for words in uniqueWordsDict.keys():
		if words in mainDict[doc].keys():
			BagOfWords[i][j] = mainDict[doc][words]
		j += 1
	i += 1

print('Creating tf-idf for training data...')

tf = TfidfTransformer(smooth_idf = True)
tfidf = tf.fit_transform(BagOfWords)
tfidf = tfidf.toarray()


i = 1
topics = {}

for row in reuters.categories():
	topics[row] = i  # dictionary storing the topics as key and an index value(1-90) corresponding to it
	i = i + 1 

# for testing data


TestmainDict = {}

for docs in testing_x:
	testing_words_list = reuters.words(docs)

	lower_testing_words_list = []
	
	for text in testing_words_list:
		lower = text.lower()
		if lower not in stop_words:
			if lower not in string.punctuation:
				lower_testing_words_list.append(lower)

	#now we have words in lower_testing_words_list after removing stopwords and punctuation

	testing_words_dict ={}
	for word in lower_testing_words_list:
		if ps.stem(word) in testing_words_dict.keys():
			testing_words_dict[ps.stem(word)] += 1
		else:
			testing_words_dict[ps.stem(word)] = 1

	TestmainDict[docs] = testing_words_dict

BagOfWordsTest = np.zeros( shape=(len(testing_x),len(uniqueWordsDict)) )
i = 0
for doc in testing_x:
	j = 0
	for words in uniqueWordsDict.keys():
		if words in TestmainDict[doc].keys():
			BagOfWordsTest[i][j] = TestmainDict[doc][words]
		j += 1
	i += 1

print('Creating tf-idf for testing data...')

tfidf_test = tf.fit_transform(BagOfWordsTest)
tfidf_test = tfidf_test.toarray()


############################## Training Begins ###############################


w = train(tfidf, training_x)

####### Accuracy #######

accuracy = test(w,tfidf_test,testing_x)

print("Accuracy:", accuracy*100)
