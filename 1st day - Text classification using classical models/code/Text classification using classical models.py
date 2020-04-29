#!/usr/bin/env python
# coding: utf-8

# # Classical model

# #### Importing Libraries

# In[15]:


import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import classification_report


# #### Chose path

# In[2]:


train_dir = "C:\\Users\\boltuzamaki\\Desktop\\NLP\\aclImdb\\train"
save_path = "C:\\Users\\boltuzamaki\\Desktop\\NLP\\aclImdb"
test_dir = "C:\\Users\\boltuzamaki\\Desktop\\NLP\\aclImdb\\test"


# #### Prepare dataset in form of CSV

# In[3]:


# Function which direct change all txt files along with labels to csv files

review = []
texts_n = []

def txt_2_csv(path,type_s ="train"):
    path_save = type_s+".csv"
    os.chdir(train_dir)
    list1 = os.listdir()
    for l in list1:
        os.chdir(train_dir+"\\"+l)
        texts = os.listdir()
        
        for text in texts:
            f = open(text, 'r', encoding="utf-8")
            new = f.read()
            texts_n.append(new)
            review.append(l)
            f.close()
        os.chdir("..")   
    # dictionary of lists  
    dict = {'text': texts_n, 'review': review}  
    df = pd.DataFrame(dict) 
    os.chdir(save_path)
    # saving the dataframe 
    df.to_csv(path_save,index = None)     


# #### Loading data from csv

# In[4]:


# Making CSV for train and testing 

txt_2_csv(train_dir)
txt_2_csv(test_dir, type_s = "test")


# In[5]:


# Load train and text from created CSV

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# In[6]:


train_data.head()


# In[7]:


train_data["text"][1]


# ####  Plotting number of counts of each class

# In[8]:


my_tags = ['pos','neg']
plt.figure(figsize=(10,4))
train_data.review.value_counts().plot(kind='bar');


# #### Helper functions

# In[20]:


def Unigram_BOW(data):                                                                    # Convert to Unigram BOW model
    count_vectorizer = CountVectorizer()
    bag_of_words = count_vectorizer.fit_transform(data)
    return bag_of_words


def stopwords_fun(text):                                                                   # Remove stopwords
    stopword = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = []
    for w in word_tokens:
        if w not in stopword:
            filtered_text.append(w)
    filtered_text1 = ' '.join(filtered_text)        
    return filtered_text1


def Feature_extraction(grams = "Uni", tfidf = True,combined_data = None, data = "train"  ):  # Convert Bigram BOW models
    if grams == "Uni":
        count_vectorizer = CountVectorizer()
        bow = count_vectorizer.fit(combined_data)
        dataf = bow.transform(data) 
        if tfidf==True:
            from sklearn.feature_extraction.text import TfidfTransformer 
            transformer = TfidfTransformer()
            tf_bow = transformer.fit(dataf)
            datat = tf_bow.transform(dataf)
            
    if grams == "Bi":
        count_vectorizer = CountVectorizer(ngram_range=(1,2))
        bow = count_vectorizer.fit(combined_data)
        dataf = bow.transform(data) 
        if tfidf==True:
            from sklearn.feature_extraction.text import TfidfTransformer 
            transformer = TfidfTransformer()
            tf_bow = transformer.fit(dataf)
            datat = tf_bow.transform(dataf)
             
    if grams == "Tri":
        count_vectorizer = CountVectorizer(ngram_range=(1,3))
        bow = count_vectorizer.fit(combined_data)
        dataf = bow.transform(data)
        if tfidf==True:
            from sklearn.feature_extraction.text import TfidfTransformer 
            transformer = TfidfTransformer()
            tf_bow = transformer.fit(dataf)
            datat = tf_bow.transform(dataf)
            
    if tfidf == True:
        return datat
    if tfidf == False:
        return dataf

    
def stochastic_descent(Xtrain, Ytrain, Xtest):                                                   # Stocastic Gradient Decent  
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    parameters = {"penalty":['l2', 'l1', 'elasticnet'],
                  "max_iter":[5,10,15]
                  }
    clf =  GridSearchCV(SGDClassifier(), parameters)
    print ("SGD(Linear Support Vector Machine) Fitting")
    clf.fit(Xtrain, Ytrain)
    print ("SGD(Linear Support Vector Machine) Predicting")
    Ytest = clf.predict(Xtest)
    return Ytest,clf

def naive_bayes(Xtrain, Ytrain, Xtest):
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    print ("Naive Bayes Fitting")
    clf.fit(Xtrain, Ytrain)
    print ("SGD(Linear Support Vector Machine) Predicting")
    Ytest = clf.predict(Xtest)
    return Ytest,clf

def Logistic(Xtrain, Ytrain, Xtest):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    parameters = {"penalty":['l2', 'l1', 'elasticnet']
                  }
    clf =  GridSearchCV(LogisticRegression(), parameters)
    print ("Logistic Fitting")
    clf.fit(Xtrain, Ytrain)
    print ("SLogistic Predicting")
    Ytest = clf.predict(Xtest)
    return Ytest,clf
    
def accuracy(Ytrain, Ytest):                                                                       # Calculate accuracy
    assert (len(Ytrain)==len(Ytest))
    num =  sum([1 for i, word in enumerate(Ytrain) if Ytest[i]==word])
    n = len(Ytrain)  
    return (num*100)/n

def remove_punct(data):                                                                           # Remove punctuation marks
    from nltk.tokenize import word_tokenize
    token = word_tokenize(data)
    words = [word for word in token if word.isalpha()]
    words = ' '.join(words) 
    return words

def remove_html_tag(data):
    texts = [re.sub('<[^<]+?>', '', textl) for textl in data]
    return texts


# #### Make list of train and test text

# In[10]:


texts_train = train_data.text
text_X = remove_html_tag(texts_train)
texts_test = test_data.text
text_Y = remove_html_tag(texts_test)
review_train = train_data.review
review_test = test_data.review


# #### Remove stopwords

# In[11]:


text_train_withoutstop = []

for text in text_X:
    sen = stopwords_fun(text)
    sen = remove_punct(sen)
    text_train_withoutstop.append(sen) 


# In[12]:



text_test_withoutstop = []
for text in text_Y:
    sen = stopwords_fun(text)
    sen = remove_punct(sen)
    text_test_withoutstop.append(sen)


# #### Converting Categorical value to model feedable form change according to your need

# In[13]:


Y_test = [1 if x=="pos" else 0 for x in review_test ]
Y_train = [1 if x=="pos" else 0 for x in review_train]


# #### Joining test and train data fro creating BOW and Tfidf

# In[14]:


full_text = text_test_withoutstop + text_train_withoutstop


# 
# #### Results For unigram without Tfidf
# 

# In[21]:


print("For Unigram BOW without Tfidf -->\n")
Uni_Bow_Tr = Feature_extraction(grams = "Uni", tfidf = False,combined_data = full_text,data = text_train_withoutstop ) 
Uni_Bow_Te = Feature_extraction(grams = "Uni", tfidf = False,combined_data = full_text,data = text_test_withoutstop ) 

Ytest_uni,clf1 = stochastic_descent(Uni_Bow_Tr, Y_train, Uni_Bow_Te)
print('accuracy %s' % accuracy(Ytest_uni, Y_test))
print(classification_report(Y_test, Ytest_uni,target_names=my_tags),"\n\n\n")

Ytest_uni,clf1 = naive_bayes(Uni_Bow_Tr, Y_train, Uni_Bow_Te)
print('accuracy %s' % accuracy(Ytest_uni, Y_test))
print(classification_report(Y_test, Ytest_uni,target_names=my_tags),"\n\n\n")

Ytest_uni,clf1 = Logistic(Uni_Bow_Tr, Y_train, Uni_Bow_Te)
print('accuracy %s' % accuracy(Ytest_uni, Y_test))
print(classification_report(Y_test, Ytest_uni,target_names=my_tags),"\n")


# #### Results For Bigram without Tfidf

# In[31]:


print("For Bigram BOW without Tfidf -->\n")
Bi_Bow_Tr = Feature_extraction(grams = "Bi", tfidf = False,combined_data = full_text,data = text_train_withoutstop ) 
Bi_Bow_Te = Feature_extraction(grams = "Bi", tfidf = False,combined_data = full_text,data = text_test_withoutstop )

Ytest_bi,clf1 = stochastic_descent(Bi_Bow_Tr, Y_train, Bi_Bow_Te)
print('accuracy %s' % accuracy(Ytest_bi, Y_test))
print(classification_report(Y_test, Ytest_bi,target_names=my_tags),"\n\n\n")

Ytest_bi,clf1 = naive_bayes(Bi_Bow_Tr, Y_train, Bi_Bow_Te)
print('accuracy %s' % accuracy(Ytest_bi, Y_test))
print(classification_report(Y_test, Ytest_bi,target_names=my_tags),"\n\n\n")

Ytest_bi,clf1 = Logistic(Bi_Bow_Tr, Y_train, Bi_Bow_Te)
print('accuracy %s' % accuracy(Ytest_bi, Y_test))
print(classification_report(Y_test, Ytest_bi,target_names=my_tags),"\n")


# #### Results For trigram without Tfidf

# In[32]:


print("For Trigram BOW without Tfidf -->\n")
Tri_Bow_Tr = Feature_extraction(grams = "Tri", tfidf = False,combined_data = full_text,data = text_train_withoutstop )
Tri_Bow_te = Feature_extraction(grams = "Tri", tfidf = False,combined_data = full_text,data = text_test_withoutstop )

Ytest_tri,clf1 = stochastic_descent(Tri_Bow_Tr, Y_train, Tri_Bow_te)
print('accuracy %s' % accuracy(Ytest_tri, Y_test))
print(classification_report(Y_test, Ytest_tri,target_names=my_tags),"\n\n\n")

Ytest_tri,clf1 = naive_bayes(Tri_Bow_Tr, Y_train, Tri_Bow_te)
print('accuracy %s' % accuracy(Ytest_tri, Y_test))
print(classification_report(Y_test, Ytest_tri,target_names=my_tags),"\n\n\n")

Ytest_tri,clf1 = Logistic(Tri_Bow_Tr, Y_train, Tri_Bow_te)
print('accuracy %s' % accuracy(Ytest_tri, Y_test))
print(classification_report(Y_test, Ytest_tri,target_names=my_tags),"\n")


# #### Results For unigram with Tfidf

# In[33]:


print("For Unigram BOW with Tfidf -->\n")
Uni_Tf_Tr = Feature_extraction(grams = "Uni", tfidf = True,combined_data = full_text,data = text_train_withoutstop ) 
Uni_Tf_Te = Feature_extraction(grams = "Uni", tfidf = True,combined_data = full_text,data = text_test_withoutstop )

Ytest_uni,clf1 = stochastic_descent(Uni_Tf_Tr, Y_train, Uni_Tf_Te)
print('accuracy %s' % accuracy(Ytest_uni, Y_test))
print(classification_report(Y_test, Ytest_uni,target_names=my_tags),"\n\n\n")

Ytest_uni,clf1 = naive_bayes(Uni_Tf_Tr, Y_train, Uni_Tf_Te)
print('accuracy %s' % accuracy(Ytest_uni, Y_test))
print(classification_report(Y_test, Ytest_uni,target_names=my_tags),"\n\n\n")

Ytest_uni,clf1 = Logistic(Uni_Tf_Tr, Y_train, Uni_Tf_Te)
print('accuracy %s' % accuracy(Ytest_uni, Y_test))
print(classification_report(Y_test, Ytest_uni,target_names=my_tags),"\n")


# #### Results For bigram with Tfidf

# In[35]:


print("For Bigram BOW with Tfidf -->\n")
Bi_Tf_Tr = Feature_extraction(grams = "Bi", tfidf = True,combined_data = full_text,data = text_train_withoutstop ) 
Bi_Tf_Te = Feature_extraction(grams = "Bi", tfidf = True,combined_data = full_text,data = text_test_withoutstop ) 

Ytest_bi,clf1 = stochastic_descent(Bi_Tf_Tr, Y_train, Bi_Tf_Te)
print('accuracy %s' % accuracy(Ytest_bi, Y_test))
print(classification_report(Y_test, Ytest_bi,target_names=my_tags),"\n\n\n")

Ytest_bi,clf1 = naive_bayes(Bi_Tf_Tr, Y_train, Bi_Tf_Te)
print('accuracy %s' % accuracy(Ytest_bi, Y_test))
print(classification_report(Y_test, Ytest_bi,target_names=my_tags),"\n\n\n")

Ytest_uni,clf1 = Logistic(Bi_Tf_Tr, Y_train, Bi_Tf_Te)
print('accuracy %s' % accuracy(Ytest_bi, Y_test))
print(classification_report(Y_test, Ytest_bi,target_names=my_tags),"\n")


# #### Results For trigram with Tfidf

# In[36]:


print("For Trigram BOW with Tfidf -->\n")
Tri_Tf_Tr =Feature_extraction(grams = "Tri", tfidf = True,combined_data = full_text,data = text_train_withoutstop ) 
Tri_Tf_Te = Feature_extraction(grams = "Tri", tfidf = True,combined_data = full_text,data = text_test_withoutstop ) 


Ytest_tri,clf1 = stochastic_descent(Tri_Tf_Tr, Y_train, Tri_Tf_Te)
print('accuracy %s' % accuracy(Ytest_tri, Y_test))
print(classification_report(Y_test, Ytest_tri,target_names=my_tags),"\n\n\n")

Ytest_tri,clf1 = naive_bayes(Tri_Tf_Tr, Y_train, Tri_Tf_Te)
print('accuracy %s' % accuracy(Ytest_tri, Y_test))
print(classification_report(Y_test, Ytest_tri,target_names=my_tags),"\n\n\n")

Ytest_tri,clf1 = Logistic(Tri_Tf_Tr, Y_train, Tri_Tf_Te)
print('accuracy %s' % accuracy(Ytest_tri, Y_test))
print(classification_report(Y_test, Ytest_tri,target_names=my_tags),"\n")


# #### Custom data test

# In[42]:


Test = ["you are a fucking bad","you are amazing"]
test = Feature_extraction(grams = "Tri", tfidf = True,combined_data = full_text,data = Test )
clf1.predict(test)

