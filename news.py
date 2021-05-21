# -*- coding: utf-8 -*-
"""
Created on Mon May 17 00:35:34 2021

@author: megha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
pip install textblob
from textblob import TextBlob
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,  GradientBoostingClassifier, ExtraTreesClassifier)
import xgboost as xgb
#pip install xgboost
#pip3.8 install xgboost
pip install plotly
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import iplot
true_news=pd.read_csv(r'C:\Users\megha\Downloads\True.csv')
fake_news=pd.read_csv(r'C:\Users\megha\Downloads\Fake.csv')
print(true_news.shape)
print(true_news.head)
print(fake_news.shape)
print(true_news.info())
print(fake_news.info())
fake_news['output']=0
true_news['output']=1
print(true_news.info())

#concatenate title and text of news into a new column news
fake_news['news']=fake_news['title']+fake_news['text']
fake_news=fake_news.drop(['title','text'], axis=1)
print(fake_news.info())

true_news['news']=true_news['title']+true_news['text']
true_news=true_news.drop(['title', 'text'], axis=1)

#rearrange columns

fake_news=fake_news[['subject', 'date', 'news','output']]
true_news=true_news[['subject','date','news','output']]
print(true_news.head)
print(true_news['date'])

fake_news['date'].value_counts()
#cleaning date column since it has links attached to it
print(fake_news['date'])

fake_news=fake_news[fake_news.date.str.contains("Januaru|February|March|April|May|June|July|August|September|October|November|December")]
print(fake_news['date'].value_counts())

true_news=true_news[true_news.date.str.contains("Januaru|February|March|April|May|June|July|August|September|October|November|December")]

#converting data type to date

fake_news['date']=pd.to_datetime(fake_news['date'])
true_news['date']=pd.to_datetime(true_news['date'])


#concatenating two dataframes

frame=[fake_news,true_news]
df=pd.concat(frame)
print(df)
print(df.info())

#EDA

# removing punctuations
df1=df.copy(deep=True)
df1
#df2=df.copy(deep=False)
#df2
def cleaning(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    textclean_news['news']= df1['news'].apply(lambda x:cleaning(x))
    return(textclean_news)
df1.head()

#remove stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop= stopwords.words('english')
df1['news'] = df1['news'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df1.head()
df1.info()
print(df1['subject'])

#visualization
#countplot for different types of news
import seaborn as sb
%matplotlib qt
chart=sb.countplot(x='subject',data=df1,orient='h')
chart.set_xticklabels(chart.get_xticklabels(),rotation=45)
plt.show()

#fake vs true
%matplotlib inline
g = sns.catplot(x="subject", col="output",   data=df1, kind="count",   height=4, aspect=2)

#Rotating the xlabels
g.set_xticklabels(rotation=45)

#count of fake news and true news
%matplotlib inline
f=sb.countplot(x='output',data=df1)
f.set_xticklabels(f.get_xticklabels(),rotation=45)
df1['output'].value_counts()

#defining polarity(which defines sentiment of the news), word count, length
df1['polarity'] = df1['news'].map(lambda text: TextBlob(text).sentiment.polarity)
df1['review_len'] = df1['news'].astype(str).apply(len)
df1['word_count'] = df1['news'].apply(lambda x: len(str(x).split()))
plt.figure(figsize = (20, 5))
plt.style.use('seaborn-white')
plt.subplot(131)
sns.distplot(df1['polarity'])
fig = plt.gcf() #fugure 5
plt.subplot(132)
sns.distplot(df1['review_len'])
fig = plt.gcf()
plt.subplot(133)
sns.distplot(df1['word_count'])# figure 6
fig = plt.gcf()

#n gram analysis of words
import sklearn
#from sklearn import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(df1['news'], 20)

for word, freq in common_words:
    print(word, freq)

#Creating the dataframe of word and frequency
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import iplot
df2 = pd.DataFrame(common_words, columns = ['news' , 'count'])

#Group by words and plot the sum
#f=sb.countplot(x='news',data=df2)
#f.set_xticklabels(f.get_xticklabels(),rotation=45)

df2.groupby('news').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in news')

#bigram
   
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

#Calling function and return only top 20 words
common_words = get_top_n_bigram(df1['news'], 20)

#Printing the word and frequency
for word, freq in common_words:
    print(word, freq)
    
#Creating the dataframe of word and frequency
df3 = pd.DataFrame(common_words, columns = ['news' , 'count'])

#Group by words and plot the sum
df3.groupby('news').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in news')

#wordcloud of fake news

pip install wordcloud
from wordcloud import WordCloud, STOPWORDS
text = true_news["news"]
wordcloud = WordCloud(width = 3000, height = 2000, background_color = 'black', stopwords = STOPWORDS).generate(str(text))
fig = plt.figure( figsize = (40, 30), facecolor = 'k', edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#time- series analysis

fake=fake_news.groupby(['date'])['output'].count()
fake=pd.DataFrame(fake)
 
true=true_news.groupby(['date'])['output'].count()
true=pd.DataFrame(true)
%matplotlib inline
import plotly.graph_objects as go
figi = go.Figure()
figi.add_trace(go.Scatter(  x=true.index,   y=true['output'], name='True', line=dict(color='blue'),    opacity=0.8))

figi.add_trace(go.Scatter(   x=fake.index,  y=fake['output'],   name='Fake',  line=dict(color='red'),    opacity=0.8))

figi.update_xaxes(  rangeslider_visible=True,    rangeselector=dict(   buttons=list([  dict(count=1, label="1m", step="month", stepmode="backward"),  dict(count=6, label="6m", step="month", stepmode="backward"),   dict(count=1, label="YTD", step="year", stepmode="todate"),    dict(count=1, label="1y", step="year", stepmode="backward"), dict(step="all")])))
        
    
figi.update_layout(title_text='True and Fake News',plot_bgcolor='rgb(248, 248, 255)',yaxis_title='Value')

figi.show()
#fig

#stemming and vectorization

df5=df1.copy()
df5=df5[['news']].reset_index(drop=True)
df5
df5.info()

stop_words=set(stopwords.words("english"))
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps=PorterStemmer()
import re

corpus = []
for i in range(0, len(df5)):
    news = re.sub('[^a-zA-Z]', ' ', df5['news'][i])
    news= news.lower()
    news = news.split()
    news = [ps.stem(word) for word in news if not word in stop_words]
    news = ' '.join(news)
    corpus.append(news)  
corpus[1]
    
#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(2,2))
# TF-IDF feature matrix
X= tfidf_vectorizer.fit_transform(df5['news'])
X.shape
y=df1['output']

#balancing the data
from collections import Counter
print(Counter(y))
class_count_0,class_count_1= df1['output'].value_counts()
class_0 = df1[df1['output'] == 0]  #minority
class_1 = df1[df1['output'] == 1] #majority
print(class_0.shape)
print(class_1.shape)
pip install imbalanced-learn
pip install imblearn
import imblearn
from imblearn.under_sampling import RandomUnderSampler
#sample majority classa
#df_class_1_under = class_1.sample(class_count_0)
#df_test_under = pd.concat([df_class_1_under, class_0])
#print('Random under-sampling:')
#print(df_test_under.output.value_counts())

df_new=df1.copy()
X=df_new.drop('output', axis=1)
Y= df_new['output']
print(Y)
rus = RandomUnderSampler(random_state=42, replacement=True)
X_rus, Y_rus = rus.fit_resample(X, Y)
df_rus = pd.concat([pd.DataFrame(X_rus), pd.DataFrame(Y_rus, columns=['output'])], axis=1)

print('imblearn over-sampling:')
print(df_rus.output.value_counts())

from sklearn.model_selection import train_test_split

y=df_rus
print(y)
y=df_rus.output
print(y.value_counts())
tfidf_vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(2,2))

X= tfidf_vectorizer.fit_transform(df_rus['news'])
X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#trying logistic regression= accuracy =98%

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, y_pred))

#NAIVE BAYES

from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
clf2 = MultinomialNB()
clf2.fit(X_train, y_train)

y_pred = clf2.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, y_pred))



#DECISION tREE
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
pip install pydotplus
feature=['news']
import pydotplus
%matplotlib qt
from sklearn.tree import plot_tree
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (2,2), dpi=500)
tree.plot_tree(clf)


# deep learning model -lstm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
encoder = Tokenizer(oov_token='OOV') #splitting larger body of text into smaller
encoder.fit_on_texts(corpus)

word_to_index = encoder.word_index
#encoding string to numbers
x = encoder.texts_to_sequences(corpus) #Word embeddings give us a way to use an efficient, dense representation in which similar words have a similar encoding. Importantly, you do not have to specify this encoding by hand. 
vocab_size = len(word_to_index)+1
vocab_size

x = pad_sequences(x, padding='post')
x.shape
from sklearn.model_selection import train_test_split
y=df1['output']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
model1 = Sequential()
model1.add(layers.Embedding(input_dim=vocab_size,     output_dim=10,       input_length=X_train.shape[1]))
model1.add(layers.GlobalAveragePooling1D())
model1.add(layers.Dense(8, activation='relu'))
model1.add(layers.Dense(1, activation='sigmoid'))
model1.compile(loss=BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
model1.summary()
callback = callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model1.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test)
  import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
cm = confusion_matrix(y_test, model1.predict_classes(X_test))
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=[0, 1])
disp = disp.plot(include_values=True, ax=None, xticks_rotation='horizontal', cmap=plt.cm.Blues)
plt.show()


#XGB classifier

pip install xgboost
from xgboost import XGBClassifier
from xgboost import XGBClassifier
