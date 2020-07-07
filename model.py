import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

df = pd.read_csv('emails.csv')

df.head()

df.shape

print(df.isna().sum())

print(df['spam'].value_counts)

sns.countplot(df['spam'])

#Function to remove punctuation and stop words
import string


#from nltk's list of stopwords in enlgish
STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


def process_text(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)

    
    return ' '.join([word for word in no_punc.split() if word.lower() not in STOPWORDS])

df['text']=df['text'].apply(process_text)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer= CountVectorizer()
message_bow = vectorizer.fit_transform(df['text'])


pickle.dump(vectorizer,open('transform.pkl','wb'))




from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(message_bow,df['spam'],test_size=0.20)

from sklearn.naive_bayes import MultinomialNB
nb= MultinomialNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import plot_roc_curve
plot_roc_curve(nb,X_test,y_test)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(nb,X_test,y_test)

from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5,shuffle=True)
print("Accuracy using Cross Validation is :",np.mean(cross_val_score(nb,message_bow,df['spam'],cv=kfold,scoring="accuracy"))*100," %")

filename = 'nlp_model.pkl'
pickle.dump(nb,open(filename,'wb'))
 
















