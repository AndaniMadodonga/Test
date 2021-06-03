# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YesCrfmMWHUNCPXncm_iwio8JJUs-UtF
"""


# Commented out IPython magic to ensure Python compatibility.

#!pip install streamlit
#!pip install deep-translator
#!pip install langdetect

# Commented out IPython magic to ensure Python compatibility.
#
import altair as alt
from sklearn import preprocessing
import pandas as pd # to read csv/excel formatted data
import matplotlib.pyplot as plt # to plot graphs
import numpy as np
#import seaborn as sns
#import missingno
# %matplotlib inline
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import nltk
import re
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
#from sklearn.svm import LinearSVC
#from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
import math

nltk.download('stopwords')
nltk.download('wordnet')




def dataload():
  drive.mount('/content/drive')
  data = pd.read_excel("/content/drive/Shareddrives/Capstone Project/EDA_sample4.xlsx")
  data=data.sample(n=800,random_state=7)
  labels = pd.read_excel("/content/drive/Shareddrives/Capstone Project/23_03_2021_Library.xlsx")
  Label_list=labels["field_name"].values.tolist() 
  data_heading=data
  data_heading.columns=Label_list

  return data_heading




def preprocess(data_heading):
  null_dict={}
  nulls_all=data_heading.isnull().sum().to_frame()
  for index, row in nulls_all.iterrows():
    if row[0]>0:
    #print(index, row[0])
      null_dict[index]=row[0]
  nulls_frame = pd.DataFrame(null_dict.items(),columns=['tweet_head','nul'])

  nulls_frame['nul_perc']= (nulls_frame['nul']/len(data_heading['input_query']))*100 # convert sum of null values into %
  nul_col=[] # To store  columns name
  nul_percent=[] # To store percentages

  for i,j in zip(nulls_frame['nul_perc'],nulls_frame['tweet_head']):
    if i>=75: 
      nul_col.append(j)
      nul_percent.append(i)
      del data_heading[j]
  data_non_eng=data_heading[data_heading['statuses_metadata_iso_language_code']!='en']
  data_en=data_heading[data_heading['statuses_metadata_iso_language_code']=='en']



  lang_code=[] # to store detected language iso code
  for i in data_non_eng['statuses_text']:

    try:
      lang = detect(i) # performing language detection
    except:
      lang="und" # put error if language is undetected/or cell is empty[any case]
    
    lang_code.append(lang)

  data_non_eng['LangDet_code']=lang_code #Create a column of the newly detected language iso code



#Check if the detected iso code are the same as the iso code on the data
  data_non_eng['Iso_Code_Comparison'] = np.where((data_non_eng['LangDet_code'] == data_non_eng['statuses_metadata_iso_language_code']), 'Yes', 'No')



  data_translated=data_non_eng[data_non_eng['LangDet_code']!='und'] #Exclude undetected
  data_translated=data_translated[data_translated['LangDet_code']!='en'] #Excluded detected as English



  data_translate_en=data_non_eng[data_non_eng['LangDet_code']=='en'] # Creat a dataframe of English detected languages [ to be merged with the other data after translation]
  data_translated=data_translated.reset_index(drop=True) #reset index



  translated_text=[] #To store translated text
  for i in data_translated['statuses_text'].index:
     to_translate = data_translated['statuses_text'].iloc[i]
     translated="text_trans"
    #translated = GoogleTranslator(source=data_translated['LangDet_code'].iloc[i], target='en').translate(to_translate) #Perform translation
     translated_text.append(translated) #Append translated text

  data_translated['Translated_tweet']=translated_text # Translated text Column 
  data_translated['statuses_text']=data_translated['Translated_tweet'].values # Overide non-english tweets with translated tweets
  data_translated['statuses_metadata_iso_language_code']=data_translated['LangDet_code'].values #overide iso code with the detected iso code

  del data_translated['Translated_tweet'] # delete additional column[since it has overriden 'statuses_text']
  Translated_final=data_translated.append(data_translate_en,ignore_index = True) # Join English detected and Translated datasets

#delete duplicating and non used columns on the dataset
  del Translated_final['Iso_Code_Comparison'] #delete added column from the dataset
  del Translated_final['LangDet_code'] #delete added column from the dataset

  Final_Dataset=data_en.append(Translated_final,ignore_index = True) # Join Originally English identified, Translated datasets and English detected Datasets

  return Final_Dataset


def TweetId(Final_Dataset):
  def C(row):
    if(row['statuses_retweeted_status_user_followers_count']>1000000):
       val="Mega Influence"
    
    elif(row['statuses_retweeted_status_user_followers_count']<1000000 and row['statuses_retweeted_status_user_followers_count']>40000):
       val="Macro Influencer"
      
    elif(row['statuses_retweeted_status_user_followers_count']<40000 and row['statuses_retweeted_status_user_followers_count']>2000):
      val="Micro Influencer"
  
    else:
      val="Non influencer"
        
    return val

  Final_Dataset['Influencer_Cat']=Final_Dataset.apply(C,axis=1)

  from collections import Counter
#status_user_id case--To determine the user with the highest retweets
  MyList = Final_Dataset['statuses_retweeted_status_user_id']
  c = Counter(MyList)
# status_id -- To determine the tweets with the highest retweets
  MyList_tweet = Final_Dataset['statuses_retweeted_status_id']
  k=Counter(MyList_tweet)

  DuplicateTweets=pd.DataFrame.from_dict(c, orient='index').reset_index() 
  D_Tweet_id=pd.DataFrame.from_dict(k, orient='index').reset_index()

  DuplicateTweets.columns=["statuses_retweeted_status_user_id","value"]
  D_Tweet_id.columns=["statuses_retweeted_status_id","value"]

#sort the values
  return D_Tweet_id




def influncerModel(Final_Dataset):

  df_influencer=Final_Dataset[['statuses_retweeted_status_user_followers_count','statuses_retweeted_status_user_friends_count','statuses_user_statuses_count','statuses_retweeted_status_user_listed_count','statuses_retweeted_status_favorite_count','statuses_retweet_count','Influencer_Cat']]
  df_influencer=df_influencer.fillna(0)

  df_influencer.loc[(df_influencer['Influencer_Cat']=='Mega Influence'), 'Influencer_class'] = 0
  df_influencer.loc[(df_influencer['Influencer_Cat']=='Macro Influencer'), 'Influencer_class'] = 1
  df_influencer.loc[(df_influencer['Influencer_Cat']=='Micro Influencer'), 'Influencer_class'] = 2
  df_influencer.loc[(df_influencer['Influencer_Cat']=='Non influencer'), 'Influencer_class'] = 3

  xx = df_influencer.iloc[:, :6].values 
  yy=df_influencer.iloc[:, 7:8]

  cols=['statuses_retweeted_status_user_followers_count','statuses_retweeted_status_user_friends_count','statuses_user_statuses_count','statuses_retweeted_status_user_listed_count','statuses_retweeted_status_favorite_count','statuses_retweet_count']
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(xx)
  df_normal1 = pd.DataFrame(x_scaled,columns=cols)
  df_normal=pd.concat([df_normal1,yy],axis=1)

  train_X, test_X, train_y, test_y  = train_test_split(df_normal.iloc[:,:5], df_normal.Influencer_class, test_size=0.2, random_state=1)

  train_X, X_val, train_y, y_val = train_test_split(train_X, train_y, test_size=0.2, random_state=1)

#train Xgboost
  import xgboost as xgb
  params = {
     'max_depth': 6,
     'objective': 'multi:softprob',
      'num_class': 4,
     'n_gpus': 0}
  pipe_xgb = Pipeline([('clf', xgb.XGBClassifier(**params))])

  parameters_xgb = {
          'clf__n_estimators':[30,40], 
          'clf__criterion':['entropy'], 
          'clf__min_samples_split':[15,20], 
          'clf__min_samples_leaf':[3,4]
     }

  grid_xgb = GridSearchCV(pipe_xgb,param_grid=parameters_xgb,scoring='f1_macro',cv=5,refit=True) 
  best_model_xgb = grid_xgb.fit(train_X,train_y)
  best_model_xgb.best_estimator_.get_params()['clf']
  best_pred_xgb=best_model_xgb.predict(test_X)

  import pickle
  pickle_out = open("classifier_xgb.pkl", mode = "wb") 
  pickle.dump(best_model_xgb, pickle_out) 
  pickle_out.close()
  return best_model_xgb

def CategoriseSA(Final_Dataset):
  Final_Dataset['statuses_text'] = Final_Dataset['statuses_text'].str.lower()
  Categorisation_dataset=Final_Dataset[(Final_Dataset['input_query']!='nfsas') & (Final_Dataset['input_query']!='#openthechurches')]

  HashTag_Covid=Categorisation_dataset[Categorisation_dataset['input_query']=='Covid']
  HashTag_Vaccine=Categorisation_dataset[Categorisation_dataset['input_query']=='vaccine']
  HashTag_SA=Categorisation_dataset[(Categorisation_dataset['input_query']=='#southafrica') | (Categorisation_dataset['input_query']=='South Africa')|(Categorisation_dataset['input_query']=='#SAlockdown')]

  C=HashTag_Covid['statuses_text'].str.contains('covid |vaccine| pandemic | corona| virus') 
  Hash_Covid=HashTag_Covid[C]

  V=HashTag_Vaccine['statuses_text'].str.contains('covid |vaccine| pandemic | corona| virus') 
  Hash_Vac=HashTag_Vaccine[V]

  S=HashTag_SA['statuses_text'].str.contains('covid |vaccine| pandemic | corona| virus') 
  Hash_SA=HashTag_SA[S]

  Hash_Relevant=Final_Dataset[(Final_Dataset['input_query']=='#covidvaccine') | (Final_Dataset['input_query']=='#VaccineforSouthAfrica')|(Final_Dataset['input_query']=='#VaccineRolloutSA')|(Final_Dataset['input_query']=='#vaccineSA')|(Final_Dataset['input_query']=='vaccine AND "South Africa"')]

  Hash_Vac=Hash_Vac.reset_index(drop=True)
  Hash_Covid=Hash_Covid.reset_index(drop=True)
  Hash_SA=Hash_SA.reset_index(drop=True)

  All_Covid_tweets=Hash_Relevant.append([Hash_SA,Hash_Vac,Hash_Covid],ignore_index=True)

  All_Covid_tweets['statuses_text'] = All_Covid_tweets['statuses_text'].str.replace(r'[^\w\s]+', '')

  All_Covid_tweets['statuses_text'] = All_Covid_tweets['statuses_text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
  All_Covid_tweets['statuses_text'] = All_Covid_tweets['statuses_text'].str.lower()

  from nltk.corpus import stopwords

  stop = stopwords.words('english')
  newStopWords = ['RT','rt','capricornfmnews']
  stop.extend(newStopWords)
  All_Covid_tweets['statuses_without_stopwords']=All_Covid_tweets['statuses_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

  A=All_Covid_tweets['statuses_text'].str.contains('south africa|southafrica|ramaphosa|mzansi|cyril|zuma|nzimande|eff|anc|zwelimkhize|mkhize|drzwelimkhize') 
  SA_tweets=All_Covid_tweets[A]
  global_tweets=All_Covid_tweets[~A]

  SA_tweets['Class']=1
  global_tweets['Class']=0

  S=SA_tweets[['statuses_without_stopwords','Class']]
  G=global_tweets[['statuses_without_stopwords','Class']]

  Data_Models=S.append(G,ignore_index=True)
#Data_Models=Data_Models.replace(r"_", "", regex=True)

  documents = []
  from nltk.stem import WordNetLemmatizer

  stemmer = WordNetLemmatizer()
  for tex in range(0, len(Data_Models)):
    # Remove all the special characters
      document = re.sub(r'\W', ' ', str(Data_Models.statuses_without_stopwords[tex]))
    
    # remove all single characters
      document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
      document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
      document = re.sub(r'\s+', ' ', document, flags=re.I)    
    # Lemmatization
      document = document.split()

      document = [stemmer.lemmatize(word) for word in document]
      document = ' '.join(document)
    
      documents.append(document)

#dataframe of clean text
  df_lem = pd.DataFrame(documents, columns=["clean_text"])
  Data_Models['clean_text']=df_lem['clean_text']

#split the data

  X_train, X_test, y_train, y_test  = train_test_split(Data_Models.clean_text, Data_Models.Class, test_size=0.2, random_state=1)

  #X_train, val_X, y_train, val_y = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

  pipe = Pipeline(steps=[('TF-IDF_vectorization',TfidfVectorizer()),('classifier', MultinomialNB())])

# Create space of candidate learning algorithms and their hyperparameters
  search_space = [{'classifier': [RandomForestClassifier()],
                 'classifier__n_estimators': [10, 100, 1000]}]

  scoring={'AUC':'roc_auc','accuracy':metrics.make_scorer(metrics.accuracy_score) }
  clf = GridSearchCV(pipe, search_space,scoring=scoring ,cv=10,n_jobs=-1,return_train_score=True,refit='AUC')
  best_model = clf.fit(X_train, y_train)

  #bestModelPred=best_model.predict(X_test)

  import pickle 
  pickle_out = open("classifier.pkl", mode = "wb") 
  pickle.dump(best_model, pickle_out) 
  pickle_out.close()

  return best_model, Data_Models




def Sent(Data_Models):
  text_sent=Data_Models.clean_text
  scores_sent=[]
  for sentence in text_sent:
     score = analyser.polarity_scores(sentence)
     scores_sent.append(score)

  dfSentiment= pd.DataFrame(scores_sent)
  Df_sent=pd.concat([text_sent,dfSentiment,Data_Models.Class],axis=1)
  Df_sent['sentiment_class']=''
  Df_sent.loc[Df_sent.compound>0,'sentiment_class']='positive'
  Df_sent.loc[Df_sent.compound==0,'sentiment_class']="Neutral"
  Df_sent.loc[Df_sent.compound<0,'sentiment_class']='Negative'
  text_Sent_SA=Df_sent[Df_sent['Class']==1]
  text_Sent_GL=Df_sent[Df_sent['Class']==0]
  return text_Sent_SA, text_Sent_GL



def main():
    import streamlit as st       
    # front end elements of the web page 
    html_temp1 = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:left;">Streamlit Tweet Classification</h1> 
    </div> 
    """

    html_temp2 = """ 
    <div style ="background-color:red;padding:13px"> 
    <h1 style ="color:black;text-align:Center;">Streamlit Tweet Sentiment</h1>
    </div> 
    """
    html_temp3 = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style =""color:black;text-align:right;">Streamlit Influencer</h1>
    </div> 
    """
    html_temp4 = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style =""color:black;text-align:right;">Choose task</h1>
    </div> 
    """
    st.markdown(html_temp4, unsafe_allow_html = True) 
    st.sidebar.subheader("Choose Task")
    task=st.sidebar.selectbox("Different tasks", ("categorise", "Sentiment", "Influencer"))
    # display the front end aspect
    st.markdown(html_temp1, unsafe_allow_html = True) 
    data=dataload()
    predata=preprocess(data)
    
    result =""
    #SA and Golbal tweets
    pred=CategoriseSA(predata)
    if task=='categorise':
      keyin_text = st.text_input("type or paste a tweet")
      if st.button("Categorise"):
        
        pred_model=pred[0]
        pred_result=pred_model.predict(keyin_text)
        if pred_result == 1:
          result = 'South African tweet'
        else:
          result = 'Global tweet'
        st.success('The tweet falls under {}'.format(result))

    data_model=pred[1]
    if task=="sentiment":
      if st.button('sentiment'):
         senti=Sent(data_model)
         st.bar_chart(senti[0].sentiment_class.value_counts())
         st.bar_chart(senti[1].sentiment_class.value_counts())
  #predata=preprocess(data)
    st.markdown(html_temp2, unsafe_allow_html = True) 
    if task=="influencer":
      influence_model=influncerModel(predata)
      inf_pred=influence_model.predict(predata.sample(n=100))
      st.bar_chart(inf_pred)
if __name__ == '__main__':
    main()
