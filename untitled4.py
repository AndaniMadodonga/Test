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
import streamlit as st 
import altair as alt
from sklearn import preprocessing
import pandas as pd # to read csv/excel formatted data
import matplotlib.pyplot as plt # to plot graphs
import numpy as np
#import seaborn as sns
#import missingno
#%matplotlib inline
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
#from sklearn.cluster import KMeans
#from sklearn.cluster import MiniBatchKMeans
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
import nltk
import re
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
#from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from mlxtend.plotting import plot_confusion_matrix
import math

nltk.download('stopwords')
nltk.download('wordnet')

# Data Pre-processing
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

@st.cache()

#Influncer Category
def influncerModel(Final_Dataset):
    
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

  return df_normal.iloc[:,:5] #best_model_xgb

#SA and Global Category Data cleaning
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

#Data_Models=Data_Models.replace(r"_", "", regex=True)
  Data_Models=S.append(G,ignore_index=True)

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

  return  Data_Models #best_model

def clean_text():
  documents = []
  from nltk.stem import WordNetLemmatizer

  stemmer = WordNetLemmatizer()
  for tex in range(0, len(text)):
    # Remove all the special characters
      document = re.sub(r'\W', ' ', str(text[tex]))
    
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
return df_lem



#Sentiment Analysis
def Sent(Data_Models):
  from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
  analyser = SentimentIntensityAnalyzer()
  if len(Data_Models)==1:
    text_sent=Data_Models
  else:
    text_sent=Data_Models.clean_text

  scores_sent=[]
  for sentence in text_sent:
     score = analyser.polarity_scores(sentence)
     scores_sent.append(score)

  dfSentiment= pd.DataFrame(scores_sent)
  if len(Data_Models)==1:
        text_sent=pd.DataFrame(text_sent)
        Df_sent=pd.concat([text_sent,dfSentiment],axis=1)
  else:      
       Df_sent=pd.concat([text_sent,dfSentiment,Data_Models.Class],axis=1)
        
  Df_sent['sentiment_class']=''
  Df_sent.loc[Df_sent.compound>0,'sentiment_class']='positive'
  Df_sent.loc[Df_sent.compound==0,'sentiment_class']="Neutral"
  Df_sent.loc[Df_sent.compound<0,'sentiment_class']='Negative'
  if len(Data_Models)>1:  
    text_Sent_SA=Df_sent[Df_sent['Class']==1]
    text_Sent_GL=Df_sent[Df_sent['Class']==0]
    return text_Sent_SA, text_Sent_GL,Df_sent
  else: 
    return Df_sent["sentiment_class"].loc[0]


def main():
    import streamlit as st       
    # front end elements of the web page 
    html_temp1 = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:Center;;">South African and International Tweet Classification</h1> 
    </div> 
    """

    html_temp2 = """ 
    <div style ="background-color:red;padding:13px"> 
    <h1 style ="color:black;text-align:Center;">Tweet Sentiment Analysis</h1>
    </div> 
    """
    html_temp3 = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style =""color:black;text-align:Center;;"> Influencer's Classification</h1>
    </div> 
    """
    html_temp4 = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style =""color:black;text-align:Center;;">Choose task</h1>
    </div> 
    """
    #st.markdown(html_temp4, unsafe_allow_html = True) 
    st.sidebar.subheader("Choose Task to perform")
    
    # display the front end aspect
    
    
    #data = pd.read_excel(data_load)
#     data_load = st.file_picker("Choose a XLSX file",folder="https://github.com/AndaniMadodonga/Test/blob/main/",type="xlsx")
#     Label_list=labels["field_name"].values.tolist() 
    
    #data=""
    def Bulk_data(data_load):
        if data_load is not None:
            data = pd.read_excel(data_load)
            Label_list=['input_query','statuses_created_at','statuses_id','statuses_text','statuses_truncated','statuses_entities_user_mentions[0]_screen_name','statuses_entities_user_mentions[0]_name','statuses_entities_user_mentions[0]_id','statuses_entities_user_mentions[0]_id_str','statuses_entities_user_mentions[0]_indices[0]','statuses_metadata_iso_language_code','statuses_metadata_result_type','statuses_source','statuses_in_reply_to_status_id','statuses_in_reply_to_status_id_str','statuses_in_reply_to_user_id','statuses_in_reply_to_user_id_str','statuses_in_reply_to_screen_name','statuses_user_id','statuses_user_id_str','statuses_user_name','statuses_user_screen_name','statuses_user_location','statuses_user_description','statuses_user_url','statuses_user_entities_url_urls[0]_url','statuses_user_entities_url_urls[0]_expanded_url','statuses_user_entities_url_urls[0]_display_url','statuses_user_entities_url_urls[0]_indices[0]','statuses_user_entities_description_urls[0]_url','statuses_user_entities_description_urls[0]_expanded_url','statuses_user_entities_description_urls[0]_display_url','statuses_user_entities_description_urls[0]_indices[0]','statuses_user_protected','statuses_user_followers_count','statuses_user_friends_count','statuses_user_listed_count','statuses_user_created_at','statuses_user_favourites_count','statuses_user_statuses_count','statuses_user_profile_background_color','statuses_user_profile_background_image_url','statuses_user_profile_background_image_url_https','statuses_user_profile_background_tile','statuses_user_profile_image_url','statuses_user_profile_image_url_https','statuses_user_profile_banner_url','statuses_user_profile_link_color','statuses_user_profile_sidebar_border_color','statuses_user_profile_sidebar_fill_color','statuses_user_profile_text_color','statuses_user_profile_use_background_image','statuses_user_has_extended_profile','statuses_user_default_profile','statuses_user_default_profile_image','statuses_retweeted_status_created_at','statuses_retweeted_status_id','statuses_retweeted_status_id_str','statuses_retweeted_status_text','statuses_retweeted_status_truncated','statuses_retweeted_status_entities_urls[0]_url','statuses_retweeted_status_entities_urls[0]_expanded_url','statuses_retweeted_status_entities_urls[0]_display_url','statuses_retweeted_status_entities_urls[0]_indices[0]','statuses_retweeted_status_metadata_iso_language_code','statuses_retweeted_status_metadata_result_type','statuses_retweeted_status_source','statuses_retweeted_status_user_id','statuses_retweeted_status_user_id_str','statuses_retweeted_status_user_name','statuses_retweeted_status_user_screen_name','statuses_retweeted_status_user_location','statuses_retweeted_status_user_description','statuses_retweeted_status_user_url','statuses_retweeted_status_user_entities_url_urls[0]_url','statuses_retweeted_status_user_entities_url_urls[0]_expanded_url','statuses_retweeted_status_user_entities_url_urls[0]_display_url','statuses_retweeted_status_user_entities_url_urls[0]_indices[0]','statuses_retweeted_status_user_protected','statuses_retweeted_status_user_followers_count','statuses_retweeted_status_user_friends_count','statuses_retweeted_status_user_listed_count','statuses_retweeted_status_user_created_at','statuses_retweeted_status_user_favourites_count','statuses_retweeted_status_user_utc_offset','statuses_retweeted_status_user_verified','statuses_retweeted_status_user_statuses_count','statuses_retweeted_status_user_contributors_enabled','statuses_retweeted_status_user_is_translator','statuses_retweeted_status_user_is_translation_enabled','statuses_retweeted_status_user_profile_background_color','statuses_retweeted_status_user_profile_background_image_url','statuses_retweeted_status_user_profile_background_image_url_https','statuses_retweeted_status_user_profile_background_tile','statuses_retweeted_status_user_profile_image_url','statuses_retweeted_status_user_profile_image_url_https','statuses_retweeted_status_user_profile_banner_url','statuses_retweeted_status_user_profile_link_color','statuses_retweeted_status_user_profile_sidebar_border_color','statuses_retweeted_status_user_profile_sidebar_fill_color','statuses_retweeted_status_user_profile_text_color','statuses_retweeted_status_user_profile_use_background_image','statuses_retweeted_status_user_has_extended_profile','statuses_retweeted_status_user_default_profile','statuses_retweeted_status_user_default_profile_image','statuses_retweeted_status_retweet_count','statuses_retweeted_status_favorite_count','statuses_retweeted_status_favorited','statuses_retweeted_status_retweeted','statuses_retweeted_status_possibly_sensitive','statuses_retweeted_status_lang','statuses_is_quote_status',	'statuses_retweet_count',	'statuses_favorite_count','statuses_favorited',	'statuses_retweeted','statuses_lang']
            data.columns=Label_list
            predata=preprocess(data)
        
            return predata
        
    
        
        #predata=preprocess(data)
       
# #     #SA and Golbal tweets
    
    task1=st.sidebar.radio("Perform analysis",("Yes","No"))
    if task1=="Yes":
             task=st.sidebar.selectbox("tasks to Perform", ("<Select>","Categorise", "Sentiment", "Influencer"))
             if task=='Categorise':
                st.markdown(html_temp1, unsafe_allow_html = True )
                cat_choice=st.selectbox("Bulk or Text",("<Select>","Bulk", "Text"))
                if cat_choice=="Text":
                    result =""
                    keyin_text = st.text_input("type or paste a tweet")
            
                    if st.button("Categorise"):
                        if  len(keyin_text)<2:
                            st.error("type or paste a tweet")
                        else:
                            keyin_text=[keyin_text]
                            Keyin_text=clean_text(keyin_text)
                #pred_model=pred[0]
                ##insert pickle model
                            pred_result=pred_model.predict(keyin_text)
                            if pred_result == 1:
                                result = 'South African tweet'
                            else:
                                result = 'Global tweet'
                            st.success('The tweet falls under {}'.format(result))
                if cat_choice=="Bulk":
                        st.write("**Import XlSX file**")
                        data_load= st.file_uploader("Choose a XLSX file",type="xlsx")
                        if st.button('Perform Categorisation'):
                           if data_load is None:
                                st.error("Upload XLSX file")
                           else:
                                predata=Bulk_data(data_load)
                                clean_cat=CategoriseSA(predata)
                                #insert pickle model
                                categorise=classifier_SACat(clean_cat)
                                categorise=categorise.tolist()
                                df_class=pd.DataFrame(categorise,columns=["Class"])
                                df_cat=pd.DataFreame.concat([clean_cat,df_class],axis=1)
                                st.write(df_cat.head())
                                chart = alt.Chart(df_cat).mark_bar().encode(alt.X("Class"),y='count()').interactive()
                                st.write(chart)

        #
             if task=="Sentiment":
                    st.markdown(html_temp2, unsafe_allow_html = True) 
                    st.write("**Select the option below to perform bulk or Single tweet sentiment**")
                    sent_choice=st.selectbox("Bulk or text", ("<Select>","Bulk", "Text"))
                    if sent_choice=='Bulk':
                        st.write("**Import XlSX file**")
                        data_load= st.file_uploader("Choose a XLSX file",type="xlsx")
                        
                        if st.button('Check Bulk Sentiment'):
                            
                            if data_load is None:
                                st.error("Upload XLSX file")
                            else:
                                #pred_cat.head() 
                                predata=Bulk_data(data_load)
                                pred_cat=CategoriseSA(predata)
                                senti=Sent(pred_cat)
                                st.write(senti[2].head())
                                st.write("**SA tweet Sentiment analysis Bar graph**")
                                #st.write(senti[0].sentiment_class.value_counts().plot(kind='bar',color='green',title="sentiment analysis for SA tweets"))
                                #st.write(senti[1].sentiment_class.value_counts().plt(kind='bar',color='red',title="sentiment analysis for Global tweets"))
                                chart1 = alt.Chart(senti[0]).mark_bar().encode(alt.X("sentiment_class"),y='count()').interactive()
                                st.write(chart1)
                                st.write("**Global tweet Sentiment analysis Bar graph**")
                                chart2 = alt.Chart(senti[1]).mark_bar().encode(alt.X("sentiment_class"),y='count()').interactive()
                                st.write(chart2)
                            
                   
                    if sent_choice=='Text':
                        keyin_text_sent = st.text_input("type or paste a tweet")
                
                    
                        if st.button('Check Text Sentiment'):
                            senti=Sent([keyin_text_sent])  
                     
                            st.success('The Sentiment of the tweet is-{}'.format(senti))
                            
                
  #predata=preprocess(data)
        
             if task=="Influencer":
                    st.markdown(html_temp3, unsafe_allow_html = True)
                    data_load= st.file_uploader("Choose a XLSX file",type="xlsx")
                    
                    if st.button('Influencers'):
                       if data_load is None:
                            st.error("Upload XLSX File"):
                       else:
                        #st.write("import XLSX file")
                            data_load= st.file_uploader("Choose a XLSX file",type="xlsx")
                                
                            predata=Bulk_data(data_load)
                            influence_model=influncerModel(predata)
                #insert pickle model
                            inf_pred=classifier_pickle.predict(influence_model[1])
                            inf_pred=inf_pred.tolist() 
                            k=pd.DataFrame(inf_pred,columns=["Influencer_cat"])
                            k=k["Influencer_cat"].astype('category')
                        #st.bar_chart(k.value_counts())
                            chart2 = alt.Chart(k).mark_bar().encode(alt.X("Influencer_cat"),y='count()').interactive()
                            st.write(chart2)
if __name__ == '__main__':
    main()
