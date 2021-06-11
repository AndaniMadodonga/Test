#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 06:56:21 2021

@author: yolandankalashe
"""

# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import seaborn as sns
import streamlit as st 
import altair as alt
import base64
import pickle
import joblib
import emoji
from sklearn import preprocessing
import pandas as pd # to read csv/excel formatted data
import matplotlib.pyplot as plt # to plot graphs
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import nltk
import re
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import math
import datetime
from datetime import datetime, timezone
from googletrans import Translator
translator = Translator()
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import pickle
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.utils import simple_preprocess 
from gensim.models import CoherenceModel

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
class Full_Data:
      # Data Pre-processing

      def preprocess(self,data_heading):
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
          #translated="text_trans"
          translated = GoogleTranslator(source=data_translated['LangDet_code'].iloc[i], target='en').translate(to_translate) #Perform translation
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
      def influncerModel(self,Final_Dataset):
          
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

        xx = df_influencer.iloc[:, :6].values 
        
        cols=['statuses_retweeted_status_user_followers_count','statuses_retweeted_status_user_friends_count','statuses_user_statuses_count','statuses_retweeted_status_user_listed_count','statuses_retweeted_status_favorite_count','statuses_retweet_count']
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(xx)
        df_normal1 = pd.DataFrame(x_scaled,columns=cols)
        #df_normal=pd.concat([df_normal1,yy],axis=1)

        return df_normal1.iloc[:,:5] #best_model_xgb

      #SA and Global Category Data cleaning
      def CategoriseSA(self,Final_Dataset):
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

      def clean_text(self,text):
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
      def Sent(self,Data_Models):
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
      def csv_downloader(self,data):
            csvfile = data.to_csv()
            b64 = base64.b64encode(csvfile.encode()).decode()
            new_filename = "new_text_file.csv"
            st.markdown("#### Download File ###")
            href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
            st.markdown(href,unsafe_allow_html=True)

      def main_full(self):
          import streamlit as st       
          # front end elements of the web page 
          html_temp1 = """ 
          <div style ="background-color:yellow;padding:13px"> 
          <h1 style ="color:black;text-align:Center;;">South African and International Tweet Classification</h1> 
          </div> 
          """

          html_temp2 = """ 
          <div style ="background-color:green;padding:13px"> 
          <h1 style ="color:black;text-align:Center;">Tweet Sentiment Analysis</h1>
          </div> 
          """
          html_temp3 = """ 
          <div style ="background-color:orange;padding:13px"> 
          <h1 style =""color:black;text-align:Center;;"> Influencer's Classification</h1>
          </div> 
          """
          html_temp4 = """ 
          <div style ="background-color:blue;padding:13px"> 
          <h1 style =""color:black;text-align:Center;;">Choose task</h1>
          </div> 
          """
        
          #st.sidebar.subheader("Choose")
          
          #img = st.image('wordCloud2.png')
          #st.write(img)
          #data=""
          def load():
              data_load= st.file_uploader("Choose a XLSX file",type="xlsx")
              return data_load
          def Bulk_data(data_load):
              if data_load is not None:
                  data = pd.read_excel(data_load)
                  Label_list=['input_query','statuses_created_at','statuses_id','statuses_text','statuses_truncated','statuses_entities_user_mentions[0]_screen_name','statuses_entities_user_mentions[0]_name','statuses_entities_user_mentions[0]_id','statuses_entities_user_mentions[0]_id_str','statuses_entities_user_mentions[0]_indices[0]','statuses_metadata_iso_language_code','statuses_metadata_result_type','statuses_source','statuses_in_reply_to_status_id','statuses_in_reply_to_status_id_str','statuses_in_reply_to_user_id','statuses_in_reply_to_user_id_str','statuses_in_reply_to_screen_name','statuses_user_id','statuses_user_id_str','statuses_user_name','statuses_user_screen_name','statuses_user_location','statuses_user_description','statuses_user_url','statuses_user_entities_url_urls[0]_url','statuses_user_entities_url_urls[0]_expanded_url','statuses_user_entities_url_urls[0]_display_url','statuses_user_entities_url_urls[0]_indices[0]','statuses_user_entities_description_urls[0]_url','statuses_user_entities_description_urls[0]_expanded_url','statuses_user_entities_description_urls[0]_display_url','statuses_user_entities_description_urls[0]_indices[0]','statuses_user_protected','statuses_user_followers_count','statuses_user_friends_count','statuses_user_listed_count','statuses_user_created_at','statuses_user_favourites_count','statuses_user_statuses_count','statuses_user_profile_background_color','statuses_user_profile_background_image_url','statuses_user_profile_background_image_url_https','statuses_user_profile_background_tile','statuses_user_profile_image_url','statuses_user_profile_image_url_https','statuses_user_profile_banner_url','statuses_user_profile_link_color','statuses_user_profile_sidebar_border_color','statuses_user_profile_sidebar_fill_color','statuses_user_profile_text_color','statuses_user_profile_use_background_image','statuses_user_has_extended_profile','statuses_user_default_profile','statuses_user_default_profile_image','statuses_retweeted_status_created_at','statuses_retweeted_status_id','statuses_retweeted_status_id_str','statuses_retweeted_status_text','statuses_retweeted_status_truncated','statuses_retweeted_status_entities_urls[0]_url','statuses_retweeted_status_entities_urls[0]_expanded_url','statuses_retweeted_status_entities_urls[0]_display_url','statuses_retweeted_status_entities_urls[0]_indices[0]','statuses_retweeted_status_metadata_iso_language_code','statuses_retweeted_status_metadata_result_type','statuses_retweeted_status_source','statuses_retweeted_status_user_id','statuses_retweeted_status_user_id_str','statuses_retweeted_status_user_name','statuses_retweeted_status_user_screen_name','statuses_retweeted_status_user_location','statuses_retweeted_status_user_description','statuses_retweeted_status_user_url','statuses_retweeted_status_user_entities_url_urls[0]_url','statuses_retweeted_status_user_entities_url_urls[0]_expanded_url','statuses_retweeted_status_user_entities_url_urls[0]_display_url','statuses_retweeted_status_user_entities_url_urls[0]_indices[0]','statuses_retweeted_status_user_protected','statuses_retweeted_status_user_followers_count','statuses_retweeted_status_user_friends_count','statuses_retweeted_status_user_listed_count','statuses_retweeted_status_user_created_at','statuses_retweeted_status_user_favourites_count','statuses_retweeted_status_user_utc_offset','statuses_retweeted_status_user_verified','statuses_retweeted_status_user_statuses_count','statuses_retweeted_status_user_contributors_enabled','statuses_retweeted_status_user_is_translator','statuses_retweeted_status_user_is_translation_enabled','statuses_retweeted_status_user_profile_background_color','statuses_retweeted_status_user_profile_background_image_url','statuses_retweeted_status_user_profile_background_image_url_https','statuses_retweeted_status_user_profile_background_tile','statuses_retweeted_status_user_profile_image_url','statuses_retweeted_status_user_profile_image_url_https','statuses_retweeted_status_user_profile_banner_url','statuses_retweeted_status_user_profile_link_color','statuses_retweeted_status_user_profile_sidebar_border_color','statuses_retweeted_status_user_profile_sidebar_fill_color','statuses_retweeted_status_user_profile_text_color','statuses_retweeted_status_user_profile_use_background_image','statuses_retweeted_status_user_has_extended_profile','statuses_retweeted_status_user_default_profile','statuses_retweeted_status_user_default_profile_image','statuses_retweeted_status_retweet_count','statuses_retweeted_status_favorite_count','statuses_retweeted_status_favorited','statuses_retweeted_status_retweeted','statuses_retweeted_status_possibly_sensitive','statuses_retweeted_status_lang','statuses_is_quote_status',	'statuses_retweet_count',	'statuses_favorite_count','statuses_favorited',	'statuses_retweeted','statuses_lang']
                  data.columns=Label_list
                  predata=self.preprocess(data)  
                  return predata
            

          def Cat_Model():
              import joblib
              pred_model = joblib.load('classifier_SACat.pkl.pkl')
              return pred_model
          
          def Inf_Model():

              import xgboost as xgb
              pred_model = xgb.Booster()
              #xgb_model_latest = XGBClassifier()
              pred_model.load_model('xgb.bin')
              return pred_model
          
          #task1=st.sidebar.radio("Perform analysis",("Yes","No"))
          #if task1=="Yes":
          task=st.sidebar.selectbox("Prediction Type", ("<Select option>","Categorise", "Sentiment", "Influencer"))
          if task=='Categorise':
              st.markdown(html_temp1, unsafe_allow_html = True )
              cat_choice=st.selectbox("Bulk or Text",("<Select option>","Bulk", "Text"))
              if cat_choice=="Text":
                  result =""
                  keyin_text = st.text_input("type or paste a tweet")

                  if st.button("Categorise"):
                      if  len(keyin_text.split())<=2:
                          st.error("type or paste a tweet")
                      else:
                          keyin_text=[keyin_text]
                          keyin_text=self.clean_text(keyin_text)

                          pred_model=Cat_Model()                            
                          pred_result=pred_model.predict(keyin_text.clean_text)
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
                              clean_cat=self.CategoriseSA(predata)

                              pred_model=Cat_Model() 

                              categorise=pred_model.predict(clean_cat.statuses_without_stopwords)
                              categorise=categorise.tolist()
                              df_class=pd.DataFrame(categorise,columns=["Class_Label"])
                              df_class=df_class.reset_index(drop=True)
                              df_class['Tweet_Category'] = np.where((df_class['Class_Label'] ==0), 'Global Tweet', 'S.A Tweet')
                              df_cat=pd.concat([clean_cat,df_class],axis=1)
                              st.write("**Table of Tweet and Predicted Category**")
                              r=df_cat[['statuses_without_stopwords','Tweet_Category']]
                              r.rename(columns={'statuses_without_stopwords':'Tweet','Tweet_Category':'Predicted Category'},inplace=True)
                              st.write(r.head())
                              df_count=pd.DataFrame([len(df_cat[df_cat['Class_Label']==1]),len(df_cat[df_cat['Class_Label']==0])],columns=["Count"])
                              #columns=["SA Count","Global Count"]
                              df_count.index=["SA","Global"]
                              st.write("**Table of Tweet Categories Count**")
                              st.write(df_count)
#                               chart = alt.Chart(df_cat).mark_bar().encode(alt.X("Tweet_Category"),y='count()').interactive()
#                               st.write(chart)
                              ax = sns.countplot(y="Tweet_Category", data=df_cat)

                              for p in ax.patches:
                                 height = p.get_height() 
                                 width = p.get_width() 
                                 ax.text(x = width+3, 
                                 y = p.get_y()+(height/2),
                                 s = "{:.0f}".format(width), 
                                 va = "center")
                              st.pyplot()

      #
          if task=="Sentiment":
                  st.markdown(html_temp2, unsafe_allow_html = True) 
                  #st.write("**Select the option below to perform bulk or Single tweet sentiment**")
                  sent_choice=st.selectbox("Bulk or text", ("<Select option>","Bulk", "Text"))
                  if sent_choice=='Bulk':
                      st.write("**Import XlSX file**")
                      data_load= load()

                      if st.button('Check Bulk Sentiment'):

                          if data_load is None:
                              st.error("Upload XLSX file")
                          else:
                              #pred_cat.head() 
                              predata=Bulk_data(data_load)
                              pred_cat=self.CategoriseSA(predata)
                              senti=self.Sent(pred_cat)
                              
                              st.write("**Table of Tweet and Predicted Sentiment**")
                              st.write(senti[2][["clean_text","sentiment_class"]].head())
                              st.write("**SA tweet Sentiment analysis Bar graph**")
                              self.csv_downloader(senti[2][["clean_text","sentiment_class"]])
#                               chart1 = alt.Chart(senti[0]).mark_bar().encode(alt.X("sentiment_class"),y='count()').interactive()
#                               st.write(chart1)
                              ax = sns.countplot(y="sentiment_class", data=senti[0])

                              for p in ax.patches:
                                 height = p.get_height() 
                                 width = p.get_width() 
                                 ax.text(x = width+3, 
                                 y = p.get_y()+(height/2),
                                 s = "{:.0f}".format(width), 
                                 va = "center")
                              st.pyplot()
                              st.write("**Global Tweet Sentiment Analysis Bar Graph**")
#                               chart2 = alt.Chart(senti[1]).mark_bar().encode(alt.X("sentiment_class"),y='count()').interactive()
#                               st.write(chart2)
                              ax = sns.countplot(y="sentiment_class", data=senti[1])

                              for p in ax.patches:
                                 height = p.get_height() 
                                 width = p.get_width() 
                                 ax.text(x = width+3, 
                                 y = p.get_y()+(height/2),
                                 s = "{:.0f}".format(width), 
                                 va = "center")
                              st.pyplot()


                  if sent_choice=='Text':
                      keyin_text_sent = st.text_input("type or paste a tweet")


                      if st.button('Check Text Sentiment'):
                          text=self.clean_text([keyin_text_sent])
                          senti=self.Sent(text.clean_text)  

                          st.success('The Sentiment of the tweet is-{}'.format(senti))
                          st.write("**clean text:-** {}".format(text.clean_text.loc[0]))


#predata=preprocess(data)

          if task=="Influencer":
                  st.markdown(html_temp3, unsafe_allow_html = True)
                  data_load= load()

                  if st.button('Influencers'):
                    if data_load is None:
                          st.error("Upload XLSX File")
                    else:
                      #st.write("import XLSX file")

                          import xgboost as xgb   
                          predata=Bulk_data(data_load)
                          influence_model=self.influncerModel(predata)
              #insert pickle model
                          inf_model=Inf_Model()
                          t=xgb.DMatrix(influence_model)
                          inf_pred=inf_model.predict(t)
                          best_preds = np.asarray([np.argmax(line) for line in inf_pred])
                          #st.write(type(best_preds))
                          #st.write(best_preds)
                          best_preds=best_preds.tolist() 
                          k=pd.DataFrame(best_preds,columns=["Influencer_cat"])
                          k0=len(k[k['Influencer_cat']==0])
                          k1=len(k[k['Influencer_cat']==1])
                          k2=len(k[k['Influencer_cat']==2])
                          k3=len(k[k['Influencer_cat']==3])
                          k_count=pd.DataFrame([k0,k1,k2,k3],columns=['Count'])
                          choices     = [ "Mega Influence", 'Macro Influencer', 'Micro Influencer','Non influencer']  
                          k_count.index=choices
                          col=["Influencer_cat"]
                          conditions  = [ k[col] ==0,k[col]==1,k[col]==2,k[col]==3]
                          
                          k["Influencer_cat_label"] = np.select(conditions, choices, default=np.nan)
                          st.write("**Table of Influencer's Category Count**")
                          st.write(k_count)  
                          st.write('**Influencers categories Bar graph**')
#                           chart2 = alt.Chart(k).mark_bar().encode(alt.X("Influencer_cat_label"),y='count()').interactive()
#                           st.write(chart2)
                          ax = sns.countplot(y="Influencer_cat_label", data=k)

                          for p in ax.patches:
                             height = p.get_height() 
                             width = p.get_width() 
                             ax.text(x = width+3, 
                             y = p.get_y()+(height/2),
                             s = "{:.0f}".format(width), 
                             va = "center")
                          st.pyplot()
def preprocess_text(text):
          # Tokenise words while ignoring punctuation
          tokeniser = RegexpTokenizer(r'\w+')
          tokens = tokeniser.tokenize(text)
          
          # Lowercase and lemmatise 
          lemmatiser = WordNetLemmatizer()
          lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]
          
          # Remove stop words
          keywords= [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
          return keywords
                                

class SubSet_Data:
    
      def preprocess_text(self,text):
          # Tokenise words while ignoring punctuation
          tokeniser = RegexpTokenizer(r'\w+')
          tokens = tokeniser.tokenize(text)
          
          # Lowercase and lemmatise 
          lemmatiser = WordNetLemmatizer()
          lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]
          
          # Remove stop words
          keywords= [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
          return keywords

      def sub_df(self,data):
          data['statuses_retweeted_status_user_created_at'] =  pd.to_datetime(data['statuses_retweeted_status_user_created_at'])
          data['statuses_retweeted_status_user_created_at'] =  pd.to_datetime(data['statuses_retweeted_status_user_created_at'],format="%Y-%m-%d")
          
          
          text_List=[]
          Length_List=[]
          for i in data['statuses_text']:
              text_List.append(i)
              Length=len(str(i))
              Length_List.append(Length)
          
          followers_List=[]
          data['statuses_retweeted_status_user_followers_count']=data['statuses_retweeted_status_user_followers_count'].fillna(0)
          for i in data['statuses_retweeted_status_user_followers_count']:
              followers_List.append(i)
                        
          listedcount_List=[]
          data['statuses_retweeted_status_user_listed_count']=data['statuses_retweeted_status_user_listed_count'].fillna(0)
          for i in data['statuses_retweeted_status_user_listed_count']:
              listedcount_List.append(i)
        
          data['statuses_retweeted_status_user_description']=data['statuses_retweeted_status_user_description'].fillna(0)
          userdescription=[]
          for i in data['statuses_retweeted_status_user_description']:
              if(i==0):
                  userdescription.append('0')
              else:
                  userdescription.append('1')
      

          dat=[]
          T_d=[]
          Days_Active=[]
          for i in data['statuses_retweeted_status_user_created_at']:
              d1=i
              #Today = datetime.datetime.now()
              Today=pd.to_datetime('today')
              diff_Created=(Today.tz_localize(None)- d1.tz_localize(None))
              DaysActive=diff_Created.days
              Days_Active.append(DaysActive)
              dat.append(i)
              T_d.append(Today)

          fav_Count=[]
          data['statuses_retweeted_status_user_favourites_count']=data['statuses_retweeted_status_user_favourites_count'].fillna(0)
          for i in data['statuses_retweeted_status_user_favourites_count']:
              fav_Count.append(i)
              
          verif=[]
          data['statuses_retweeted_status_user_verified']=data['statuses_retweeted_status_user_verified'].fillna(0)
          for i in data['statuses_retweeted_status_user_verified']:
              verif.append(i)
              
          user_Statuses=[]
          data['statuses_retweeted_status_user_statuses_count']=data['statuses_retweeted_status_user_statuses_count'].fillna(0)
          for i in data['statuses_retweeted_status_user_statuses_count']:
              user_Statuses.append(i)
          
          has_Image=[]
          data['statuses_retweeted_status_user_profile_use_background_image']=data['statuses_retweeted_status_user_profile_use_background_image'].fillna(0)
          for i in data['statuses_retweeted_status_user_profile_use_background_image']:
              has_Image.append(i)

          data_tuples = list(zip(text_List,followers_List,listedcount_List,Length_List,fav_Count,verif,user_Statuses,has_Image,Days_Active,userdescription))

          MicroblogText_df=pd.DataFrame(data_tuples, columns=['Microblog_text','number_of_followers','number_of_times_listed','Length','fav_Count','user_verified','status_Count','has_image','DaysActive','has_decription'])
          MicroblogText_df['DaysActive']=MicroblogText_df['DaysActive'].fillna(0)
          
          return MicroblogText_df
          

      def Topic_num(self,corpus_df):
          corpus_df=corpus_df.to_list()
          Topic_ls=[]
          #global model
          model=self.model
          for i in range(len(corpus_df)):
              new_text_corpus =  model.id2word.doc2bow(corpus_df[i].split())
              Topic_array=model.get_document_topics(new_text_corpus)
              T_per=[]
                
              for i in range(len(corpus_df)):
                  new_text_corpus =  model.id2word.doc2bow(corpus_df[i].split())
                  Topic_array=model.get_document_topics(new_text_corpus)
                  
                  T_per=[]
                  
                  for i in range(3):
                      T1=Topic_array[i]
                      T1_per=T1[1]
                      T_per.append(T1_per)
                      
                  TP_num=T_per.index(max(T_per))

                  if TP_num==0:
                        TP_num="1"
                  elif TP_num==1:
                        TP_num="2"
                  elif TP_num==2:
                        TP_num="3"
                  
                  Topic_ls.append(TP_num)        

          return Topic_ls


      import vaderSentiment
      from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
      import re
        
      def Find(self,string):
          import re
          regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
          url = re.findall(regex,string)      
          return len(url)

      def Sentiment_url(self,corpus_df):
          import vaderSentiment
          from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer   
          corpus=corpus_df.to_list()
          sentences = corpus
          sentiment = list()
          url = list()
          
          for sentence in sentences:
              sent_vader = list(SentimentIntensityAnalyzer().polarity_scores(sentence).values())
              sentiment.append(sent_vader[3])
              hasUrl = self.Find(sentence)
              url.append(hasUrl)
                  
              sentiment_url= pd.DataFrame({'sentiment':sentiment,'urls':url})
              
              def A(row):
                  if(row['sentiment']>0):
                      val="Pos" 
                  elif(row['sentiment']<0): 
                      val="Neg"
                  else: 
                      val="neu"
                  return val
              
              def B(row):
                      if(row['sentiment']>0):
                          val="0"
                      elif(row['sentiment']<0):
                          val="1"
                      else: 
                          val="2"
              
                      return val
                  
              sentiment_url['Sentiment_Cat']=sentiment_url.apply(B,axis=1)
              sentiment_url['sentiment']=sentiment_url.apply(A,axis=1)
              
              
          return sentiment_url
      pickle_in = open("Topic1_classfier.pkl", "rb") 
      model=pickle.load(pickle_in)

      pickle_in = open("Trending_classfier.pkl", "rb") 
      Trending_model=pickle.load(pickle_in)

      pickle_in = open("Topic2_classfier.pkl", "rb") 
      Topic_m=pickle.load(pickle_in)
                
      
          

      




      def main_sub(self):
          
          
          
          
            st.subheader("User & Content Based Feature Table:")
          
          
          
            #Creating side bar to upload the file
            menu=["<Select option>","Bulk prediction","Single prediction"]
            choice=st.sidebar.selectbox("bulk/single prediction", menu)
            
            if choice=='Bulk prediction':
                Data_file=st.sidebar.file_uploader(label="Upload csv raw file", type=['xlsx'])
                
                if st.checkbox('Generate User & Content Based Feature Table'):
                   
                
                   st.header("User & Content Based Feature Table Modelling:")
                   st.subheader("Sub-Table based on input data")
     
                   data=pd.read_excel(Data_file)

                   sub_data=self.sub_df(data)
                   corpus=sub_data['Microblog_text']
                   Tp=self.Topic_num(corpus)
                 
                   Senti=self.Sentiment_url(corpus)
 
                   sub_data['Sentiment']=Senti["sentiment"]
                   sub_data['Sentiment_Cat']=Senti["Sentiment_Cat"]
                   sub_data['No_Urls']=Senti["urls"]
                   sub_data['Topic']=pd.DataFrame(Tp, columns={'Topic'})
                 
                   sub_data_pred=sub_data[['number_of_followers','number_of_times_listed','Length','fav_Count','user_verified','status_Count','has_image','DaysActive','Sentiment_Cat','No_Urls','Topic','has_decription']]
                   
                   st.dataframe(sub_data_pred)
                   
                   st.subheader("Overall Sentiment:")
                   
                   sent=sub_data['Sentiment']
                   face_det=(sent.value_counts()/len(sent))*100
                   
                   
                   emo=[]
                   for i in face_det:
                       if i=="Pos":
                           emoji_=emoji.emojize(':grinning_face_with_big_eyes:')
                           emo.append(emoji_)
                       elif i=="Neg":
                           emoji_=emoji.emojize(':neutral_face:')
                           emo.append(emoji_)
                       else:
                           emoji_=emoji.emojize(':angry_face:')
                           emo.append(emoji_)
                           
                   
         
                   st.write("{}:{}".format("Positive Sentiment"+emoji.emojize(':grinning_face_with_big_eyes:'),str(int(face_det[0]))+'%'))
                   st.write("{}:{}".format("Neutral Sentiment"+ emoji.emojize(':neutral_face:') ,str(int(face_det[1]))+'%'))
                   st.write("{}:{}".format("Negative Sentiment"+emoji.emojize(':angry_face:') ,str(int(face_det[2]))+'%'))
                         
                if st.checkbox('Predict hourly rate of transmission'):
                    
                    pred_cat=pd.DataFrame(self.Trending_model.predict(sub_data_pred))
                    pred_val=[]

                    for i in pred_cat[0]:
                        if i==0:
                           val='Trending'
                           pred_val.append(val)
                        else:
                            val='Wont Trending'
                            pred_val.append(val)
                                       
                    cf_lvl=pd.DataFrame(self.Trending_model.predict_proba(sub_data_pred))
                   
                    pred_cat['Pred category']=pred_cat[0]
                    pred_cat['Text']=sub_data['Microblog_text']
                    pred_cat['Projected Status']=pred_val
                    pred_cat['Confidence Level']=cf_lvl[0]
                   
                    st.write('Topic analysis/prediciton:')
                    st.write('Topology table')
                    pred_Topic=self.Topic_m.predict(sub_data['Microblog_text'])
                    topic_name=[]
                    
                    for i in pred_Topic:
                        if i=='0':
                          val='T_Vaccine'
                          topic_name.append(val)
                        elif i=='1':
                          val='T_Covid19'
                          topic_name.append(val)
                        else:
                          val='T_SA_lockdown'
                          topic_name.append(val)
                          
                    pred_Topic=pd.DataFrame(pred_Topic)
                    pred_Topic['Microblog']=sub_data['Microblog_text']
                    pred_Topic['Topic_Cat']=pred_Topic[0]
                    pred_Topic['Topic_Name']=topic_name
            
                    pred_Topic=pred_Topic[['Microblog','Topic_Cat','Topic_Name']]
            
                    st.dataframe(pred_Topic)
                    lamda_T1=[27,3,11,10,19,0,5,18,2,6,7,6,18,2,4,27,13,7,2,5,6,1,6,6,9,14,2,6,4,3,3,8,4,
                     3,4,4,2,1,10,3,1,1,1,1,1,3,2,8,2,2,4,1,1,9,3,2,2,3,2,2,1,1,2,5,5,7,8,3,13,
                     8,4,1]
                    lamda_T2=[0,1178,2,1,1,359,2,4,1,1,16,74,6,7,2,3,159,3,11,65,12,4,1,22,5,1,1,43,
                      1,63,2,5,3,1,1,314,357,16,10,21,4,12,2,2,1,3,61,32,1,2,1,1,1,2,3,1,11,
                      4,1,7,15,17,4,2,65,1,1,1,1,1,22,8]
            
                    lamda_T3=[3,3,0,1,3,3,1,1,1,1,1,1,4,1,1,1,6,2,1,1,1,4,1,3,1,1,1,3,1,1,1,1,2,1,1,1,
                      1,4,2,4,1,1,2,3,1,1,1,1,1,1,1,1,3,2,2,1,3,5,3,1,1,1,1,2,3,1,2,1,1,
                      1,2,1]

                    plt_dist=pd.DataFrame(lamda_T1,index=pd.RangeIndex(72, name='x'))
                    plt_dist['T1: #Covid']=plt_dist[0]
                    plt_dist['T2: #Vaccine']=lamda_T2
                    plt_dist['T3: #SA_Lockdown']=lamda_T3
                    plt_dist=plt_dist[['T1: #Covid','T2: #Vaccine','T3: #SA_Lockdown']]
                    plt1=plt_dist[['T1: #Covid']]
                    plt2=plt_dist[['T2: #Vaccine']]
                    plt3=plt_dist[['T3: #SA_Lockdown']]
                    
                    plt1 = plt1.reset_index().melt('x', var_name='category', value_name='y')
                
                    line_chart1= alt.Chart(plt1).mark_line(interpolate='basis').encode(
                          alt.X('x', title='hour'),
                          alt.Y('y', title='count of retweets'),
                          color='category:N'
                          ).properties(
                           title='Topic1'
                          )

                
                    plt2 = plt2.reset_index().melt('x', var_name='category', value_name='y')
                                
                    line_chart2= alt.Chart(plt2).mark_line(interpolate='basis').encode(
                          alt.X('x', title='hour'),
                          alt.Y('y', title='count of retweets'),
                          color='category:N'
                          ).properties(
                           title='Topic2'
                          )

                
            
                    plt3 = plt3.reset_index().melt('x', var_name='category', value_name='y')
                                                
                    line_chart3= alt.Chart(plt3).mark_line(interpolate='basis').encode(
                          alt.X('x', title='hour'),
                          alt.Y('y', title='count of retweets'),
                          color='category:N'
                          ).properties(
                           title='Topic3'
                          )   
                              
                    plt_dist = plt_dist.reset_index().melt('x', var_name='category', value_name='y') 
                    line_chart = alt.Chart(plt_dist).mark_line(interpolate='basis').encode(
                          alt.X('x', title='hour'),
                          alt.Y('y', title='count of retweets'),
                          color='category:N'
                          ).properties(
                           title='retweet count distribution in the first 72hours')
         
                    st.subheader('Expected distribution plot per topic')
            
            
                    st.subheader("Individual Distrubtion plot")
                    radi_distribu=st.radio('Show distribution plot', ['Combined plot','Topic1','Topic2','Topic3'])
                    if radi_distribu=='Combined plot':
                       st.subheader("Combined Distrubtion plot")
                       st.altair_chart(line_chart,use_container_width=True)
                
                    if radi_distribu=='Topic1':
                       st.subheader("Topic1 Distrubtion plot")
                       st.altair_chart(line_chart1,use_container_width=True)
            
                    if radi_distribu=='Topic2':
                       st.subheader("Topic2 Distrubtion plot")
                       st.altair_chart(line_chart2,use_container_width=True)
                
                    if radi_distribu=='Topic3':
                       st.subheader("Topic3 Distrubtion plot")
                       st.altair_chart(line_chart2,use_container_width=True)    
                       
                    st.subheader('Get probability of retweet count based on topic.')
                    
            
                    Count_tweet = st.slider('Count of Tweet',step=1, max_value=500)
                    hr_tweet=st.slider('hour since tweeted',max_value=72,step=1)
                    tweet_topic=st.slider('Topic Number',max_value=3,step=1)
                    from numpy import random
                    from scipy.stats import poisson
                    
                    if tweet_topic==1:
                        lambda_dist=lamda_T1
                        lambda_val=lambda_dist[hr_tweet]
                        prob=poisson.pmf(Count_tweet,lambda_val)
                
                        st.write('Probabilty of retweet count is:')
                        st.write(prob)
                
                    elif tweet_topic==2:
                        lambda_dist=lamda_T2
                        lambda_val=lambda_dist[hr_tweet]
                        prob=poisson.pmf(Count_tweet,lambda_val)
                
                        st.write('Probabilty of retweet count is:')
                        st.write(prob)
                        
                    elif tweet_topic==3:
                        lambda_dist=lamda_T3
                        lambda_val=lambda_dist[hr_tweet]
                        prob=poisson.pmf(Count_tweet,lambda_val)
                
                        st.write('Probabilty of retweet count is:')
                        st.write(prob)
                        
                    if st.checkbox('Predict probability microblog will trend'):
                       st.subheader("Likelihood of microblog trending:") 
                       st.write("Probability split:") 

                       st.dataframe(pred_cat)
  
  
            if choice=="Single prediction":
              
              st.sidebar.subheader("User Information:")
              number_of_followers=st.sidebar.number_input("number of followers",min_value=0, max_value=10000000,step=1)
              number_of_times_listed=st.sidebar.number_input("number of times listed",min_value=0, max_value=10000000,step=1)
              fav_Count=st.sidebar.number_input("fav Count",min_value=0, max_value=10000000, step=1)
              status_Count=st.sidebar.number_input("status Count",min_value=0, max_value=10000000, step=1)
              has_image=st.sidebar.number_input("has image (1-yes 0-No)",min_value=0,max_value=1, step=1)
              has_decription=st.sidebar.number_input("has decription (1-yes 0-No)",min_value=0,max_value=1, step=1)
              user_verified=st.sidebar.number_input(" Verified (1-yes 0-No)",min_value=0,max_value=1, step=1)
              Date_user_created=st.sidebar.date_input("Date user created")
              Microblog_text=st.sidebar.text_input("Microblog text")
              
              features={'number_of_followers': number_of_followers,
                  'number_of_times_listed': number_of_times_listed,'fav_Count': fav_Count,'user_verified':user_verified,
                  'status_Count':status_Count,'has_image': has_image, 'has_decription': has_decription,
                  'Date_user_created': Date_user_created, 'Microblog_text': Microblog_text,
                  
                  }
              
              data = pd.DataFrame([features])
              
                
              
              if st.checkbox('Generate User & Content Based Feature Table'):
                  
                  
                                  
                   st.header("User & Content Based Feature Table Modelling:")
                   st.subheader("Sub-Table based on input data")
                   
                   corpus=data['Microblog_text']
                   Tp=self.Topic_m.predict(data['Microblog_text'])
                 
                   Senti=self.Sentiment_url(corpus)
                   
                   Length=len(data['Microblog_text'])
                   
                   import datetime
              
                   data['Date_user_created'] =  pd.to_datetime(data['Date_user_created'])
                   data['Date_user_created'] =  pd.to_datetime(data['Date_user_created'],format="%Y-%m-%d")
                   
                   d1 = data['Date_user_created']
                   Today = datetime.datetime.now()

                   diff_Created=(Today - d1.min())
                   DaysActive=diff_Created.days
              
                   data['Sentiment']=Senti["sentiment"]
                   data['Sentiment_Cat']=Senti["Sentiment_Cat"]
                   data['No_Urls']=Senti["urls"]
                   data['Topic']=pd.DataFrame(Tp, columns={'Topic'})
                   data['Length']=Length
                   data['DaysActive']=DaysActive

                   
                    
                   sub_data_pred=data[['number_of_followers','number_of_times_listed','Length','fav_Count','user_verified','status_Count','has_image','DaysActive','Sentiment_Cat','No_Urls','Topic','has_decription']]
                   st.dataframe(data)
                   
                   st.subheader("Overall Sentiment:")
                   
                   
                   if data['Sentiment'][0]=='Pos':
                        st.write("{} : {}".format("Positive Sentiment",emoji.emojize(':grinning_face_with_big_eyes:') ))
                   elif data['Sentiment'][0]=='Neg':    
                        st.write("{} : {}".format("Negative Sentiment",emoji.emojize(':angry_face:') ))
                   else:    
                        st.write("{} : {}".format("Nuetral Sentiment",emoji.emojize(':neutral_face:') ))
              
            
              if st.checkbox('Predict hourly rate of transmission'):
                  
                  pred_cat=self.Trending_model.predict(sub_data_pred)
                  
                  st.write(pred_cat[0])
                  pred_val=""

                  if pred_cat==0:
                       pred_val ='Trending'

                  else:
                       pred_val='Wont Trending'

                                       
                  cf_lvl=pd.DataFrame(self.Trending_model.predict_proba(sub_data_pred))
                  s_clf=pd.Series(cf_lvl.max(axis=0))
                  pred_cat=pd.DataFrame(pred_cat) 
 
                  pred_cat=pd.DataFrame(data['Microblog_text'])
                  pred_cat['Confidence Level']=s_clf.max()
                  pred_cat['Projected Status']=pred_val                   
                    
                  st.write('Topic analysis/prediciton:')
                  st.write('Topology table')
                  pred_Topic=self.Topic_m.predict(data['Microblog_text'])
                  name=[]
                  

                  if pred_Topic=='0':
                        topic_name='T_Vaccine'
                        name.append(topic_name)
                  elif pred_Topic=='1':
                          topic_name='T_Covid19'
                          name.append(topic_name)
                  else:
                          topic_name='T_SA_lockdown'
                          name.append(topic_name)

                          
                  pred_Topic=pd.DataFrame(pred_Topic)
                  pred_Topic['Microblog']=data['Microblog_text']
                  pred_Topic['Topic_Cat']=pred_Topic[0]
                  pred_Topic['Topic_Name']=name[0]
                  
                  pred_Topic=pred_Topic[['Microblog','Topic_Cat','Topic_Name']]
                   
                  st.dataframe(pred_Topic)
                  
                  lamda_T1=[27,3,11,10,19,0,5,18,2,6,7,6,18,2,4,27,13,7,2,5,6,1,6,6,9,14,2,6,4,3,3,8,4,
                     3,4,4,2,1,10,3,1,1,1,1,1,3,2,8,2,2,4,1,1,9,3,2,2,3,2,2,1,1,2,5,5,7,8,3,13,
                     8,4,1]
                  lamda_T2=[0,1178,2,1,1,359,2,4,1,1,16,74,6,7,2,3,159,3,11,65,12,4,1,22,5,1,1,43,
                      1,63,2,5,3,1,1,314,357,16,10,21,4,12,2,2,1,3,61,32,1,2,1,1,1,2,3,1,11,
                      4,1,7,15,17,4,2,65,1,1,1,1,1,22,8]
            
                  lamda_T3=[3,3,0,1,3,3,1,1,1,1,1,1,4,1,1,1,6,2,1,1,1,4,1,3,1,1,1,3,1,1,1,1,2,1,1,1,
                      1,4,2,4,1,1,2,3,1,1,1,1,1,1,1,1,3,2,2,1,3,5,3,1,1,1,1,2,3,1,2,1,1,
                      1,2,1]
                  
                  
                  
                  plt_dist=pd.DataFrame(lamda_T1,index=pd.RangeIndex(72, name='x'))
                  plt_dist['T1: #Covid']=plt_dist[0]
                  plt_dist['T2: #Vaccine']=lamda_T2
                  plt_dist['T3: #SA_Lockdown']=lamda_T3
                  plt_dist=plt_dist[['T1: #Covid','T2: #Vaccine','T3: #SA_Lockdown']]
                  plt1=plt_dist[['T1: #Covid']]
                  plt2=plt_dist[['T2: #Vaccine']]
                  plt3=plt_dist[['T3: #SA_Lockdown']]
                    
                  plt1 = plt1.reset_index().melt('x', var_name='category', value_name='y')
                  
                
                  line_chart1= alt.Chart(plt1).mark_line(interpolate='basis').encode(
                          alt.X('x', title='hour'),
                          alt.Y('y', title='count of retweets'),
                          color='category:N'
                          ).properties(
                           title='Topic1'
                          )

                
                  plt2 = plt2.reset_index().melt('x', var_name='category', value_name='y')
                                
                  line_chart2= alt.Chart(plt2).mark_line(interpolate='basis').encode(
                          alt.X('x', title='hour'),
                          alt.Y('y', title='count of retweets'),
                          color='category:N'
                          ).properties(
                           title='Topic2'
                          )

                
            
                  plt3 = plt3.reset_index().melt('x', var_name='category', value_name='y')
                                                
                  line_chart3= alt.Chart(plt3).mark_line(interpolate='basis').encode(
                          alt.X('x', title='hour'),
                          alt.Y('y', title='count of retweets'),
                          color='category:N'
                          ).properties(
                           title='Topic3'
                          )   
                              
                  plt_dist = plt_dist.reset_index().melt('x', var_name='category', value_name='y') 
                  line_chart = alt.Chart(plt_dist).mark_line(interpolate='basis').encode(
                          alt.X('x', title='hour'),
                          alt.Y('y', title='count of retweets'),
                          color='category:N'
                          ).properties(
                           title='retweet count distribution in the first 72hours')
                              
                
                  st.subheader("Individual Distrubtion plot")
                  
                  st.write(pred_Topic['Topic_Cat'][0])
    
                  if pred_Topic['Topic_Cat'][0]=='1':
                       st.subheader("Topic1 Distrubtion plot")
                       st.altair_chart(line_chart1,use_container_width=True)
            
                  elif pred_Topic['Topic_Cat'][0]=='2':
                       st.subheader("Topic2 Distrubtion plot")
                       st.altair_chart(line_chart2,use_container_width=True)
                
                  elif pred_Topic['Topic_Cat'][0]=='3':
                       st.subheader("Topic3 Distrubtion plot")
                       st.altair_chart(line_chart3,use_container_width=True)    
                       
                  st.subheader('Get probability of retweet count based on topic.')
               
     
                  from numpy import random
                  from scipy.stats import poisson
                    
                  if pred_Topic['Topic_Cat'][0]=='1':
                        lambda_dist=lamda_T1
                        Count_tweet = st.slider('Count of Tweet',step=1, max_value=50)
                        hr_tweet=st.slider('hour since tweeted',max_value=72,step=1)
                        lambda_val=lambda_dist[hr_tweet]
                        prob=poisson.pmf(Count_tweet,lambda_val)
                
                        st.write('Probabilty of retweet count is:')
                        st.write(prob)
                
                  elif pred_Topic['Topic_Cat'][0]=='2':
                        lambda_dist=lamda_T2
                        Count_tweet = st.slider('Count of Tweet',step=1, max_value=1200)
                        hr_tweet=st.slider('hour since tweeted',max_value=72,step=1)
                        lambda_val=lambda_dist[hr_tweet]
                        prob=poisson.pmf(Count_tweet,lambda_val)
                
                        st.write('Probabilty of retweet count is:')
                        st.write(prob)
                        
                  elif pred_Topic['Topic_Cat'][0]=='3':
                        lambda_dist=lamda_T3
                        Count_tweet = st.slider('Count of Tweet',step=1, max_value=20)
                        hr_tweet=st.slider('hour since tweeted',max_value=72,step=1)
                        lambda_val=lambda_dist[hr_tweet]
                        prob=poisson.pmf(Count_tweet,lambda_val)
                
                        st.write('Probabilty of retweet count is:')
                        st.write(prob)
                        
                  if st.checkbox('Predict probability microblog will trend'):
                       st.subheader("Likelihood of microblog trending:") 
                       st.write("Probability split:") 

                       st.dataframe(pred_cat)
class Dash:
      def dash_full(self):
                def Bulk_data(data_load):
                    if data_load is not None:
                        #data = pd.read_excel(data_load)
                        Label_list=['input_query','statuses_created_at','statuses_id','statuses_text','statuses_truncated','statuses_entities_user_mentions[0]_screen_name','statuses_entities_user_mentions[0]_name','statuses_entities_user_mentions[0]_id','statuses_entities_user_mentions[0]_id_str','statuses_entities_user_mentions[0]_indices[0]','statuses_metadata_iso_language_code','statuses_metadata_result_type','statuses_source','statuses_in_reply_to_status_id','statuses_in_reply_to_status_id_str','statuses_in_reply_to_user_id','statuses_in_reply_to_user_id_str','statuses_in_reply_to_screen_name','statuses_user_id','statuses_user_id_str','statuses_user_name','statuses_user_screen_name','statuses_user_location','statuses_user_description','statuses_user_url','statuses_user_entities_url_urls[0]_url','statuses_user_entities_url_urls[0]_expanded_url','statuses_user_entities_url_urls[0]_display_url','statuses_user_entities_url_urls[0]_indices[0]','statuses_user_entities_description_urls[0]_url','statuses_user_entities_description_urls[0]_expanded_url','statuses_user_entities_description_urls[0]_display_url','statuses_user_entities_description_urls[0]_indices[0]','statuses_user_protected','statuses_user_followers_count','statuses_user_friends_count','statuses_user_listed_count','statuses_user_created_at','statuses_user_favourites_count','statuses_user_statuses_count','statuses_user_profile_background_color','statuses_user_profile_background_image_url','statuses_user_profile_background_image_url_https','statuses_user_profile_background_tile','statuses_user_profile_image_url','statuses_user_profile_image_url_https','statuses_user_profile_banner_url','statuses_user_profile_link_color','statuses_user_profile_sidebar_border_color','statuses_user_profile_sidebar_fill_color','statuses_user_profile_text_color','statuses_user_profile_use_background_image','statuses_user_has_extended_profile','statuses_user_default_profile','statuses_user_default_profile_image','statuses_retweeted_status_created_at','statuses_retweeted_status_id','statuses_retweeted_status_id_str','statuses_retweeted_status_text','statuses_retweeted_status_truncated','statuses_retweeted_status_entities_urls[0]_url','statuses_retweeted_status_entities_urls[0]_expanded_url','statuses_retweeted_status_entities_urls[0]_display_url','statuses_retweeted_status_entities_urls[0]_indices[0]','statuses_retweeted_status_metadata_iso_language_code','statuses_retweeted_status_metadata_result_type','statuses_retweeted_status_source','statuses_retweeted_status_user_id','statuses_retweeted_status_user_id_str','statuses_retweeted_status_user_name','statuses_retweeted_status_user_screen_name','statuses_retweeted_status_user_location','statuses_retweeted_status_user_description','statuses_retweeted_status_user_url','statuses_retweeted_status_user_entities_url_urls[0]_url','statuses_retweeted_status_user_entities_url_urls[0]_expanded_url','statuses_retweeted_status_user_entities_url_urls[0]_display_url','statuses_retweeted_status_user_entities_url_urls[0]_indices[0]','statuses_retweeted_status_user_protected','statuses_retweeted_status_user_followers_count','statuses_retweeted_status_user_friends_count','statuses_retweeted_status_user_listed_count','statuses_retweeted_status_user_created_at','statuses_retweeted_status_user_favourites_count','statuses_retweeted_status_user_utc_offset','statuses_retweeted_status_user_verified','statuses_retweeted_status_user_statuses_count','statuses_retweeted_status_user_contributors_enabled','statuses_retweeted_status_user_is_translator','statuses_retweeted_status_user_is_translation_enabled','statuses_retweeted_status_user_profile_background_color','statuses_retweeted_status_user_profile_background_image_url','statuses_retweeted_status_user_profile_background_image_url_https','statuses_retweeted_status_user_profile_background_tile','statuses_retweeted_status_user_profile_image_url','statuses_retweeted_status_user_profile_image_url_https','statuses_retweeted_status_user_profile_banner_url','statuses_retweeted_status_user_profile_link_color','statuses_retweeted_status_user_profile_sidebar_border_color','statuses_retweeted_status_user_profile_sidebar_fill_color','statuses_retweeted_status_user_profile_text_color','statuses_retweeted_status_user_profile_use_background_image','statuses_retweeted_status_user_has_extended_profile','statuses_retweeted_status_user_default_profile','statuses_retweeted_status_user_default_profile_image','statuses_retweeted_status_retweet_count','statuses_retweeted_status_favorite_count','statuses_retweeted_status_favorited','statuses_retweeted_status_retweeted','statuses_retweeted_status_possibly_sensitive','statuses_retweeted_status_lang','statuses_is_quote_status',	'statuses_retweet_count',	'statuses_favorite_count','statuses_favorited',	'statuses_retweeted','statuses_lang']
                        data_load.columns=Label_list
                        #predata=self.preprocess(data)  
                        return data_load
                url="https://github.com/AndaniMadodonga/Test/blob/main/Tweetdatatest%20-%20Copy.xlsx?raw=true"
                
                Data_file = pd.read_excel(url)
                
                if Data_file is None:
                   st.write("Check the data reference link ")
                else:
                  st.subheader("Visuals based on Trained data")
                   #Full_Data().main_full()


                  data=Bulk_data(Data_file)

                  data_processed=Full_Data().preprocess(data)

                  #fin_data=Full_Data().
                  st.write(data_processed.head())
                  hashbar = st.checkbox("Generate hashtags vs Topics",value = False)
                  if hashbar:
                    st.write( "**Hash_Tags vs Topics Bar Graph**") 
                    
                    sns.set(rc={"figure.figsize":(10,5)})
                    ax = sns.countplot(y="input_query", data=data_processed)
 
                    for p in ax.patches:
                         height = p.get_height() 
                         width = p.get_width() 
                         ax.text(x = width+3, 
                         y = p.get_y()+(height/2),
                         s = "{:.0f}".format(width), 
                         va = "center")
                    st.pyplot()   
                    
                  wordclo = st.checkbox("Generate WordCloud",value = False)
                  if wordclo:
                     from wordcloud import WordCloud
                     import matplotlib.pyplot as plt
                     st.write("**Tweets WordCloud**")
                     data_processed_text=Full_Data().clean_text(data_processed.statuses_text)

                     
                     st.set_option('deprecation.showPyplotGlobalUse',False)  
                     text=" ".join(clean_text for clean_text in data_processed_text.clean_text)
                     wordcloud = WordCloud(max_words=100).generate(text)

# Display the generated image:
                     plt.imshow(wordcloud, interpolation='bilinear')
                     plt.axis("off")
                     plt.title('Prevalent words in Tweets')
                     plt.show()
                     st.pyplot() 
                  cat_check=st.checkbox("Generate SA vs Global Bar Graph",value = False)
                  if cat_check:
                        def Cat_Model():
                            import joblib
                            pred_model = joblib.load('classifier_SACat.pkl.pkl')
                            return pred_model
                        clean_cat=Full_Data().CategoriseSA(data_processed)
                        pred_model=Cat_Model() 
                        categorise=pred_model.predict(clean_cat.statuses_without_stopwords)
                        categorise=categorise.tolist()
                        df_class=pd.DataFrame(categorise,columns=["Class_Label"])
                        df_class=df_class.reset_index(drop=True)
                        df_class['Tweet_Category'] = np.where((df_class['Class_Label'] ==0), 'Global Tweet', 'S.A Tweet')
                        df_cat=pd.concat([clean_cat,df_class],axis=1)
                        st.write('**SA vs Global Bar Graph**')
                          
                        ax = sns.countplot(y="Tweet_Category", data=df_cat)

                        for p in ax.patches:
                             height = p.get_height() 
                             width = p.get_width() 
                             ax.text(x = width+3, 
                             y = p.get_y()+(height/2),
                             s = "{:.0f}".format(width), 
                             va = "center")
                        st.pyplot()

def main():
    
  st.sidebar.header('Model and Visualization Selection')
  pick=["","Visualization/Dashboard","Prediction"]
  choice=st.sidebar.selectbox("Select Page to view", pick)
  
  if choice=='Prediction':
    st.title("Covid19za Consortium")
    st.subheader("Analysis and Predictor Models for Covid19 Microblog data ")
    st.write("This app uses a microblog, twitter data to help identify communication straegy for health and government officials during pandemics in social platforms. The dataset used contains user, retweet and microblog content information. ")
  
    st.write("The app will:")
    st.write('1. Predict if microblog text is SA or International, and identify sentiment between the two clusters.')
    st.write('2. Predict influncer status of user')
    st.write('3. Identify distribution of microblog based on predicted topic')
    st.write('4. Predict if Microblog will trend.')
  
    st.markdown('The first two predictions uses original dataset for predicitons whilst the rest uses the subset   **select data to achieve required prediction**.')  
  
    st.sidebar.subheader("SELECT DATA TO USE")
    data_option=st.sidebar.selectbox("Data Option",("<Select Option>","Original Dataset","Sub Dataset"))
    if data_option=="Original Dataset":
        Full_Data().main_full()
    
    elif data_option=='Sub Dataset':
        SubSet_Data().main_sub()
        

      
  elif choice=="Visualization/Dashboard":
          Dash().dash_full()
    

if __name__=='__main__':
          main()
