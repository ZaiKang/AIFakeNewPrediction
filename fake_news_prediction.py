import streamlit as st
from joblib import load
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Load random forest classifier model and tdidfVectorizer 
rf_loaded = load('random_forest_classifier.joblib')
tv_loaded = load('tfidfVectorizer.joblib')


# Check if stopwords are available and download if necessary
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Create a function to clean text
def processWord(script):
    # lower case 
    script = script.lower() 
    # remove anything with and within brackets
    script = re.sub('\[.*?\]','', script) 
    # removes any character not a letter, digit, or underscore
    script = re.sub('\\W',' ',script) 
    # removes any links starting with https
    script= re.sub('https?://\S+|www\.\S+','',script) 
    # removes anything with and within < >
    script = re.sub('<.*?>+','', script) 
    # removes any string with % in it
    script = re.sub('[%s]' % re.escape(string.punctuation), '', script)  
    # remove next lines
    script = re.sub('\n','',script)
    # removes any string that contains at least a digit with zero or more characters
    script = re.sub('\w*\d\w*','', script)
    #remove stopwords (split the script(text class) -> filter out stopwords ->  join the words)
    script = ' '.join([word for word in script.split() if word not in stop_words])
    return script

# Prediction function 
def news_prediction(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(processWord)
    new_x_test = new_def_test['text']
    new_tfidf_test = tv_loaded.transform(new_x_test)
    pred_rf = rf_loaded.predict(new_tfidf_test)
    
    if pred_rf[0] == 0:
        return "This is Fake News! Don't Listen what the kopitiam uncle and aunty say."
    else:
        return "The News seems to be True!"

# Streamlit application starts here 
def main():
    # Display the image at the top of the page
    st.image('fakenews.jpeg', use_column_width=True)
    # Title of web app
    st.title("Fake News Prediction System")
    st.write("""This app predicts if a news article contains Fake News or not. Just copy the news into the following box
            and click on the predict button.""")
    user_text = st.text_area("Enter or copy a news article to check if it's true or fake:", height=350)
   
    if st.button("Predict"):
        if user_text.strip():  # Check if the input text is not just empty or spaces
            news_pred = news_prediction(user_text)
            if news_pred.startswith("This is Fake News!"):  # Adjusted condition here
                st.error(news_pred, icon="🚨")
            else:
                st.success(news_pred)
                st.balloons()
        else:
            st.error("Please enter some text before analyzing the news article!")  # Updated error message for clarity

if __name__ == "__main__":
    main()
