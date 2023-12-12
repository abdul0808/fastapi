# # -*- coding: utf-8 -*-
# """
# Created on Sun Jun 12 11:36:48 2022

# @author: abdul
# """

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pickle
# import json


# app = FastAPI()

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class model_input(BaseModel):
    
#     news_heading: str
    

# # loading the saved model
# fakenews_model = pickle.load(open('FakeNews_model.sav','rb'))


# @app.post('/FakeNews_prediction')
# def diabetes_pred(input_parameters : model_input):
    
#     input_data = input_parameters.json()
#     input_dictionary = json.loads(input_data)
#     news = input_dictionary['news_heading']
#     input_list = [news]
#     prediction = fakenews_model.predict([input_list])
    
#     if prediction[0] == 0:
#         return 'The news is Real'
#     else:
#         return 'The news is Fake'





# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pickle
# import json
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer

# app = FastAPI()

# # Load your model
# fakenews_model = pickle.load(open('FakeNews_model.sav', 'rb'))
# X = news_dataset['content'].values
# # Load your TF-IDF vectorizer
# vectorizer = TfidfVectorizer()
# vectorizer.fit(X)  # Assuming X is the original text data used to train the model

# # Create an instance of PorterStemmer
# port_stem = PorterStemmer()

# # CORS middleware
# origins = ["*"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Pydantic model for input validation
# class ModelInput(BaseModel):
#     news_heading: str

# # Function for text preprocessing
# def stemming(content):
#     stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
#     stemmed_content = stemmed_content.lower()
#     stemmed_content = stemmed_content.split()
#     stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
#     stemmed_content = ' '.join(stemmed_content)
#     return stemmed_content

# @app.post('/FakeNews_prediction')
# def fake_news_prediction(input_data: ModelInput):
#     try:
#         # Preprocess the input data
#         input_news = input_data.news_heading
#         input_news_processed = stemming(input_news)  # Apply the same stemming function
#         input_vectorized = vectorizer.transform([input_news_processed])

#         # Make prediction
#         prediction = fakenews_model.predict(input_vectorized)

#         if prediction[0] == 0:
#             return 'The news is Real'
#         else:
#             return 'The news is Fake'

#     except Exception as e:
#         # Log the error (you might want to replace this with your preferred logging mechanism)
#         print(f'Error during prediction: {e}')
#         # Return an error response
#         return {'error': 'Internal Server Error'}, 500





# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pickle
# import json


# app = FastAPI()

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class model_input(BaseModel):
    
#     news_heading: str
    

# # loading the saved model
# fakenews_model = pickle.load(open('FakeNews_model.sav','rb'))


# @app.post('/FakeNews_prediction')
# def diabetes_pred(input_parameters : model_input):
    
#     input_data = input_parameters.json()
#     input_dictionary = json.loads(input_data)
#     news = input_dictionary['news_heading']
#     input_list = [news]
#     prediction = fakenews_model.predict([input_list])
    
#     if prediction[0] == 0:
#         return 'The news is Real'
#     else:
#         return 'The news is Fake'




from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    news_heading: str

# Load the saved model
fakenews_model = pickle.load(open('FakeNews_model.sav', 'rb'))
# fakenews_model = pickle.load(open(r'C:\Users\abdul\Desktop\ML model an API\FakeNews_model1.sav', 'rb'))

# Load the TF-IDF vectorizer
vectorizer = pickle.load(open('TfidfVectorizer_model.sav', 'rb'))
# vectorizer = pickle.load(open(r'C:\Users\abdul\Desktop\ML model an API\TfidfVectorizer_model.sav', 'rb'))

port_stem = PorterStemmer()

def preprocess_text(content):
    # Apply stemming and other preprocessing steps
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
   
@app.post('/FakeNews_prediction')
def fake_news_prediction(input_parameters: ModelInput):
    # Preprocess the input text
    news_heading = preprocess_text(input_parameters.news_heading)

    # Vectorize the preprocessed text
    input_vector = vectorizer.transform([news_heading])

    # Make predictions using the loaded model
    prediction = fakenews_model.predict(input_vector)

    # Return the prediction result
    if prediction[0] == 0:
        return 'The news is Real'
    else:
        return 'The news is Fake'
