import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from django.shortcuts import render
from django.http import JsonResponse

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load External Dataset
df = pd.read_csv("./chatbot/data/medical_data.csv")  # Make sure your dataset exists in this path

# Preprocess the dataset
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

df["Processed_Question"] = df["Question"].apply(preprocess_text)

# Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Processed_Question"])

# Train Nearest Neighbors Model
model = NearestNeighbors(n_neighbors=1, metric="cosine")
model.fit(X)

# Function to Get Best Response
def chatbot_response(user_message):
    processed_message = preprocess_text(user_message)
    vectorized_message = vectorizer.transform([processed_message])
    _, index = model.kneighbors(vectorized_message)
    
    best_match = df.iloc[index[0][0]]
    return best_match["Answer"]

# Django View to Handle Chat Requests
def chat(request):
    if request.method == "POST":
        user_message = request.POST.get("message", "")
        response = chatbot_response(user_message)
        return JsonResponse({"response": response})

    return render(request, "chatbot/chat.html")
