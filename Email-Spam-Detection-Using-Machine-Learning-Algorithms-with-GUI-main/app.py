import nltk

# Download stopwords and 'punkt' tokenizer from NLTK (only need to run once)
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as s
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download stopwords dari NLTK (hanya perlu dijalankan sekali)
nltk.download('stopwords')

# Menginisialisasi PorterStemmer
ps = PorterStemmer()

# Fungsi untuk memuat file pickle dengan aman
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        s.error(f"Error loading {file_path}: {e}")
        return None

# Memuat TF-IDF vectorizer dan model yang sudah dilatih dari file pickle
v = load_pickle('vectorizer.pkl')
mnb_model = load_pickle('model.pkl')

# Fungsi untuk memproses teks input
def change_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    l = []
    for i in text:
        if i.isalnum():
            l.append(i)

    text = l[:]
    l.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            l.append(i)

    text = l[:]
    l.clear()

    for i in text:
        l.append(ps.stem(i))

    return " ".join(l)

# Judul aplikasi Streamlit
s.title("Email Spam Classifier")

# Kotak input untuk pesan email
input_msg = s.text_input("Enter the email message")

# Tombol untuk memicu prediksi
if s.button('Predict'):
    if v is not None and mnb_model is not None:
        # Memproses pesan input
        changed_msg = change_text(input_msg)
        
        # Mengubah pesan menggunakan TF-IDF vectorizer
        to_be_predicted_msg = v.transform([changed_msg])
        
        # Memprediksi menggunakan model yang sudah dilatih
        prediction = mnb_model.predict(to_be_predicted_msg)[0]
        
        # Menampilkan hasil
        if prediction == 1:
            s.header("It's a spam message")
        else:
            s.header("Not a spam message")
    else:
        s.error("Failed to load model or vectorizer. Please check the files.")
