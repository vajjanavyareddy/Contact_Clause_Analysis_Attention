import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================
# Load model and tokenizer
# ==========================

model = tf.keras.models.load_model(
    "contract_model.keras",
    compile=False
)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LENGTH = 100

# ==========================
# Text Cleaning
# ==========================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()

# ==========================
# Positional Encoding
# ==========================

def positional_encoding(
    position,
    d_model
):

    pe = np.zeros((position,d_model))

    for pos in range(position):

        for i in range(0,d_model,2):

            pe[pos,i] = np.sin(
                pos /
                (10000**(i/d_model))
            )

            if i+1 < d_model:

                pe[pos,i+1] = np.cos(
                    pos /
                    (10000**(i/d_model))
                )

    return pe

# ==========================
# Prediction
# ==========================

def predict_clause(text):

    text = clean_text(text)

    seq = tokenizer.texts_to_sequences(
        [text]
    )

    pad = tf.keras.preprocessing.sequence.pad_sequences(
        seq,
        maxlen=MAX_LENGTH,
        padding='post'
    )

    pred = model.predict(
        pad,
        verbose=0
    )

    class_id = np.argmax(pred)

    label = label_encoder.inverse_transform(
        [class_id]
    )[0]

    confidence = np.max(pred)

    return label, confidence

# ==========================
# Streamlit UI
# ==========================

st.title(
    "⚖️AI Contract Intelligence System"
)

uploaded_file = st.file_uploader(
    "📄Upload Contract",
    type=["txt"]
)

if uploaded_file:

    contract_text = uploaded_file.read().decode()

    st.subheader(
        "📜Contract Content"
    )

    st.write(contract_text)

    # Prediction

    label, conf = predict_clause(
        contract_text
    )

    st.subheader(
        "🎯Predicted Clause Type"
    )

    st.success(
        f"{label} ({conf:.2f})"
    )

    # Important terms

    st.subheader(
        "📍Important Terms"
    )

    keywords = [
        "payment",
        "termination",
        "confidential",
        "liability",
        "agreement",
        "party"
    ]

    found_words = []

    for word in keywords:

        if word.lower() in contract_text.lower():

            found_words.append(word)

    st.write(found_words)

    # Attention Map (Demo)

    st.subheader(
        "Attention Map"
    )

    attention = np.random.rand(
        15,
        15
    )

    fig, ax = plt.subplots(
        figsize=(8,6)
    )

    sns.heatmap(
        attention,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

    # Positional Encoding

    st.subheader(
        "Positional Encoding Heatmap"
    )

    pe = positional_encoding(
        50,
        64
    )

    fig2, ax2 = plt.subplots(
        figsize=(10,6)
    )

    sns.heatmap(
        pe,
        cmap="viridis",
        ax=ax2
    )

    st.pyplot(fig2)