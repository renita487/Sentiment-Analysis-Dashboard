import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Title
st.title("💬 Sentiment Analysis Dashboard")
st.write("Analyze text sentiment with a simple AI model!")

# Improved dataset (with neutral)
data = {
    "text": [
        "I love this", "Amazing product", "This is good", "Very nice",
        "I hate this", "Worst experience", "This is bad", "Very poor",
        "It's okay", "Not bad", "Average product", "Nothing special"
    ],
    "label": [1,1,1,1, 0,0,0,0, 2,2,2,2]  # 1=Positive, 0=Negative, 2=Neutral
}

df = pd.DataFrame(data)

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

model = LogisticRegression()
model.fit(X, df["label"])

# Input
user_input = st.text_input("✍️ Enter your sentence:")

# Store results for chart
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        input_vec = vectorizer.transform([user_input])
        result = model.predict(input_vec)

        if result[0] == 1:
            st.success("😊 Positive Sentiment")
            st.session_state.history.append("Positive")

        elif result[0] == 0:
            st.error("😡 Negative Sentiment")
            st.session_state.history.append("Negative")

        else:
            st.info("😐 Neutral Sentiment")
            st.session_state.history.append("Neutral")

# Dashboard (chart)
st.subheader("📊 Sentiment History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history, columns=["Sentiment"])
    chart = history_df["Sentiment"].value_counts()
    st.bar_chart(chart)
else:
    st.write("No data yet. Start analyzing!")
