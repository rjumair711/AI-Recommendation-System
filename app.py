import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# ==============================
# LOAD DATASET
# ==============================

movies = pd.read_csv("dataset/movies.csv")

movies["combined_features"] = (
    movies["genre"] + " " + movies["tags"]
)

# ==============================
# TF-IDF MODEL
# ==============================

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(
    movies["combined_features"]
)

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("🎬 About Project")

st.sidebar.info(
    """
    This AI system recommends movies using:

    ✅ TF-IDF Vectorization

    ✅ Cosine Similarity

    ✅ Content-Based Filtering
    """
)

st.sidebar.subheader("Filter by Genre")

genre_option = st.sidebar.selectbox(
    "Choose Genre",
    ["All"] + sorted(movies["genre"].unique().tolist())
)

# ==============================
# MAIN TITLE
# ==============================

st.title("🎬 AI Movie Recommendation System")

st.write(
    "Get personalized movie recommendations using Artificial Intelligence."
)

# ==============================
# USER INPUT
# ==============================

user_input = st.text_input(
    "Enter movie interests:",
    placeholder="Example: AI future space"
)

# ==============================
# RECOMMENDATION BUTTON
# ==============================

if st.button("Get Recommendations"):

    if user_input.strip() == "":
        st.warning("Please enter some interests.")

    else:

        # Filter dataset by genre
        filtered_movies = movies.copy()

        if genre_option != "All":
            filtered_movies = filtered_movies[
                filtered_movies["genre"] == genre_option
            ]

        # Recalculate vectors
        filtered_vectors = vectorizer.fit_transform(
            filtered_movies["combined_features"]
        )

        # User vector
        user_vector = vectorizer.transform([user_input])

        # Similarity
        similarity_scores = cosine_similarity(
            user_vector,
            filtered_vectors
        )

        # Scores
        scores = list(enumerate(similarity_scores[0]))

        # Sort descending
        sorted_movies = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("🎯 Top Recommendations")

        recommendation_count = 0

        for movie in sorted_movies:

            index = movie[0]
            score = movie[1]

            if score > 0:

                title = filtered_movies.iloc[index]["title"]
                genre = filtered_movies.iloc[index]["genre"]
                tags = filtered_movies.iloc[index]["tags"]

                recommendation_count += 1

                st.markdown(f"""
                ---
                ## {recommendation_count}. {title}

                🎭 **Genre:** {genre}

                🏷️ **Tags:** {tags}

                ⭐ **Match Score:** {score:.2f}
                """)

            if recommendation_count == 5:
                break

        if recommendation_count == 0:
            st.error("No matching movies found.")

# ==============================
# FOOTER
# ==============================

st.markdown("---")

st.caption(
    "Built with Python, Streamlit, TF-IDF and Cosine Similarity"
)