import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# LOAD DATASET
# ==============================

movies = pd.read_csv("dataset/movies.csv")

# Combine features
movies["combined_features"] = movies["genre"] + " " + movies["tags"]

# ==============================
# TF-IDF VECTORIZATION
# ==============================

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(movies["combined_features"])

# ==============================
# PROJECT HEADER
# ==============================

print("=" * 50)
print("     AI MOVIE RECOMMENDATION SYSTEM")
print("=" * 50)

# ==============================
# USER INPUT
# ==============================

user_input = input("\nEnter movie interests: ").lower()

# Handle empty input
if user_input.strip() == "":
    print("\nPlease enter some interests.")
    exit()

# ==============================
# USER VECTOR
# ==============================

user_vector = vectorizer.transform([user_input])

# ==============================
# CALCULATE SIMILARITY
# ==============================

similarity_scores = cosine_similarity(user_vector, feature_vectors)

# Convert to list
scores = list(enumerate(similarity_scores[0]))

# Sort descending
sorted_movies = sorted(scores, key=lambda x: x[1], reverse=True)

# ==============================
# DISPLAY RESULTS
# ==============================

print("\nTop Recommended Movies:\n")

recommendation_count = 0

for movie in sorted_movies:

    index = movie[0]
    score = movie[1]

    # Ignore very low similarity
    if score > 0:

        title = movies.iloc[index]["title"]
        genre = movies.iloc[index]["genre"]

        recommendation_count += 1

        print(f"{recommendation_count}. {title}")
        print(f"   Genre: {genre}")
        print(f"   Match Score: {score:.2f}")
        print("-" * 40)

    # Show only top 5
    if recommendation_count == 5:
        break

# ==============================
# NO MATCH FOUND
# ==============================

if recommendation_count == 0:
    print("\nNo matching movies found.")