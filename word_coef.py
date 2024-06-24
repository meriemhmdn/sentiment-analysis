from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np


treated_comments_strings = [' '.join(comment) for comment in treated_comments]
vectorizer = TfidfVectorizer()
X_1 = vectorizer.fit_transform(treated_comments_strings)

# Entraînement du modèle de régression logistique
model = LogisticRegression()
model.fit(X_1, Y['target'])

# Récupération des coefficients
coefficients = model.coef_[0]

# Création d'un dictionnaire de mots et de leurs coefficients
word_coef = {word: coef for word, coef in zip(vectorizer.get_feature_names(), coefficients)}

# Tri des mots par coefficient
sorted_word_coef = sorted(word_coef.items(), key=lambda x: x[1], reverse=True)

# Mots les plus positifs
top_positive_words = sorted_word_coef[:10]

# Mots les plus négatifs
top_negative_words = sorted_word_coef[-10:]
positive_words = [(word, coef) for word, coef in sorted_word_coef if coef > 1]
negative_words = [(word, coef) for word, coef in sorted_word_coef if coef < -1]

with open('positive_words.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Word', 'Coefficient'])
    for word, coef in positive_words:
        writer.writerow([word, coef])

# Enregistrer les mots avec des coefficients négatifs dans un fichier CSV
with open('negative_words.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Word', 'Coefficient'])
    for word, coef in negative_words:
        writer.writerow([word, coef])
        
# Enregistrer les mots avec des coefficients négatifs dans un fichier CSV
with open('sorted_word_coef.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Word', 'Coefficient'])
    for word, coef in sorted_word_coef:
        writer.writerow([word, coef])

print("Top 10 mots les plus positifs :")
for word, coef in top_positive_words:
    print(f"{word}: {coef}")

print("\nTop 10 mots les plus négatifs :")
for word, coef in top_negative_words:
    print(f"{word}: {coef}")
