import csv
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
import json
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk                         # NLP toolbox
from os import getcwd
import pandas as pd                 # Library for Dataframes 
from nltk.corpus import twitter_samples 
import matplotlib.pyplot as plt     # Library for visualization
import numpy as np 
import pandas as pd
import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer
import os
from unidecode import unidecode
nltk.download('words')

input_file = "sorted_word_coef_norm.csv"
word_coef_dict = {}

with open(input_file, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        word = row['Word']
        coef = float(row['Coefficient'])
        word_coef_dict[word] = coef


            
        
frequencies = {}
file_path = "frequencies2.csv"
with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        word = row['Word']
        positive_freq = float(row['Positive_Frequency'])
        negative_freq = float(row['Negative_Frequency'])
        frequencies[word] = (positive_freq, negative_freq)
        

        
        
def scrape_imdb_reviews(movie_id):
    base_url = f"https://www.imdb.com/title/{movie_id}/reviews/_ajax"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    all_reviews = []
    pagination_key = None
    page = 0
    while True:
        params = {
            'ref_': 'undefined',
            'paginationKey': pagination_key
        }
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            reviews_divs = soup.find_all('div', class_='text show-more__control')
            
            if not reviews_divs:
                print("Aucun commentaire trouvé.")
                break
            
            all_reviews.extend(review.get_text(strip=True) for review in reviews_divs)
            
            # Mettez à jour la clé de pagination pour obtenir la page suivante
            pagination_div = soup.find('div', class_='load-more-data')
            if pagination_div:
                pagination_key = pagination_div.get('data-key')
                # print(pagination_key)
            else:
                print("Clé de pagination non trouvée. Arrêt de la récupération.")
                break

            if pagination_key is None:
                print("Clé de pagination non trouvée. Arrêt de la récupération.")
                break
            
            # Pause pour éviter de surcharger le serveur
            # time.sleep(1)
        else:
            print(f"Erreur lors de la requête : {response.status_code}")
            break
    return all_reviews

def separate_phrases(comments):
    # Liste pour stocker les résultats
    result = []
    
    # Parcourir chaque commentaire dans la liste
    for comment in comments:
        # Enlever les espaces blancs inutiles
        comment = comment.strip()
        
        # Séparer les phrases en utilisant une fonction de tokenisation
        phrases = sent_tokenize(comment)
        
        # Ajouter la liste des phrases au résultat
        result.append([phrase.strip() for phrase in phrases])
    
    return result


def preprocess1(text):
    
    text1 = unidecode(text.lower())
    
    text1 = re.sub(r'^RT[\s]+', '', text1)

    text1 = re.sub(r'https?://[^\s\n\r]+', '', text1)

    text1 = re.sub(r'#', '', text1)
    
    text1 = text1.replace("'", " ")
    
    text1 = text1.replace("_", " ")
    
    text1 = text1.replace("-", " ") 
    
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    text1_tokens = tokenizer.tokenize(text1)
    stopwords_english = stopwords.words('english') 
    alphabet = list(string.ascii_lowercase)
    
    
    text1_clean = []

    for word in text1_tokens: # Go through every word in your tokens list
        if((word not in stopwords_english) and (word not in string.punctuation) and (word not in alphabet)):  
            text1_clean.append(word)
    
    
    stemmer = PorterStemmer() 

    text1_stem = [] 
    
    for word in text1_clean:
        stem_word = stemmer.stem(word)
        text1_stem.append(stem_word) 
    
    
    text1_final = [] 
    for word in text1_stem: # Go through every word in your tokens list
        if((word not in stopwords_english) and (word not in string.punctuation) and (word not in alphabet)):  
            text1_final.append(word)
        
    return text1_final


def separate_phrases(comments):
    result = []
    for comment in comments:
        comment = comment.strip()
        phrases = sent_tokenize(comment)
        result.append([phrase.strip() for phrase in phrases])
    return result

def extract_aspect_context(comment, aspect_keywords, window_size=4):
    sentences = sent_tokenize(comment)
    aspect_contexts = []

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for keyword in aspect_keywords:
            if keyword in words:
                keyword_index = words.index(keyword)
                start_index = max(0, keyword_index - window_size)
                end_index = min(len(words), keyword_index + window_size + 1)
                context = words[start_index:end_index]
                aspect_contexts.append(' '.join(context))
    
    return ' '.join(aspect_contexts)

def extract_aspects(comment):
    aspects = {
        "realisateur": [
            "director", "filmmaker", "moviemaker", "helmer", "auteur", "cinematographer", 
            "producer", "visionary"
        ],
        "acteur": [
            "actor", "performer", "lead", "protagonist", "thespian", "cast member", 
            "character", "star", "co-star", "supporting actor", "actress", "talent"
        ],
        "scenario": [
            "screenplay", "script", "storyline", "plot", "writing", "narrative", 
            "dialogue", "screenwriting", "story", "story arc", "scriptwriting"
        ],
        "decor": [
            "set", "scenery", "decoration", "backdrop", "stage", "setting", 
            "set design", "props", "environment", "locale"
        ],
        "design": [
            "design", "aesthetics", "styling", "artistry", "craftsmanship", "visual design", 
            "production design", "set design", "art direction", "visuals", "look"
        ],
        "musique": [
            "music", "soundtrack", "composition", "melody", "tunes", 
            "songs", "musical score", "orchestration", "sound", "soundscape", "opening", "ost"
        ],
        "effets_visuels": [
            "visual", "VFX", "special", "FX", "visual effects", "CGI", "computer graphics", 
            "special effects", "animation", "graphics", "effects"
        ],
        "production":[
        "production", "filmmaking", "movie making", "film production", "cinema production", 
        "film creation", "movie creation", "production process", "production work"]
    }

    extracted_aspects = {aspect: [] for aspect in aspects}
    for aspect, keywords in aspects.items():
        context = extract_aspect_context(comment, keywords)
        if context:
            extracted_aspects[aspect].append(context)
    
    return extracted_aspects

def retrieval_models(key_words):
    key_words_preprocessed = preprocess1(key_words)
    score = {0: 0, 1: 0}
    max_score = 0
    words_count = 0
    for word in key_words_preprocessed:
        if word in frequencies and word in word_coef_dict_norm:
            score[0] += frequencies[word][1] - word_coef_dict[word]
            score[1] += frequencies[word][0] + word_coef_dict[word]
            #if word in word_coef_dict_norm:
            max_score += (word_coef_dict_norm[word]*10)
                #if ((word_coef_dict[word]+10)/2) > 1:
            words_count += 1

    max_class = max(score, key=score.get)
    if (words_count != 0):
        max_score = max_score / words_count

    return max_class, max_score

def classify_comment(comment):
    classification, score = retrieval_models(comment)
    return classification, score

def classify_comment_by_aspect(comment, aspect):
    aspects = extract_aspects(comment)
    if aspect in aspects and aspects[aspect]:
        aspect_comment = ' '.join(aspects[aspect])
        classification, score = retrieval_models(aspect_comment)
        return classification, score
    else:
        return "Aspect not found in comment", None

def classify_comments_by_aspects(comments, aspects):
    results = {aspect: [] for aspect in aspects}
    global_scores = []
    for comment in comments:
        global_classification, global_score = classify_comment(comment)
        global_scores.append(global_score)
        for aspect in aspects:
            classification, score = classify_comment_by_aspect(comment, aspect)
            if score is not None:
                results[aspect].append({'comment': comment, 'classification': classification, 'score': score})
    results['note_global'] = global_scores
    return results

def format_classification_results(results):
    formatted_results = {}
    for aspect, aspect_results in results.items():
        if aspect != 'note_global':
            formatted_results[aspect] = []
            for result in aspect_results:
                formatted_results[aspect].append({
                    'comment': result['comment'],
                    'classification': result['classification'],
                    'score': result['score']
                })
    formatted_results['note_global'] = results['note_global']
    return formatted_results

def process_comments_list(comments_list, aspects):
    all_results = {}
    for idx, comments in enumerate(comments_list):
        filename = f"file_{idx}.txt"
        results = classify_comments_by_aspects(comments, aspects)
        formatted_results = format_classification_results(results)
        all_results[filename] = {}
        for i, comment in enumerate(comments):
            all_results[filename][i] = {aspect: {'classification': res['classification'], 'score': res['score']}
                                        for aspect, res_list in formatted_results.items() 
                                        if aspect != 'note_global' for res in res_list if res['comment'].strip() == comment.strip()}
        all_results[filename]['note_global'] = formatted_results['note_global']
    return all_results

def save_results_to_json(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def calculate_aspect_averages(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    aspect_totals = {}
    aspect_counts = {}
    comment_aspect_averages = []

    for filename, file_results in results.items():
        for line_num, aspect_results in file_results.items():
            if line_num != 'note_global':
                aspect_scores = []
                for aspect, res in aspect_results.items():
                    if aspect not in aspect_totals:
                        aspect_totals[aspect] = 0
                        aspect_counts[aspect] = 0
                    aspect_totals[aspect] += res['score']
                    aspect_counts[aspect] += 1
                    aspect_scores.append(res['score'])

                if aspect_scores:
                    comment_average = sum(aspect_scores) / len(aspect_scores)
                    comment_aspect_averages.append(comment_average)

    aspect_averages = {aspect: ((aspect_totals[aspect] / aspect_counts[aspect])) for aspect in aspect_totals}
    overall_average = sum(comment_aspect_averages) / len(comment_aspect_averages) if comment_aspect_averages else 0
    aspect_averages['note_global'] = overall_average

    return aspect_averages

def calculate_aspect_averages(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    aspect_totals = {}
    aspect_counts = {}
    comment_aspect_averages = []

    for filename, file_results in results.items():
        for line_num, aspect_results in file_results.items():
            if line_num != 'note_global':
                all_aspects_present = True
                aspect_scores = []
                for aspect in aspect_totals.keys():
                    if aspect in aspect_results:
                        score = aspect_results[aspect]['score']
                        aspect_totals[aspect] += score
                        aspect_counts[aspect] += 1
                        aspect_scores.append(score)
                    else:
                        all_aspects_present = False
                        break

                if all_aspects_present and aspect_scores:
                    comment_average = sum(aspect_scores) / len(aspect_scores)
                    comment_aspect_averages.append(comment_average)

    # Calculate the aspect averages
    aspect_averages = {aspect: (aspect_totals[aspect] / aspect_counts[aspect]) for aspect in aspect_totals}
    overall_average = sum(comment_aspect_averages) / len(comment_aspect_averages) if comment_aspect_averages else 0
    aspect_averages['note_global'] = overall_average

    return aspect_averages
    
def score_movie(movie_id):
    all_reviews = scrape_imdb_reviews(movie_id)
    comments = separate_phrases(all_reviews)
    aspects = ["realisateur", "acteur", "scenario", "decor", "design", "musique", "effets_visuels", "production"]
    output_file = 'results_note_glob.json'

    final_results = process_comments_list(comments, aspects)
    save_results_to_json(final_results, output_file)
    """
    for filename, file_results in final_results.items():
        print(f"Results for file '{filename}':")
        for line_num, aspect_results in file_results.items():
            if line_num != 'note_global':
                print(f"Line {line_num}: {aspect_results}")
        print(f"Global note for '{filename}': {file_results['note_global']}")"""

    

    # Fichier JSON d'entrée
    input_file = 'results_note_glob.json'

    # Calculer les moyennes des aspects
    average_results = calculate_aspect_averages(input_file)

    # Enregistrer les résultats dans un vecteur
    output_file_json = 'average_results.json'
    average_vector = list(average_results.values())
    with open(output_file_json, 'w', encoding='utf-8') as f:
        json.dump(average_results, f, ensure_ascii=False, indent=4)
    return average_vector

movie_id = "tt0084516"
score_movie(movie_id)
