# read your train set
repertoire = 'pos'
fichiers_txt = [f for f in os.listdir(repertoire) if f.endswith('.txt')]
i = 0
pos_treated_comments = []
for nom_fichier in fichiers_txt:
    chemin_fichier = os.path.join(repertoire, nom_fichier)
    with open(chemin_fichier, 'r') as fichier:
        contenu = fichier.read()
    preproceced_comment = preprocess(contenu)
    pos_treated_comments.append(preproceced_comment)


repertoire = 'neg'
fichiers_txt = [f for f in os.listdir(repertoire) if f.endswith('.txt')]
i = 0
neg_treated_comments = []
for nom_fichier in fichiers_txt:
    chemin_fichier = os.path.join(repertoire, nom_fichier)
    with open(chemin_fichier, 'r') as fichier:
        contenu = fichier.read()
    preproceced_comment = preprocess1(contenu)
    neg_treated_comments.append(preproceced_comment)

treated_comments = pos_treated_comments + neg_treated_comments
# calculate word frequencies
frequencies = {}
pos_freq = 0
neg_freq = 0

for pos_comment in pos_treated_comments:
    for word in pos_comment:
        if word not in frequencies:
            frequencies[word] = [1, 0]
        else:
            frequencies[word][0] += 1
        
            
for neg_comment in neg_treated_comments:
    for word in neg_comment:
        if word not in frequencies:
            frequencies[word] = [0, 1]
        else:
            frequencies[word][1] += 1
            
for key in frequencies:
    somme = frequencies[key][0]+frequencies[key][1]
    frequencies[key][0] = (frequencies[key][0])/somme
    frequencies[key][1] = (frequencies[key][1])/somme

# write it in files
with open('frequencies.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Word', 'Positive_Frequency', 'Negative_Frequency'])
    for word, freq in frequencies.items():
        writer.writerow([word, freq[0], freq[1]])




#transform your trainset into a matrix where each row represents a comment in this forme [biais, sum_posiyiv_freq, sum_negativ_freq]

num_comments = len(treated_comments)
num_features = 3 
X = np.zeros((num_comments, num_features))
i = 0

for comment in treated_comments:
    for word in comment:
        if word in word_coef_dict:
            X[i][0] = 1
            X[i][1] += frequencies[word][0]
            X[i][2] += frequencies[word][1]
    i += 1
