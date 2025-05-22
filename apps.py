import pandas as pd
import numpy as np
from nltk.corpus import wordnet
import csv
import json
import itertools
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import en_core_web_sm
import joblib
from flask import Flask, render_template, request, session ,jsonify,url_for
from flask_sqlalchemy import SQLAlchemy

import traceback

def backup_session():
    try:
        with open("session_backup.json", "w") as f:
            json.dump(dict(session), f)
    except Exception as e:
        print("Session backup failed:", e)

def restore_session():
    try:
        with open("session_backup.json", "r") as f:
            saved = json.load(f)
            session.update(saved)
    except Exception:
        pass

import matplotlib.pyplot as plt
import os
#import mysql.connector

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')

# save data

# Only create the JSON file if it doesn't exist
if not os.path.exists('DATA.json'):
    with open('DATA.json', 'w') as outfile:
        json.dump({"users": []}, outfile, indent=4)

def write_json(new_data, filename='DATA.json'):
    try:
        # Read existing data
        with open(filename, 'r') as file:
            file_data = json.load(file)

        # Append new data
        file_data["users"].append(new_data)

        # Write updated data
        with open(filename, 'w') as file:
            json.dump(file_data, file, indent=4)
    except Exception as e:
        print(f"Error writing to {filename}: {e}")


df_tr = pd.read_csv('Medical_dataset/Training.csv')
df_tt = pd.read_csv('Medical_dataset/Testing.csv')

symp = []
disease = []
for i in range(len(df_tr)):
    symp.append(df_tr.columns[df_tr.iloc[i] == 1].to_list())
    disease.append(df_tr.iloc[i, -1])

####################################################            dont do anything in above         ###########################################################################################################
# ################################ Submission data test                           ##################################



@app.route('/submit-appointment', methods=['POST'])
def submit_appointment():
    content = request.get_json()
    data = {
        "name": content.get('name'),
        "age": content.get('age'),
        "gender": content.get('gender'),
        "phone": content.get('phone'),
        "address": content.get('address')
    }


    # Save to submission.json
    submissions_file = 'submission.json'

    if os.path.exists(submissions_file):
        with open(submissions_file, 'r') as f:
            try:
                submissions = json.load(f)
            except json.JSONDecodeError:
                submissions = []
    else:
        submissions = []

    submissions.append(data)

    with open(submissions_file, 'w') as f:
        json.dump(submissions, f, indent=4)

    return jsonify({"message": "Appointment submitted successfully."})

# ✅ Flask Route to Handle Chatbot Interaction
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    
    # Assuming you have a function to analyze symptoms and predict disease
    disease, symptoms = analyze_symptoms(user_message)  # Replace with actual function
    
    if disease:  # If disease is detected, store user data
        name = session.get("name", "Unknown")  # Get name from session or set default
        age = session.get("age", 0)
        gender = session.get("gender", "Unknown")

        save_patient_data(name, age, gender, disease, symptoms )

        return jsonify({"response": f"You might have {disease}. Data saved!"})
    
    return jsonify({"response": "I'm still analyzing your symptoms..."})

# ✅ Flask Route to Collect User Information
@app.route("/user_info", methods=["POST"])
def user_info():
    data = request.json
    session["name"] = data.get("name")
    session["age"] = data.get("age")
    session["gender"] = data.get("gender")

    return jsonify({"response": "User information saved!"})



####################################################            dont do anything in below         ###########################################################################################################

    
# # I- GET ALL SYMPTOMS

all_symp_col = list(df_tr.columns[:-1])
print(all_symp_col)

def clean_symp(sym):
    return sym.replace('_', ' ').replace('.1', '').replace('(typhos)', '').replace('yellowish', 'yellow').replace(
        'yellowing', 'yellow')


all_symp = [clean_symp(sym) for sym in (all_symp_col)]


def preprocess(doc):
    nlp_doc = nlp(doc)
    d = []
    for token in nlp_doc:
        if (not token.text.lower() in STOP_WORDS and token.text.isalpha()):
            d.append(token.lemma_.lower())
    return ' '.join(d)


all_symp_pr = [preprocess(sym) for sym in all_symp]

# associate each processed symp with column name
col_dict = dict(zip(all_symp_pr, all_symp_col))

# II- Syntactic Similarity

# Returns all the subsets of a set. This is a generator.
# {1,2,3}->[{},{1},{2},{3},{1,3},{1,2},..]
def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item


# Sort list based on length
def sort(a):
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if len(a[j]) > len(a[i]):
                a[i], a[j] = a[j], a[i]
    a.pop()
    return a

# find all permutations of a list
def permutations(s):
    permutations = list(itertools.permutations(s))
    return ([' '.join(permutation) for permutation in permutations])


# check if a txt and all diferrent combination if it exists in processed symp list
def DoesExist(txt):
    txt = txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations:
        # print(permutations(comb))
        for sym in permutations(comb):
            if sym in all_symp_pr:
                # print(sym)
                return sym
    return False

# Jaccard similarity 2docs
def jaccard_set(str1, str2):
    list1 = str1.split(' ')
    list2 = str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# apply vanilla jaccard to symp with all corpus
def syntactic_similarity(symp_t, corpus):
    most_sim = []
    poss_sym = []
    for symp in corpus:
        d = jaccard_set(symp_t, symp)
        most_sim.append(d)
    order = np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if DoesExist(corpus[i]):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None


# check a pattern if it exists in processed symp list
def check_pattern(inp, dis_list):
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, None


# III- Semantic Similarity


from nltk.wsd import lesk
from nltk.tokenize import word_tokenize


def WSD(word, context):
    sens = lesk(context, word)
    return sens


# semantic similarity 2docs
def semanticD(doc1, doc2):
    doc1_p = preprocess(doc1).split(' ')
    doc2_p = preprocess(doc2).split(' ')
    score = 0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1, doc1)
            syn2 = WSD(tock2, doc2)
            if syn1 is not None and syn2 is not None:
                x = syn1.wup_similarity(syn2)
                # x=syn1.path_similarity((syn2))
                if x is not None and x > 0.25:
                    score += x
    return score / (len(doc1_p) * len(doc2_p))


# apply semantic simarity to symp with all corpus
def semantic_similarity(symp_t, corpus):
    max_sim = 0
    most_sim = None
    for symp in corpus:
        d = semanticD(symp_t, symp)
        if d > max_sim:
            most_sim = symp
            max_sim = d
    return max_sim, most_sim


# given a symp suggest possible synonyms
def suggest_syn(sym):
    symp = []
    synonyms = wordnet.synsets(sym)
    lemmas = [word.lemma_names() for word in synonyms]
    lemmas = list(set(itertools.chain(*lemmas)))
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symp_pr)
        if res != 0:
            symp.append(sym1)
    return list(set(symp))


# One-Hot-Vector dataframe
def OHV(cl_sym, all_sym):
    l = np.zeros([1, len(all_sym)])
    for sym in cl_sym:
        l[0, all_sym.index(sym)] = 1
    return pd.DataFrame(l, columns=all_symp)


def contains(small, big):
    a = True
    for i in small:
        if i not in big:
            a = False
    return a


# list of symptoms --> possible diseases
def possible_diseases(l):
    poss_dis = []
    for dis in set(disease):
        if contains(l, symVONdisease(df_tr, dis)):
            poss_dis.append(dis)
    return poss_dis


# disease --> all symptoms
def symVONdisease(df, disease):
    ddf = df[df.prognosis == disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()


# IV- Prediction Model (KNN)
# load model
knn_clf = joblib.load('model/knn.pkl')

# ##  VI- SEVERITY / DESCRIPTION / PRECAUTION
# get dictionaries for severity-description-precaution for all diseases

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()


def getDescription():
    global description_list
    with open('Medical_dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('Medical_dataset/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('Medical_dataset/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


# load dictionaries
getSeverityDict()
getprecautionDict()
getDescription()


# calcul patient condition
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        if item in severityDictionary.keys():
            sum = sum + severityDictionary[item]
    if ((sum * days) / (len(exp)) > 13):
        return 1
        print("You should take the consultation from doctor. ")
    else:
        return 0
        print("It might not be that bad but you should take precautions.")


# print possible symptoms
def related_sym(psym1):
    s = "searches related to input: <br>"
    i = len(s)
    for num, it in enumerate(psym1):
        s += str(num) + ") " + clean_symp(it) + "<br>"
    if num != 0:
        s += "Select the one you meant."
        return s
    else:
        return 0

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/service")
def service():
    return render_template("service.html")

@app.route("/Appointment")
def Appointment():
    return render_template("Appointment.html")

@app.route("/get")
def get_bot_response():
    restore_session()
    s = request.args.get('msg')

    # Handle conversation termination or restart
    if "step" in session and session["step"] == "Q_C":
        name = session.get("name", "Unknown") # Use .get for safer access
        age = session.get("age", 0)
        gender = session.get("gender", "Unknown")
        session.clear() # Clear session first
        if s.lower() == "q": # Use .lower() for case-insensitive comparison
            return "Thank you for using our service Mr/Ms " + name
        else:
            session["step"] = "FS"
            session["name"] = name # Restore essential info
            session["age"] = age
            session["gender"] = gender
            backup_session() # Backup after restoring
            return "Well, Hello again Mr/Ms " + session["name"] + ", now I will be asking some few questions about your symptoms. Tap S to start diagnostic!"


    if s.upper() in ["HI", "HELLO", "HEY", "GREETINGS", "NAMASTE", "VANAKKAM", "NAMASKARAM", "NAMASKARA", "PRANAM", "SASTRIYAKAL", "KEM CHO", "KHAMOSH", "NOMOSKAR","HOLA", "BUENOS DÍAS", "SALUDOS", "QUE TAL",  # Spanish
                    "BONJOUR", "SALUT", "COUCOU",  # French
                    "HALLO", "GUTEN TAG", "SERVUS", "GRÜSS DICH",  # German
                    "CIAO", "SALVE", "BUONGIORNO", "BUONASERA",  # Italian
                    "OLÁ", "BOM DIA", "BOA TARDE", "SAUDAÇÕES",  # Portuguese
                    "SZIA", "JÓ NAPOT", "ÜDVÖZÖLLEK",  # Hungarian
                    "DAG", "GOEDEMORGEN", "GOEIENDAG", "HOI",  # Dutch
                    "HEJ", "GOD DAG", "GOD MORGON",  # Swedish
                    "HALLO", "GOD DAG", "GOD MANDAG",  # Norwegian

]:
        return "What is your name ?"

    # Handle initial information gathering
    if 'name' not in session and 'step' not in session:
        session['name'] = s
        session['step'] = "age"
        backup_session() # Backup after step change
        return "How old are you? "

    if session.get("step") == "age": # Use .get() for safer access
        try:
            session["age"] = int(s)
            session["step"] = "gender"
            backup_session() # Backup after step change
            return "Can you specify your gender ?"
        except ValueError:
            return "Please enter a valid age as a number."


    if session.get("step") == "gender":
        session["gender"] = s
        session["step"] = "Depart"
        backup_session() # Backup after step change

    if session.get('step') == "Depart":
        session['step'] = "BFS"
        backup_session() # Backup after step change
        return "Well, Hello again Mr/Ms " + session.get("name", "Unknown") + ", now I will be asking some few questions about your symptoms. Tap S to start diagnostic!"

    if session.get('step') == "BFS":
        session['step'] = "FS"  # first symp
        backup_session() # Backup after step change
        return "Can you precise your main symptom Mr/Ms " + session.get("name", "Unknown") + " ?"

    if session.get('step') == "FS":
        sym1 = preprocess(s)
        sim1, psym1 = syntactic_similarity(sym1, all_symp_pr)
        session['FSY'] = [sym1, sim1, psym1]  # info du 1er symptome
        session['step'] = "SS"  # second symptomee
        backup_session() # Backup after step change
        if sim1 == 1:
            session['step'] = "RS1"  # related_sym1
            backup_session() # Backup after step change
            s = related_sym(psym1)
            if s != 0:
                return s
        else:
            return "You are probably facing another symptom, if so, can you specify it?"

    if session.get('step') == "RS1":
        try:
            index = int(s)
            temp = session['FSY']
            psym1 = temp[2]
            if 0 <= index < len(psym1):
                temp[2] = psym1[index]
                session['FSY'] = temp
                session['step'] = 'SS'
                backup_session() # Backup after step change
                return "You are probably facing another symptom, if so, can you specify it?"
            else:
                 return "Invalid selection. Please select a number from the list."
        except ValueError:
            return "Invalid input. Please enter the number corresponding to your symptom."


    if session.get('step') == "SS":
        sym2 = preprocess(s)
        sim2 = 0
        psym2 = []
        if len(sym2) != 0:
            sim2, psym2 = syntactic_similarity(sym2, all_symp_pr)
        session['SSY'] = [sym2, sim2, psym2]  # info du 2eME symptome(sym,sim,psym)
        session['step'] = "semantic"  # face semantic
        backup_session() # Backup after step change
        if sim2 == 1:
            session['step'] = "RS2"  # related sym2
            backup_session() # Backup after step change
            s = related_sym(psym2)
            if s != 0:
                return s
        # Added else to handle the case when sim2 is not 1
        else:
             # Proceed to semantic similarity or the next logical step
             pass # The next if statement will handle the semantic step

    if session.get('step') == "RS2":
        try:
            index = int(s)
            temp = session['SSY']
            psym2 = temp[2]
            if 0 <= index < len(psym2):
                temp[2] = psym2[index]
                session['SSY'] = temp
                session['step'] = "semantic"
                backup_session() # Backup after step change
            else:
                 return "Invalid selection. Please select a number from the list."
        except ValueError:
            return "Invalid input. Please enter the number corresponding to your symptom."

    if session.get('step') == "semantic":
        temp = session["FSY"]  # recuperer info du premier
        sym1 = temp[0]
        sim1 = temp[1]
        temp = session["SSY"]  # recuperer info du 2 eme symptome
        sym2 = temp[0]
        sim2 = temp[1]
        if sim1 == 0 or sim2 == 0:
            session['step'] = "BFsim1=0"
            backup_session() # Backup after step change
        else:
            session['step'] = 'PD'  # to possible_diseases
            backup_session() # Backup after step change

    if session.get('step') == "BFsim1=0":
        temp = session["FSY"]
        sym1 = temp[0]
        sim1 = temp[1]
        if sim1 == 0 and len(sym1) != 0:
            sim1, psym1 = semantic_similarity(sym1, all_symp_pr)
            session['FSY'] = [sym1, sim1, psym1]
            session['step'] = "sim1=0"  # process of semantic similarity=1 for first sympt.
            backup_session() # Backup after step change
        else:
            session['step'] = "BFsim2=0"
            backup_session() # Backup after step change
            # Call get_bot_response() to process the next step immediately
            return get_bot_response()


    if session.get('step') == "sim1=0":  # semantic no => suggestion
        temp = session["FSY"]
        sym1 = temp[0]
        sim1 = temp[1]
        if sim1 == 0:
            if "suggested" in session:
                sugg = session["suggested"]
                if s.lower() == "yes" and len(sugg) > 0: # Check if suggestion exists
                    psym1 = sugg[0]
                    sim1 = 1
                    temp[1] = sim1
                    temp[2] = psym1
                    session["FSY"] = temp
                    del session["suggested"] # Clear suggested list
                    session['step'] = "BFsim2=0" # Move to the next check
                    backup_session() # Backup after step change
                    return get_bot_response() # Process next step
                elif len(sugg) > 0: # If not 'yes' and suggestions exist, try next suggestion
                     del sugg[0]
                     session["suggested"] = sugg
                     if len(sugg) > 0:
                         msg = "are you experiencing any  " + clean_symp(sugg[0]) + "?" # Use clean_symp for display
                         session["suggested"] = sugg # Update session with remaining suggestions
                         backup_session() # Backup after step change
                         return msg
                     else: # No more suggestions for sim1
                         del session["suggested"]
                         session['step'] = "BFsim2=0" # Move to the next check
                         backup_session() # Backup after step change
                         return get_bot_response() # Process next step

            if "suggested" not in session:
                suggestions = suggest_syn(sym1)
                if len(suggestions) > 0:
                    session["suggested"] = suggestions
                    msg = "are you experiencing any  " + clean_symp(suggestions[0]) + "?" # Use clean_symp for display
                    backup_session() # Backup after step change
                    return msg
                else: # No suggestions found for sim1
                    session['step'] = "BFsim2=0" # Move to the next check
                    backup_session() # Backup after step change
                    return get_bot_response() # Process next step

        # If sim1 is not 0 (either initially or after suggestion), move to next check
        session['step'] = "BFsim2=0"
        backup_session() # Backup after step change
        return get_bot_response() # Process next step


    if session.get('step') == "BFsim2=0":
        temp = session["SSY"]  # recuperer info du 2 eme symptome
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0 and len(sym2) != 0:
            sim2, psym2 = semantic_similarity(sym2, all_symp_pr)
            session['SSY'] = [sym2, sim2, psym2]
            session['step'] = "sim2=0"
            backup_session() # Backup after step change
        else:
            session['step'] = "TEST"
            backup_session() # Backup after step change
            return get_bot_response()

    if session.get('step') == "sim2=0":
        temp = session["SSY"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0:
            if "suggested_2" in session:
                sugg = session["suggested_2"]
                if s.lower() == "yes" and len(sugg) > 0: # Check if suggestion exists
                    psym2 = sugg[0]
                    sim2 = 1
                    temp[1] = sim2
                    temp[2] = psym2
                    session["SSY"] = temp
                    del session["suggested_2"] # Clear suggested list
                    session['step'] = "TEST" # Move to the next check
                    backup_session() # Backup after step change
                    return get_bot_response() # Process next step
                elif len(sugg) > 0: # If not 'yes' and suggestions exist, try next suggestion
                    del sugg[0]
                    session["suggested_2"] = sugg
                    if len(sugg) > 0:
                        msg = "Do you feel " + clean_symp(sugg[0]) + "?" # Use clean_symp for display
                        session["suggested_2"] = sugg # Update session with remaining suggestions
                        backup_session() # Backup after step change
                        return msg
                    else: # No more suggestions for sim2
                        del session["suggested_2"]
                        session['step'] = "TEST" # Move to the next check
                        backup_session() # Backup after step change
                        return get_bot_response() # Process next step

            if "suggested_2" not in session:
                suggestions = suggest_syn(sym2)
                if len(suggestions) > 0:
                    session["suggested_2"] = suggestions
                    msg = "Do you feel " + clean_symp(suggestions[0]) + "?" # Use clean_symp for display
                    backup_session() # Backup after step change
                    return msg
                else: # No suggestions found for sim2
                    session['step'] = "TEST" # Move to the next check
                    backup_session() # Backup after step change
                    return get_bot_response() # Process next step

        # If sim2 is not 0 (either initially or after suggestion), move to next check
        session['step'] = "TEST"  # test if semantic and syntaxic and suggestion not found
        backup_session() # Backup after step change
        return get_bot_response() # Process next step


    if session.get('step') == "TEST":
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]

        if sim1 == 0 and sim2 == 0:
            # GO TO THE END
            result = None
            session['step'] = "END"
            backup_session() # Backup after step change
            return get_bot_response() # Process the END step
        else:
            if sim1 == 0:
                psym1 = psym2
                temp = session["FSY"]
                temp[2] = psym2
                session["FSY"] = temp
            if sim2 == 0:
                psym2 = psym1
                temp = session["SSY"]
                temp[2] = psym1
                session["SSY"] = temp
            session['step'] = 'PD'  # to possible_diseases
            backup_session() # Backup after step change
            return get_bot_response() # Process the PD step

    if session.get('step') == 'PD':
        # MAYBE THE LAST STEP
        # create patient symp list
        temp_fsy = session["FSY"] # Use different variable names to avoid confusion
        sim1 = temp_fsy[1]
        psym1 = temp_fsy[2]
        temp_ssy = session["SSY"] # Use different variable names
        sim2 = temp_ssy[1]
        psym2 = temp_ssy[2]

        if "all" not in session:
            session["asked"] = []
            session["all"] = [col_dict[psym1], col_dict[psym2]]
            backup_session() # Backup after initializing session["all"]
        session["diseases"] = possible_diseases(session["all"])
        backup_session() # Backup after calculating possible diseases

        all_sym = session["all"] # Keep this variable for clarity

        if len(session["diseases"]) <= 1:
            session['step'] = "PREDICT"
            backup_session() # Backup after step change
            return get_bot_response() # Process the PREDICT step
        else:
            diseases = session["diseases"]
            session["dis"] = diseases[0]
            session['step'] = "for_dis"
            backup_session() # Backup after step change
            return get_bot_response() # Process the for_dis step


    if session.get('step') == "DIS":
        if "symv" in session:
            if len(s) > 0:
                symts = session["symv"]
                all_sym = session["all"]
                if s.lower() == "yes" and len(symts) > 0: # Check if symts has elements
                    all_sym.append(symts[0])
                    session["all"] = all_sym
                    backup_session() # Backup after updating session["all"]
                if len(symts) > 0: # Ensure symts is not empty before deleting
                    del symts[0]
                    session["symv"] = symts
                    backup_session() # Backup after updating session["symv"]

        # Iterate through symv using a loop instead of recursion
        while "symv" in session and len(session["symv"]) > 0:
            symts = session["symv"]
            if symts[0] not in session["all"] and symts[0] not in session["asked"]:
                asked = session["asked"]
                asked.append(symts[0])
                session["asked"] = asked
                backup_session() # Backup after updating session["asked"]
                msg = "do you feel " + clean_symp(symts[0]) + "?"
                return msg
            else:
                del symts[0]
                session["symv"] = symts
                backup_session() # Backup after updating session["symv"]

        # After the loop, check if there are still possible diseases
        PD = possible_diseases(session["all"])
        if len(PD) == 1: # If only one possible disease remains
            session["symv"] = [] # Clear symv as we are done with this disease
            backup_session() # Backup after clearing symv

        diseases = session.get("diseases", []) # Use .get with default empty list
        if session.get("dis") in PD: # Use .get for safer access
             PD.remove(session["dis"]) # Remove the current disease after processing its symptoms

        session["diseases"] = PD
        backup_session() # Backup after updating session["diseases"]
        session['step'] = "for_dis" # Move to check for the next disease
        backup_session() # Backup after step change
        return get_bot_response() # Process the for_dis step

    if session.get('step') == "for_dis":
        diseases = session.get("diseases", []) # Use .get with default empty list
        if len(diseases) <= 0 or len(possible_diseases(session.get("all", []))) <= 1: # Use .get for safer access
            session['step'] = 'PREDICT'
            backup_session() # Backup after step change
            return get_bot_response() # Process the PREDICT step
        else:
            session["dis"] = diseases[0]
            session['step'] = "DIS"
            session["symv"] = symVONdisease(df_tr, session["dis"])
            backup_session() # Backup after setting session["symv"] and step
            return get_bot_response()  # Turn around sympt of dis

    # predict possible diseases
    if session.get('step') == "PREDICT":
        try:
            ohv_data = OHV(session.get("all", []), all_symp_col) # Use .get with default empty list
            result = knn_clf.predict(ohv_data)
            session['step'] = "END"
            session["disease"] = result[0]
            backup_session() # Backup after setting session["disease"] and step
            return "Well Mr/Ms " + session.get("name", "Unknown") + ", you may have " + result[0] + ". Tap D to get a description of the disease ."
        except Exception as e: # Catch potential errors during prediction
            print(f"Prediction error: {e}")
            session['step'] = "Q_C" # Go to termination step
            backup_session() # Backup after step change
            return "An error occurred during prediction. Can you specify more what you feel or Tap q to stop the conversation"


    if session.get('step') == "END":
        # The previous step should set the disease
        if session.get("disease"): # Check if disease is set
            session['step'] = "Description"
            backup_session() # Backup after step change
            return get_bot_response() # Process the Description step
        else:
            session['step'] = "Q_C"  # test if user want to continue the conversation or not
            backup_session() # Backup after step change
            return "can you specify more what you feel or Tap q to stop the conversation"


    if session.get('step') == "Description":
        # Data saving logic
        y = {"Name": session.get("name", "Unknown"), "Age": session.get("age", 0), "Gender": session.get("gender", "Unknown"), "Disease": session.get("disease", "Unknown"),
             "Sympts": session.get("all", [])}
        write_json(y)

        session['step'] = "Severity"
        backup_session() # Backup after step change

        disease = session.get("disease", "Unknown") # Use .get
        if disease in description_list: # Check if disease is in dictionary
            return description_list[disease] + " \n <br>  How many days have you had symptoms?"
        else:
            # Format the disease name for the Wikipedia link
            wiki_disease = disease.replace(" ", "_")
            return f"Information about {disease} is not available. Please visit <a href='https://en.wikipedia.org/wiki/{wiki_disease}'>  here  </a> for more details. <br> How many days have you had symptoms?"

    if session.get('step') == "Severity":
        session['step'] = 'FINAL'
        backup_session() # Backup after step change
        try:
            days = int(s)
            if calc_condition(session.get("all", []), days) == 1: # Use .get
                return f'You should take the consultation from a doctor. <br> <a href="{url_for("Appointment")}">Click here to book an appointment</a> <br> Tap q to exit'
            else:
                disease = session.get("disease", "Unknown") # Use .get
                msg = 'Nothing to worry about, but you should take the following precautions :<br> '
                if disease in precautionDictionary: # Check if disease is in dictionary
                    i = 1
                    for e in precautionDictionary[disease]:
                        msg += f'\n {i} - {e}<br>' # Use f-string for cleaner formatting
                        i += 1
                else:
                    msg += "Precautions for this disease are not available."
                msg += ' Tap q to end'
                return msg
        except ValueError:
            return "Invalid input. Please enter the number of days as a number."


    if session.get('step') == "FINAL":
        session['step'] = "BYE"
        backup_session() # Backup after step change
        return "Your diagnosis was perfectly completed. Do you need another medical consultation (yes or no)? "

    if session.get('step') == "BYE":
        name = session.get("name", "Unknown")
        age = session.get("age", 0)
        gender = session.get("gender", "Unknown")
        session.clear() # Clear session first
        if s.lower() == "yes":
            session["gender"] = gender # Restore essential info
            session["name"] = name
            session["age"] = age
            session['step'] = "FS" # Restart from the first symptom
            backup_session() # Backup after restoring and changing step
            return "HELLO again Mr/Ms " + session.get("name", "Unknown") + " Please tell me your main symptom. "
        else:
            return "THANKS Mr/Ms " + name + " for using our service"

    # Add a default response if the step is not recognized
    return "I'm sorry, I didn't understand that. Can you please rephrase?"


if __name__ == "__main__":
    import random
    import string

    S = 10  # number of characters in the string.
    # call random.choices() string module to find the string in Uppercase + numeric data.
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k=S))
    # chat_sp()
    app.secret_key = str(ran)
    app.run(debug=True)