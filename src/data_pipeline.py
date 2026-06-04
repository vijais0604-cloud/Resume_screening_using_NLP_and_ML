import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import json
import os

if os.path.exists("src/skills.json"):
    with open("src/skills.json", "r") as f:
        SKILLS = set(json.load(f))

#Loading the dataset
dataset = pd.read_csv("data/resume_data.csv")

all_text = pd.DataFrame()

#cleaning the column names
dataset.columns = dataset.columns.str.strip()

dataset.columns = dataset.columns.str.replace(r"[﻿1./]", "", regex=True)

dataset.columns = dataset.columns.str.replace(" ","", regex=True)

dataset.columns = dataset.columns.str.lower()


drop_cols = [
    "online_links",
    "extra_curricular_organization_names",
    "extra_curricular_organization_links",
    "address",
    "company_urls",
    "issue_dates",
    "expiry_dates",
    "locations",
]

dataset = dataset.drop(columns=drop_cols)    

target_col = dataset["matched_score"]

dataset = dataset.fillna("")



def clean_text(text):

    text = str(text)

    text = re.sub(r"\[|\]|'", " ", text)

    text = re.sub(r"\bnone\b", " ", text)

    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"\bn\s*a\b", " ", text)

    return text.strip().lower()


dataset = dataset.apply(
    lambda col: col.map(clean_text)
)

dataset = dataset.replace(r"\bnone\b", "", regex=True)




resume = dataset[['career_objective',
 'skills',
 'educational_institution_name',
 'degree_names',
 'passing_years',
 'educational_results',
 'result_types',
 'major_field_of_studies',
 'professional_company_names',
 'start_dates',
 'end_dates',
 'related_skils_in_job',
 'positions',
 'responsibilities',
 'extra_curricular_activity_types',
 'role_positions',
 'languages',
 'proficiency_levels',
 'certification_providers',
 'certification_skills',
]]


job = dataset[['job_position_name',
 'educational_requirements',
 'experiencere_requirement',
 'age_requirement',
 'responsibilities',
 'skills_required']]





resume["resume_text"] = resume.astype(str).agg(" ".join, axis=1)

job["job_text"] = job.astype(str).agg(" ".join, axis=1)

dataset["resume_text"] = resume["resume_text"]
dataset["job_text"] = job["job_text"]




model = SentenceTransformer('all-MiniLM-L6-v2')

joblib.dump(model, 'sentence_transformer_model.joblib')

resume_embeddings =model.encode(resume["resume_text"].tolist())

job_embeddings =model.encode(job["job_text"].tolist())



def extract_skills(text):

    text = text.lower()

    found_skills = set()

    for skill in SKILLS:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):

            found_skills.add(skill)

    return found_skills

def calculate_skill_score(resume_text,job_text):

    resume_skills = extract_skills(resume_text)

    job_skills = extract_skills(job_text)

    if len(job_skills) == 0:
        return 0

    matched = len(
        resume_skills.intersection(
            job_skills
        )
    )

    score = (
        matched / len(job_skills)
    ) * 100

    return round(score,2)


all_text["skill_score"] = dataset.apply(
    lambda row: calculate_skill_score(
        row["resume_text"],
        row["job_text"]
    ),
    axis=1
)


embeddings = np.concatenate(

    [resume_embeddings, job_embeddings],

    axis=1

)

all_text["embeddings"] = embeddings.tolist()


semantic_scores = []

for i in range(len(dataset)):

    similarity = cosine_similarity(
        [resume_embeddings[i]],
        [job_embeddings[i]]
    )

    score = similarity[0][0] * 100

    semantic_scores.append(score)

all_text["semantic_score"] = semantic_scores



def extract_experience(text):

    text = text.lower()

    patterns = [

        r"(\d+)\+?\s*years",

        r"(\d+)\+?\s*yrs",

        r"(\d+)\+?\s*year",

        r"experience\s*:\s*(\d+)"

    ]

    for pattern in patterns:

        match = re.search(

            pattern,

            text

        )

        if match:

            return int(

                match.group(1)

            )

    return 0

def calculate_experience_score(resume_text,job_text):

    candidate_exp = extract_experience(resume_text)

    required_exp = extract_experience(job_text)

    if required_exp == 0:

        return 100

    score = min(

        (candidate_exp / required_exp)

        * 100,

        100

    )

    return round(score,2)

all_text["experience_score"] = dataset.apply(
    lambda row: calculate_experience_score(
        row["resume_text"],
        row["job_text"]
    ),
    axis=1
)

all_text["matched_score"] = target_col.astype(float)



all_text.to_csv("data/embedded_data.csv", index=False)