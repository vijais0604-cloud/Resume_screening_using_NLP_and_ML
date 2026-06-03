import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

#Loading the dataset
dataset = pd.read_csv("data/resume_data.csv")

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


model = SentenceTransformer('all-MiniLM-L6-v2')


resume_embeddings =model.encode(resume["resume_text"].tolist())


job_embeddings =model.encode(job["job_text"].tolist())


resume_skills = resume["resume_text"].apply(lambda x: set(x.split()))

job_skills = job["job_text"].apply(lambda x: set(x.split()))





skill_scores = []

for i in range(len(dataset)):

    resume_skill_set = set(
        str(resume_skills[i]).split()
    )

    job_skill_set = set(
        str(job_skills[i]).split()
    )

    common_skills = resume_skill_set.intersection(
        job_skill_set
    )

    if len(job_skill_set) > 0:

        score = (
            len(common_skills) / len(job_skill_set)
        ) * 100

    else:
        score = 0

    skill_scores.append(score)


embeddings = np.concatenate(

    [resume_embeddings, job_embeddings],

    axis=1

)

all_text = pd.DataFrame()
all_text["embeddings"] = embeddings.tolist()

all_text['skill_score'] = skill_scores

semantic_scores = []

for i in range(len(dataset)):

    similarity = cosine_similarity(
        [resume_embeddings[i]],
        [job_embeddings[i]]
    )

    score = similarity[0][0] * 100

    semantic_scores.append(score)

all_text["semantic_score"] = semantic_scores



experience_scores = []

for i in range(len(dataset)):

    candidate_text = str(dataset["start_dates"][i])

    required_text = str(dataset["experiencere_requirement"][i])

    candidate_years = len(
        re.findall(r"\b(19|20)\d{2}\b", candidate_text)
    )

    required_years = re.findall(r"\d+", required_text)

    if len(required_years) > 0:

        required_years = int(required_years[0])

    else:
        required_years = 0

    if required_years > 0:

        score = min(
            (candidate_years / required_years) * 100,
            100
        )

    else:
        score = 0

    experience_scores.append(score)

all_text["experience_score"] = experience_scores



all_text["matched_score"] = target_col.astype(float)



