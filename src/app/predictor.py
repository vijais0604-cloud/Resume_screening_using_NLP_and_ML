import joblib
import numpy as np


from sklearn.metrics.pairwise import cosine_similarity


from src.app.feature_engineering import (
    calculate_skill_score,
    calculate_experience_score
)

xgb_model = joblib.load(
    "src/model/best_xgboost_model.pkl"
)

embedding_model = joblib.load("src/model/sentence_transformer_model.joblib")

def predict_ats_score(
    resume_text,   
    job_text
):

    resume_embedding = (
        embedding_model.encode(
            resume_text
        )
    )

    job_embedding = (
        embedding_model.encode(
            job_text
        )
    )

    semantic_score = (
        cosine_similarity(
            [resume_embedding],
            [job_embedding]
        )[0][0]
        * 100
    )

    skill_score = (
        calculate_skill_score(
            resume_text,
            job_text
        )
    )

    experience_score = (
        calculate_experience_score(
            resume_text,
            job_text
        )
    )
    embedding = np.concatenate([
        resume_embedding,
        job_embedding
    ])

    features = np.concatenate([
        [skill_score],

        embedding,

        [semantic_score],

        [experience_score]

    ]).reshape(1, -1)

    score = xgb_model.predict(
        features
    )[0]

    return {

        "ats_score": round(
            float(score),
            2
        ),

        "skill_score": round(
            float(skill_score),
            2
        ),

        "semantic_score": round(
            float(semantic_score),
            2
        ),

        "experience_score": round(
            float(experience_score),
            2
        )
    }