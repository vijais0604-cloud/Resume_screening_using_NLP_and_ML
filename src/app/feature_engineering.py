import json

import re

with open("src/skills.json", "r") as f:

    SKILLS = set(json.load(f))

def extract_skills(text):

    text = text.lower()

    found_skills = set()

    for skill in SKILLS:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):

            found_skills.add(skill)

    return found_skills


def calculate_skill_score(
    resume_text,
    job_text
):

    resume_skills = extract_skills(
        resume_text
    )

    job_skills = extract_skills(
        job_text
    )

    if len(job_skills) == 0:

        return 0

    return (
        len(
            resume_skills.intersection(
                job_skills
            )
        )
        /
        len(job_skills)
    ) * 100


def extract_experience(text):

    patterns = [

        r"(\d+)\+?\s*years",

        r"(\d+)\+?\s*yrs",

        r"(\d+)\+?\s*year"
    ]

    text = text.lower()

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



def calculate_experience_score(
    resume_text,
    job_text
):

    candidate_exp = extract_experience(
        resume_text
    )

    required_exp = extract_experience(
        job_text
    )

    if required_exp == 0:

        return 100

    return min(
        candidate_exp
        /
        required_exp
        * 100,
        100
    )

