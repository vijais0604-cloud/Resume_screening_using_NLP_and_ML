from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import Form

import tempfile

from app.pdf_parser import (
    extract_text_from_pdf
)

from app.predictor import (
    predict_ats_score)

app = FastAPI()

@app.post("/predict")
async def predict(

    resume: UploadFile = File(...),

    job_description: str = Form(...)
):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp:

        temp.write(
            await resume.read()
        )

        pdf_path = temp.name

    resume_text = extract_text_from_pdf(
        pdf_path
    )

    result = predict_ats_score(

        resume_text,

        job_description
    )

    return result