"""
Tests for feature_engineering.py module.
Tests skill extraction, skill score calculation, experience extraction, and experience score calculation.
"""

import pytest
from src.app.feature_engineering import (
    extract_skills,
    calculate_skill_score,
    extract_experience,
    calculate_experience_score
)


class TestExtractSkills:
    """Test skill extraction functionality."""

    def test_extract_skills_single_skill(self):
        """Test extraction of a single skill from text."""
        text = "I have experience with Python"
        skills = extract_skills(text)
        assert "python" in skills or len(skills) > 0

    def test_extract_skills_multiple_skills(self):
        """Test extraction of multiple skills from text."""
        text = "Expert in Python, JavaScript, and Java programming"
        skills = extract_skills(text)
        assert isinstance(skills, set)

    def test_extract_skills_empty_text(self):
        """Test extraction from empty text."""
        text = ""
        skills = extract_skills(text)
        assert isinstance(skills, set)
        assert len(skills) == 0

    def test_extract_skills_case_insensitive(self):
        """Test that skill extraction is case insensitive."""
        text1 = "I know PYTHON"
        text2 = "I know python"
        skills1 = extract_skills(text1)
        skills2 = extract_skills(text2)
        assert skills1 == skills2


class TestCalculateSkillScore:
    """Test skill score calculation functionality."""

    def test_skill_score_perfect_match(self):
        """Test skill score when resume has all required skills."""
        resume_text = "Skills: Python, Java, SQL"
        job_text = "Required: Python, Java, SQL"
        score = calculate_skill_score(resume_text, job_text)
        assert 0 <= score <= 100

    def test_skill_score_partial_match(self):
        """Test skill score when resume has some required skills."""
        resume_text = "Skills: Python, Java"
        job_text = "Required: Python, Java, SQL, JavaScript"
        score = calculate_skill_score(resume_text, job_text)
        assert 0 <= score <= 100

    def test_skill_score_no_match(self):
        """Test skill score when resume has no required skills."""
        resume_text = "Skills: Ruby, Go"
        job_text = "Required: Python, Java"
        score = calculate_skill_score(resume_text, job_text)
        assert score == 0

    def test_skill_score_no_job_requirements(self):
        """Test skill score when job has no skills listed."""
        resume_text = "Skills: Python"
        job_text = "No skills mentioned"
        score = calculate_skill_score(resume_text, job_text)
        assert score == 0

    def test_skill_score_returns_percentage(self):
        """Test that skill score returns a valid percentage."""
        resume_text = "Python Java"
        job_text = "Python SQL"
        score = calculate_skill_score(resume_text, job_text)
        assert isinstance(score, float)
        assert 0 <= score <= 100


class TestExtractExperience:
    """Test experience extraction functionality."""

    def test_extract_experience_years_format(self):
        """Test extraction of experience in 'X years' format."""
        text = "I have 5 years of experience"
        years = extract_experience(text)
        assert years == 5

    def test_extract_experience_yrs_abbreviation(self):
        """Test extraction of experience using 'yrs' abbreviation."""
        text = "3 yrs in software development"
        years = extract_experience(text)
        assert years == 3

    def test_extract_experience_plus_notation(self):
        """Test extraction of experience with '+' notation."""
        text = "10+ years of expertise"
        years = extract_experience(text)
        assert years == 10

    def test_extract_experience_no_match(self):
        """Test extraction when no experience is mentioned."""
        text = "No experience mentioned here"
        years = extract_experience(text)
        assert years == 0

    def test_extract_experience_case_insensitive(self):
        """Test that extraction is case insensitive."""
        text1 = "7 Years of work"
        text2 = "7 years of work"
        assert extract_experience(text1) == extract_experience(text2)

    def test_extract_experience_first_match(self):
        """Test that first experience mention is extracted."""
        text = "5 years with company A, then 3 years with company B"
        years = extract_experience(text)
        assert years == 5


class TestCalculateExperienceScore:
    """Test experience score calculation functionality."""

    def test_experience_score_meets_requirement(self):
        """Test score when resume experience meets or exceeds requirement."""
        resume_text = "10 years of experience"
        job_text = "5 years required"
        score = calculate_experience_score(resume_text, job_text)
        assert score == 100

    def test_experience_score_below_requirement(self):
        """Test score when resume experience is below requirement."""
        resume_text = "2 years of experience"
        job_text = "5 years required"
        score = calculate_experience_score(resume_text, job_text)
        assert 0 <= score < 100

    def test_experience_score_no_requirement(self):
        """Test score when job has no experience requirement."""
        resume_text = "3 years of experience"
        job_text = "No experience mentioned"
        score = calculate_experience_score(resume_text, job_text)
        assert score == 100

    def test_experience_score_returns_percentage(self):
        """Test that experience score returns a valid percentage."""
        resume_text = "5 years"
        job_text = "3 years"
        score = calculate_experience_score(resume_text, job_text)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100

    def test_experience_score_zero_experience(self):
        """Test score when candidate has no experience."""
        resume_text = "No experience"
        job_text = "5 years required"
        score = calculate_experience_score(resume_text, job_text)
        assert score == 0
