"""
Tests for main.py (FastAPI application).
Tests the FastAPI endpoint and its integration with other modules.
"""

import pytest
from fastapi.testclient import TestClient
from src.app.main import app
from unittest.mock import patch, MagicMock
import io


client = TestClient(app)


class TestPredictEndpoint:
    """Test the /predict endpoint."""

    @patch('src.app.main.extract_text_from_pdf')
    @patch('src.app.main.predict_ats_score')
    def test_predict_endpoint_success(self, mock_predict, mock_extract):
        """Test successful prediction endpoint call."""
        # Mock the functions
        mock_extract.return_value = "Sample resume text"
        mock_predict.return_value = {
            "ats_score": 85.5,
            "skill_score": 90.0,
            "semantic_score": 88.0,
            "experience_score": 80.0
        }

        # Create a fake PDF file
        pdf_content = b"fake pdf content"
        
        response = client.post(
            "/predict",
            data={"job_description": "Python developer role"},
            files={"resume": ("resume.pdf", io.BytesIO(pdf_content), "application/pdf")}
        )

        assert response.status_code == 200
        data = response.json()
        assert "ats_score" in data
        assert "skill_score" in data
        assert "semantic_score" in data
        assert "experience_score" in data

    @patch('src.app.main.extract_text_from_pdf')
    @patch('src.app.main.predict_ats_score')
    def test_predict_endpoint_missing_resume(self, mock_predict, mock_extract):
        """Test endpoint when resume file is missing."""
        response = client.post(
            "/predict",
            data={"job_description": "Python developer role"}
        )

        # Should return 422 Unprocessable Entity due to missing file
        assert response.status_code == 422

    @patch('src.app.main.extract_text_from_pdf')
    @patch('src.app.main.predict_ats_score')
    def test_predict_endpoint_missing_job_description(self, mock_predict, mock_extract):
        """Test endpoint when job description is missing."""
        pdf_content = b"fake pdf content"
        
        response = client.post(
            "/predict",
            files={"resume": ("resume.pdf", io.BytesIO(pdf_content), "application/pdf")}
        )

        # Should return 422 Unprocessable Entity due to missing form data
        assert response.status_code == 422

    @patch('src.app.main.extract_text_from_pdf')
    @patch('src.app.main.predict_ats_score')
    def test_predict_endpoint_response_format(self, mock_predict, mock_extract):
        """Test that response has correct format and data types."""
        mock_extract.return_value = "Resume content"
        mock_predict.return_value = {
            "ats_score": 75.25,
            "skill_score": 80.0,
            "semantic_score": 85.5,
            "experience_score": 70.0
        }

        pdf_content = b"fake pdf"
        
        response = client.post(
            "/predict",
            data={"job_description": "Senior Developer"},
            files={"resume": ("resume.pdf", io.BytesIO(pdf_content), "application/pdf")}
        )

        assert response.status_code == 200
        data = response.json()
        
        # Check all scores are numeric
        assert isinstance(data["ats_score"], (int, float))
        assert isinstance(data["skill_score"], (int, float))
        assert isinstance(data["semantic_score"], (int, float))
        assert isinstance(data["experience_score"], (int, float))

    @patch('src.app.main.extract_text_from_pdf')
    @patch('src.app.main.predict_ats_score')
    def test_predict_endpoint_calls_correct_functions(self, mock_predict, mock_extract):
        """Test that endpoint calls the expected functions."""
        mock_extract.return_value = "Extracted text"
        mock_predict.return_value = {
            "ats_score": 90.0,
            "skill_score": 85.0,
            "semantic_score": 88.0,
            "experience_score": 95.0
        }

        pdf_content = b"fake pdf"
        job_desc = "Machine Learning Engineer"
        
        client.post(
            "/predict",
            data={"job_description": job_desc},
            files={"resume": ("resume.pdf", io.BytesIO(pdf_content), "application/pdf")}
        )

        # Verify extract_text_from_pdf was called
        assert mock_extract.called

        # Verify predict_ats_score was called with correct arguments
        assert mock_predict.called
        call_args = mock_predict.call_args
        assert call_args[0][1] == job_desc  # job_description should match


class TestEndpointIntegration:
    """Test integration between endpoint and other modules."""

    @patch('src.app.main.extract_text_from_pdf')
    @patch('src.app.main.predict_ats_score')
    def test_endpoint_handles_large_resume(self, mock_predict, mock_extract):
        """Test endpoint can handle large resume content."""
        large_text = "Experience " * 1000  # Large resume text
        mock_extract.return_value = large_text
        mock_predict.return_value = {
            "ats_score": 65.0,
            "skill_score": 70.0,
            "semantic_score": 60.0,
            "experience_score": 75.0
        }

        pdf_content = b"large pdf content" * 100
        
        response = client.post(
            "/predict",
            data={"job_description": "Any role"},
            files={"resume": ("resume.pdf", io.BytesIO(pdf_content), "application/pdf")}
        )

        assert response.status_code == 200

    @patch('src.app.main.extract_text_from_pdf')
    @patch('src.app.main.predict_ats_score')
    def test_endpoint_handles_special_characters(self, mock_predict, mock_extract):
        """Test endpoint can handle special characters in job description."""
        mock_extract.return_value = "Resume text"
        mock_predict.return_value = {
            "ats_score": 72.0,
            "skill_score": 75.0,
            "semantic_score": 70.0,
            "experience_score": 72.0
        }

        pdf_content = b"pdf"
        job_desc = "C++/C# Developer | $100k-$150k | (Experience Required)"
        
        response = client.post(
            "/predict",
            data={"job_description": job_desc},
            files={"resume": ("resume.pdf", io.BytesIO(pdf_content), "application/pdf")}
        )

        assert response.status_code == 200
        assert response.json()["ats_score"] == 72.0
