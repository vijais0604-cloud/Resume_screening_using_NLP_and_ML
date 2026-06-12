"""
Tests for predictor.py module.
Tests the predict_ats_score function and its component calculations.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.app.predictor import predict_ats_score


class TestPredictATSScore:
    """Test the predict_ats_score function."""

    @patch('src.app.predictor.xgb_model')
    @patch('src.app.predictor.embedding_model')
    @patch('src.app.predictor.calculate_skill_score')
    @patch('src.app.predictor.calculate_experience_score')
    @patch('src.app.predictor.cosine_similarity')
    def test_predict_ats_score_returns_dict(
        self, mock_cosine, mock_exp_score, mock_skill_score, 
        mock_embed_model, mock_xgb
    ):
        """Test that predict_ats_score returns a dictionary with required keys."""
        # Setup mocks
        mock_embed_model.encode.return_value = np.random.rand(384)
        mock_cosine.return_value = np.array([[0.85]])
        mock_skill_score.return_value = 80.0
        mock_exp_score.return_value = 90.0
        mock_xgb.predict.return_value = np.array([75.5])

        result = predict_ats_score(
            "Sample resume text",
            "Sample job description"
        )

        assert isinstance(result, dict)
        assert "ats_score" in result
        assert "skill_score" in result
        assert "semantic_score" in result
        assert "experience_score" in result

    @patch('src.app.predictor.xgb_model')
    @patch('src.app.predictor.embedding_model')
    @patch('src.app.predictor.calculate_skill_score')
    @patch('src.app.predictor.calculate_experience_score')
    @patch('src.app.predictor.cosine_similarity')
    def test_predict_ats_score_values_are_floats(
        self, mock_cosine, mock_exp_score, mock_skill_score, 
        mock_embed_model, mock_xgb
    ):
        """Test that all returned scores are floats."""
        mock_embed_model.encode.return_value = np.random.rand(384)
        mock_cosine.return_value = np.array([[0.75]])
        mock_skill_score.return_value = 85.0
        mock_exp_score.return_value = 70.0
        mock_xgb.predict.return_value = np.array([80.0])

        result = predict_ats_score("resume", "job")

        assert isinstance(result["ats_score"], float)
        assert isinstance(result["skill_score"], float)
        assert isinstance(result["semantic_score"], float)
        assert isinstance(result["experience_score"], float)

    @patch('src.app.predictor.xgb_model')
    @patch('src.app.predictor.embedding_model')
    @patch('src.app.predictor.calculate_skill_score')
    @patch('src.app.predictor.calculate_experience_score')
    @patch('src.app.predictor.cosine_similarity')
    def test_predict_ats_score_semantic_score_calculation(
        self, mock_cosine, mock_exp_score, mock_skill_score, 
        mock_embed_model, mock_xgb
    ):
        """Test that semantic score is calculated correctly as percentage."""
        mock_embed_model.encode.return_value = np.random.rand(384)
        mock_cosine.return_value = np.array([[0.92]])  # 92% similarity
        mock_skill_score.return_value = 75.0
        mock_exp_score.return_value = 80.0
        mock_xgb.predict.return_value = np.array([78.0])

        result = predict_ats_score("resume", "job")

        # Semantic score should be 92.0 (0.92 * 100)
        assert result["semantic_score"] == 92.0

    @patch('src.app.predictor.xgb_model')
    @patch('src.app.predictor.embedding_model')
    @patch('src.app.predictor.calculate_skill_score')
    @patch('src.app.predictor.calculate_experience_score')
    @patch('src.app.predictor.cosine_similarity')
    def test_predict_ats_score_calls_all_components(
        self, mock_cosine, mock_exp_score, mock_skill_score, 
        mock_embed_model, mock_xgb
    ):
        """Test that all component scoring functions are called."""
        mock_embed_model.encode.return_value = np.random.rand(384)
        mock_cosine.return_value = np.array([[0.8]])
        mock_skill_score.return_value = 85.0
        mock_exp_score.return_value = 90.0
        mock_xgb.predict.return_value = np.array([82.0])

        predict_ats_score("resume text", "job description")

        # Verify all scoring functions were called
        assert mock_skill_score.called
        assert mock_exp_score.called
        assert mock_xgb.predict.called
        assert mock_cosine.called

    @patch('src.app.predictor.xgb_model')
    @patch('src.app.predictor.embedding_model')
    @patch('src.app.predictor.calculate_skill_score')
    @patch('src.app.predictor.calculate_experience_score')
    @patch('src.app.predictor.cosine_similarity')
    def test_predict_ats_score_rounding_precision(
        self, mock_cosine, mock_exp_score, mock_skill_score, 
        mock_embed_model, mock_xgb
    ):
        """Test that scores are rounded to 2 decimal places."""
        mock_embed_model.encode.return_value = np.random.rand(384)
        mock_cosine.return_value = np.array([[0.856789]])
        mock_skill_score.return_value = 82.3456
        mock_exp_score.return_value = 75.9999
        mock_xgb.predict.return_value = np.array([79.3456])

        result = predict_ats_score("resume", "job")

        # All scores should have at most 2 decimal places
        assert len(str(result["ats_score"]).split('.')[-1]) <= 2
        assert len(str(result["skill_score"]).split('.')[-1]) <= 2
        assert len(str(result["semantic_score"]).split('.')[-1]) <= 2
        assert len(str(result["experience_score"]).split('.')[-1]) <= 2

    @patch('src.app.predictor.xgb_model')
    @patch('src.app.predictor.embedding_model')
    @patch('src.app.predictor.calculate_skill_score')
    @patch('src.app.predictor.calculate_experience_score')
    @patch('src.app.predictor.cosine_similarity')
    def test_predict_ats_score_edge_case_zero_scores(
        self, mock_cosine, mock_exp_score, mock_skill_score, 
        mock_embed_model, mock_xgb
    ):
        """Test prediction with zero scores."""
        mock_embed_model.encode.return_value = np.random.rand(384)
        mock_cosine.return_value = np.array([[0.0]])
        mock_skill_score.return_value = 0.0
        mock_exp_score.return_value = 0.0
        mock_xgb.predict.return_value = np.array([0.0])

        result = predict_ats_score("resume", "job")

        assert result["ats_score"] == 0.0
        assert result["skill_score"] == 0.0
        assert result["semantic_score"] == 0.0
        assert result["experience_score"] == 0.0

    @patch('src.app.predictor.xgb_model')
    @patch('src.app.predictor.embedding_model')
    @patch('src.app.predictor.calculate_skill_score')
    @patch('src.app.predictor.calculate_experience_score')
    @patch('src.app.predictor.cosine_similarity')
    def test_predict_ats_score_edge_case_perfect_scores(
        self, mock_cosine, mock_exp_score, mock_skill_score, 
        mock_embed_model, mock_xgb
    ):
        """Test prediction with perfect scores."""
        mock_embed_model.encode.return_value = np.random.rand(384)
        mock_cosine.return_value = np.array([[1.0]])
        mock_skill_score.return_value = 100.0
        mock_exp_score.return_value = 100.0
        mock_xgb.predict.return_value = np.array([100.0])

        result = predict_ats_score("resume", "job")

        assert result["ats_score"] == 100.0
        assert result["skill_score"] == 100.0
        assert result["semantic_score"] == 100.0
        assert result["experience_score"] == 100.0

    @patch('src.app.predictor.xgb_model')
    @patch('src.app.predictor.embedding_model')
    @patch('src.app.predictor.calculate_skill_score')
    @patch('src.app.predictor.calculate_experience_score')
    @patch('src.app.predictor.cosine_similarity')
    def test_predict_ats_score_feature_concatenation(
        self, mock_cosine, mock_exp_score, mock_skill_score, 
        mock_embed_model, mock_xgb
    ):
        """Test that features are properly concatenated for model input."""
        embedding = np.random.rand(384)
        mock_embed_model.encode.return_value = embedding
        mock_cosine.return_value = np.array([[0.8]])
        mock_skill_score.return_value = 75.0
        mock_exp_score.return_value = 85.0
        mock_xgb.predict.return_value = np.array([80.0])

        predict_ats_score("resume", "job")

        # Verify xgb_model.predict was called with feature array
        assert mock_xgb.predict.called
        call_args = mock_xgb.predict.call_args[0][0]
        assert call_args.shape[0] == 1  # Single prediction

    @patch('src.app.predictor.xgb_model')
    @patch('src.app.predictor.embedding_model')
    @patch('src.app.predictor.calculate_skill_score')
    @patch('src.app.predictor.calculate_experience_score')
    @patch('src.app.predictor.cosine_similarity')
    def test_predict_ats_score_with_empty_strings(
        self, mock_cosine, mock_exp_score, mock_skill_score, 
        mock_embed_model, mock_xgb
    ):
        """Test prediction with empty strings."""
        mock_embed_model.encode.return_value = np.random.rand(384)
        mock_cosine.return_value = np.array([[0.0]])
        mock_skill_score.return_value = 0.0
        mock_exp_score.return_value = 0.0
        mock_xgb.predict.return_value = np.array([5.0])

        result = predict_ats_score("", "")

        assert isinstance(result, dict)
        assert "ats_score" in result
