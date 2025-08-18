import asyncio
from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate

from structured_evals.base import ItemEvalOutput
from structured_evals.eval_llm_as_judge import (
    DEFAULT_MAX_CONCURRENT_CALLS,
    JudgeScore,
    LlmAsJudge,
)


@pytest.fixture
def mock_llm() -> Mock:
    """Create a mock LLM for testing."""
    llm = Mock(spec=BaseChatModel)
    llm.model_name = "test-model"

    mock_chain = Mock()
    llm.with_structured_output.return_value = mock_chain

    return llm


@pytest.fixture
def judge_score_response() -> JudgeScore:
    """Create a mock JudgeScore response."""
    return JudgeScore(score=0.8)


def test_init_with_defaults(mock_llm: Mock) -> None:
    """Test initialization with default parameters."""
    judge = LlmAsJudge(llm=mock_llm)

    assert judge.llm == mock_llm
    assert judge.semaphore._value == DEFAULT_MAX_CONCURRENT_CALLS
    assert isinstance(judge.prompt_template, ChatPromptTemplate)
    assert judge.name == "LlmAsJudge(llm=test-model)"


def test_zero_score(mock_llm: Mock) -> None:
    """Test zero_score property returns correct ItemEvalOutput."""
    judge = LlmAsJudge(llm=mock_llm)
    zero_score = judge.zero_score

    assert isinstance(zero_score, ItemEvalOutput)
    assert zero_score.score == 0.0


def test_max_score(mock_llm: Mock) -> None:
    """Test max_score property returns correct ItemEvalOutput."""
    judge = LlmAsJudge(llm=mock_llm)
    max_score = judge.max_score

    assert isinstance(max_score, ItemEvalOutput)
    assert max_score.score == 1.0


class TestLlmAsJudgeUtilityMethods:
    """Test utility methods of LlmAsJudge."""

    def test_is_null_with_none(self, mock_llm: Mock) -> None:
        """Test is_null method with None value."""
        judge = LlmAsJudge(llm=mock_llm)
        assert judge.is_null(None) is True

    def test_is_null_with_empty_string(self, mock_llm: Mock) -> None:
        """Test is_null method with empty string."""
        judge = LlmAsJudge(llm=mock_llm)
        assert judge.is_null("") is True

    def test_is_null_with_valid_string(self, mock_llm: Mock) -> None:
        """Test is_null method with valid string."""
        judge = LlmAsJudge(llm=mock_llm)
        assert judge.is_null("valid text") is False

    def test_check_dtype_with_valid_strings(self, mock_llm: Mock) -> None:
        """Test check_dtype method with valid string inputs."""
        judge = LlmAsJudge(llm=mock_llm)
        assert judge.check_dtype("pred", "target") is True

    def test_check_dtype_with_invalid_types(self, mock_llm: Mock) -> None:
        """Test check_dtype method with invalid types."""
        judge = LlmAsJudge(llm=mock_llm)
        assert judge.check_dtype(123, "target") is False
        assert judge.check_dtype("pred", 456) is False
        assert judge.check_dtype(123, 456) is False


class TestLlmAsJudgeBypassLogic:
    """Test _bypass_llm method logic."""

    def test_bypass_with_identical_strings(self, mock_llm: Mock) -> None:
        """Test bypass logic when pred equals target."""
        judge = LlmAsJudge(llm=mock_llm)
        result = judge._bypass_llm("same text", "same text")

        assert isinstance(result, ItemEvalOutput)
        assert result.score == 1.0

    def test_bypass_with_null_pred(self, mock_llm: Mock) -> None:
        """Test bypass logic when pred is null."""
        judge = LlmAsJudge(llm=mock_llm)
        result = judge._bypass_llm(None, "target")  # type: ignore[arg-type]

        assert isinstance(result, ItemEvalOutput)
        assert result.score == 0.0

    def test_bypass_with_empty_pred(self, mock_llm: Mock) -> None:
        """Test bypass logic when pred is empty string."""
        judge = LlmAsJudge(llm=mock_llm)
        result = judge._bypass_llm("", "target")

        assert isinstance(result, ItemEvalOutput)
        assert result.score == 0.0

    def test_bypass_with_null_target(self, mock_llm: Mock) -> None:
        """Test bypass logic when target is null."""
        judge = LlmAsJudge(llm=mock_llm)
        result = judge._bypass_llm("pred", None)  # type: ignore[arg-type]

        assert isinstance(result, ItemEvalOutput)
        assert result.score == 0.0

    def test_bypass_with_empty_target(self, mock_llm: Mock) -> None:
        """Test bypass logic when target is empty string."""
        judge = LlmAsJudge(llm=mock_llm)
        result = judge._bypass_llm("pred", "")

        assert isinstance(result, ItemEvalOutput)
        assert result.score == 0.0

    def test_bypass_with_invalid_types(self, mock_llm: Mock) -> None:
        """Test bypass logic when inputs have invalid types."""
        judge = LlmAsJudge(llm=mock_llm)
        result = judge._bypass_llm(123, "target")  # type: ignore[arg-type]

        assert isinstance(result, ItemEvalOutput)
        assert result.score == 0.0

    def test_bypass_returns_none_for_valid_different_strings(self, mock_llm: Mock) -> None:
        """Test bypass logic returns None for valid different strings."""
        judge = LlmAsJudge(llm=mock_llm)
        result = judge._bypass_llm("different", "strings")

        assert result is None


class TestLlmAsJudgeEvaluation:
    """Test evaluation methods of LlmAsJudge."""

    def test_evaluate_with_bypass(self, mock_llm: Mock) -> None:
        """Test evaluate method when bypass logic applies."""
        judge = LlmAsJudge(llm=mock_llm)
        result = judge.evaluate("same", "same")

        assert isinstance(result, ItemEvalOutput)
        assert result.score == 1.0
        mock_llm.with_structured_output.assert_called_once()

    @patch.object(LlmAsJudge, "_call_llm")
    def test_evaluate_with_llm_call(
        self, mock_call_llm: Mock, mock_llm: Mock, judge_score_response: JudgeScore
    ) -> None:
        """Test evaluate method when LLM is called."""
        mock_call_llm.return_value = judge_score_response
        judge = LlmAsJudge(llm=mock_llm)

        result = judge.evaluate("pred", "target")

        assert isinstance(result, ItemEvalOutput)
        assert result.score == 0.8
        mock_call_llm.assert_called_once_with(pred="pred", target="target")


class TestLlmAsJudgeBatchEvaluation:
    """Test batch evaluation methods of LlmAsJudge."""

    @patch.object(LlmAsJudge, "_async_evaluate_batch")
    def test_evaluate_batch(self, mock_async_batch: Mock, mock_llm: Mock) -> None:
        """Test evaluate_batch method."""
        expected_results = [ItemEvalOutput(score=1.0), ItemEvalOutput(score=0.5)]
        mock_async_batch.return_value = expected_results
        judge = LlmAsJudge(llm=mock_llm)

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = expected_results
            results = judge.evaluate_batch(["pred1", "pred2"], ["target1", "target2"])

        assert results == expected_results
        mock_run.assert_called_once()

    @pytest.mark.filterwarnings(
        "ignore:coroutine 'AsyncMockMixin._execute_mock_call' was never awaited:RuntimeWarning"
    )
    def test_async_evaluate_batch(self, mock_llm: Mock) -> None:
        """Test _async_evaluate_batch method."""
        judge = LlmAsJudge(llm=mock_llm)

        # Mock the async_evaluate method directly on the instance
        call_count = 0

        async def mock_async_evaluate(pred: str, target: str) -> ItemEvalOutput:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ItemEvalOutput(score=1.0)
            else:
                return ItemEvalOutput(score=0.5)

        judge.async_evaluate = mock_async_evaluate  # type: ignore[assignment]

        async def run_test() -> None:
            results = await judge._async_evaluate_batch(["pred1", "pred2"], ["target1", "target2"])

            assert len(results) == 2
            assert results[0].score == 1.0
            assert results[1].score == 0.5
            assert call_count == 2

        asyncio.run(run_test())
