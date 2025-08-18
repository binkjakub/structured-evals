import asyncio
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.message import BaseMessagePromptTemplate
from pydantic import BaseModel
from pydantic.fields import Field
from tenacity import retry, stop_after_attempt, wait_exponential

from structured_evals.base import EvaluatorBase, ItemEvalOutput

DEFAULT_MAX_CONCURRENT_CALLS = 30
DEFAULT_SYSTEM_PROMPT = "You are a judge that scores the quality of the prediction."
DEFAULT_PROMPT = """
Score the quality of the prediction based on Reference Answer.
Reference Answer: {target}
Prediction: {pred}
"""


class JudgeScore(BaseModel):
    score: float = Field(..., description="The score of the prediction, either 0 or 1")


class LlmAsJudge(EvaluatorBase[str, ItemEvalOutput]):
    def __init__(
        self,
        llm: BaseChatModel,
        prompt: str = DEFAULT_PROMPT,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
    ) -> None:
        super().__init__(f"LlmAsJudge(llm={llm.name})")
        self.llm = llm

        messages: list[BaseMessage | BaseMessagePromptTemplate] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(
            HumanMessagePromptTemplate.from_template(prompt, template_format="f-string")
        )

        self.prompt_template = ChatPromptTemplate.from_messages(messages)
        self.chain = self.prompt_template | self.llm.with_structured_output(JudgeScore)

        self.semaphore = asyncio.Semaphore(max_concurrent_calls)

    @property
    def zero_score(self) -> ItemEvalOutput:
        return ItemEvalOutput(score=0.0)

    @property
    def max_score(self) -> ItemEvalOutput:
        return ItemEvalOutput(score=1.0)

    def evaluate_batch(self, pred: list[str], target: list[str]) -> list[ItemEvalOutput]:
        return asyncio.run(self._async_evaluate_batch(pred, target))

    async def _async_evaluate_batch(
        self,
        pred: list[str],
        target: list[str],
    ) -> list[ItemEvalOutput]:
        return await asyncio.gather(
            *[self.async_evaluate(pred, target) for pred, target in zip(pred, target, strict=True)]
        )

    def evaluate(self, pred: str, target: str) -> ItemEvalOutput:
        bypass_output = self._bypass_llm(pred, target)
        if bypass_output is not None:
            return bypass_output

        res = self._call_llm(pred=pred, target=target)
        return ItemEvalOutput(score=res.score)

    async def async_evaluate(self, pred: str, target: str) -> ItemEvalOutput:
        bypass_output = self._bypass_llm(pred, target)
        if bypass_output is not None:
            return bypass_output

        res = await self._async_call_llm(pred=pred, target=target)
        return ItemEvalOutput(score=res.score)

    def is_null(self, item: str | None) -> bool:
        return item is None or item == ""

    def check_dtype(self, pred: Any, target: Any) -> bool:
        return isinstance(pred, str) and isinstance(target, str)

    def _bypass_llm(self, pred: str, target: str) -> ItemEvalOutput | None:
        if pred == target:
            return ItemEvalOutput(score=1.0)
        elif self.is_null(pred) or self.is_null(target):
            return ItemEvalOutput(score=0.0)
        if not self.check_dtype(pred, target):
            return ItemEvalOutput(score=0.0)
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_llm(self, **template_kwargs: Any) -> JudgeScore:
        return self.chain.invoke(template_kwargs)  # type: ignore

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _async_call_llm(self, **template_kwargs: Any) -> JudgeScore:
        async with self.semaphore:
            return await self.chain.ainvoke(template_kwargs)  # type: ignore
