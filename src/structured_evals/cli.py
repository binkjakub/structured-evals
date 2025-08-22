"""CLI for structured evaluations using typer."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from loguru import logger

from structured_evals import (
    EvaluationBatch,
    infer_structured_evaluator_from_predictions,
    infer_structured_evaluator_from_schema,
)
from structured_evals.aggregations import AverageAggregation
from structured_evals.eval_batch import BatchDictEval
from structured_evals.eval_dict import DictEval
from structured_evals.report import EvaluationReport

app = typer.Typer(help="Structured evaluations CLI for evaluating LLM structured outputs")


def setup_cache() -> None:
    """Setup LLM cache in user's home directory."""
    cache_dir = Path.home() / ".cache" / "structured-evals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "langchain_cache.db"
    set_llm_cache(SQLiteCache(database_path=str(cache_file)))
    logger.info(f"Using cache at {cache_file}")


@app.command()
def eval_from_schema(
    predictions_file: Annotated[
        Path, typer.Argument(help="Path to JSON file containing predictions and targets")
    ],
    schema_file: Annotated[Path, typer.Argument(help="Path to YAML schema file")],
    output_file: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output file for results")
    ] = None,
    pred_key: Annotated[
        str, typer.Option("--pred-key", help="Key for predictions in JSON")
    ] = "answer",
    target_key: Annotated[
        str, typer.Option("--target-key", help="Key for targets in JSON")
    ] = "gold",
    text_evaluator: Annotated[
        str, typer.Option("--text-evaluator", help="Text evaluator to use")
    ] = "llm",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Evaluate predictions using a schema file to infer the evaluator structure."""
    setup_cache()

    if output_file is None:
        output_file = Path("results.json")

    logger.info(f"Loading data from {predictions_file}")
    eval_batch = EvaluationBatch.from_json(
        path=str(predictions_file),
        record_format="json",
        pred_key=pred_key,
        target_key=target_key,
    )

    logger.info(f"Loading schema from {schema_file}")
    with open(schema_file, "r") as f:
        schema = yaml.safe_load(f)

    logger.info("Inferring evaluator from schema")
    item_evaluator = infer_structured_evaluator_from_schema(schema, text_evaluator=text_evaluator)
    assert isinstance(item_evaluator, DictEval)

    evaluator = BatchDictEval.from_dict_eval(item_evaluator, verbose=verbose)

    logger.info("Running evaluation")
    results = evaluator(pred=eval_batch.pred, target=eval_batch.target)
    report = EvaluationReport.from_batch_dict_eval_output(results, aggregation=AverageAggregation())

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

    logger.info("Evaluation completed")


@app.command()
def eval_from_predictions(
    predictions_file: Annotated[
        Path, typer.Argument(help="Path to JSON file containing predictions and targets")
    ],
    output_file: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output file for results")
    ] = None,
    pred_key: Annotated[
        str, typer.Option("--pred-key", help="Key for predictions in JSON")
    ] = "answer",
    target_key: Annotated[
        str, typer.Option("--target-key", help="Key for targets in JSON")
    ] = "gold",
    text_evaluator: Annotated[
        str, typer.Option("--text-evaluator", help="Text evaluator to use")
    ] = "llm",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Evaluate predictions by inferring the evaluator structure from the target data."""
    setup_cache()

    if output_file is None:
        output_file = Path("results.json")

    logger.info(f"Loading data from {predictions_file}")
    eval_batch = EvaluationBatch.from_json(
        path=str(predictions_file),
        record_format="json",
        pred_key=pred_key,
        target_key=target_key,
    )

    logger.info("Inferring evaluator from raw predictions")
    item_evaluator = infer_structured_evaluator_from_predictions(
        eval_batch.target[0], text_evaluator=text_evaluator
    )
    assert isinstance(item_evaluator, DictEval)

    evaluator = BatchDictEval.from_dict_eval(item_evaluator, verbose=verbose)

    logger.info("Running evaluation")
    results = evaluator(pred=eval_batch.pred, target=eval_batch.target)
    report = EvaluationReport.from_batch_dict_eval_output(results, aggregation=AverageAggregation())

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

    logger.info("Evaluation completed")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
