from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Literal

import typer
import uvicorn

from multimodal_rag.config import get_settings
from multimodal_rag.engine import MultimodalRAG
from multimodal_rag.eval import load_eval_cases, parse_k_values, run_evaluation, save_evaluation_report

app = typer.Typer(help="Multimodal RAG CLI")


def _build_engine(backend: Literal["faiss", "qdrant"] | None = None) -> MultimodalRAG:
    settings = get_settings()
    if backend:
        settings = settings.model_copy(update={"vector_backend": backend})
    return MultimodalRAG(settings=settings)


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory path to ingest."),
    collection: str | None = typer.Option(None, help="Collection name."),
    backend: Literal["faiss", "qdrant"] | None = typer.Option(None, help="Vector backend override."),
) -> None:
    engine = _build_engine(backend)
    stats = engine.ingest_paths([path], collection=collection)
    typer.echo(
        f"Ingested files={stats['files']} chunks={stats['chunks']} "
        f"text={stats['text']} table={stats['table']} image={stats['image']}"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question for retrieval + generation."),
    image: Path | None = typer.Option(None, help="Optional query image path for multimodal retrieval."),
    collection: str | None = typer.Option(None, help="Collection name."),
    top_k: int | None = typer.Option(None, min=1, max=50, help="Top-k per modality."),
    backend: Literal["faiss", "qdrant"] | None = typer.Option(None, help="Vector backend override."),
) -> None:
    engine = _build_engine(backend)
    if image is not None and not image.exists():
        raise typer.BadParameter(f"Image path not found: {image}")
    result = engine.query(
        question=question,
        collection=collection,
        top_k=top_k,
        query_image_path=image,
    )
    typer.echo(result.answer)
    if result.citations:
        typer.echo("\nCitations:")
        for citation in result.citations:
            page = f", page={citation.page_number}" if citation.page_number is not None else ""
            typer.echo(
                f"- [{citation.modality.value}] {citation.source_path}{page}"
            )
    if result.hits:
        typer.echo("\nSources:")
        for hit in result.hits:
            typer.echo(
                f"- [{hit.chunk.modality.value}] {hit.chunk.source_path} "
                f"(score={hit.score:.4f})"
            )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    uvicorn.run("multimodal_rag.api.app:app", host=host, port=port, reload=reload)


@app.command("eval")
def evaluate(
    dataset: Path = typer.Argument(..., help="Path to evaluation dataset (.jsonl or .json)."),
    ingest_path: list[Path] | None = typer.Option(
        None,
        "--ingest-path",
        help="Optional corpus path to ingest before evaluation. Repeat for multiple paths.",
    ),
    collection: str | None = typer.Option(None, help="Collection name."),
    k_values: str = typer.Option("1,3,5", help="Comma-separated K values, for example: 1,3,5,10"),
    output: Path | None = typer.Option(None, help="Optional output path for JSON report."),
    backend: Literal["faiss", "qdrant"] | None = typer.Option(None, help="Vector backend override."),
) -> None:
    engine = _build_engine(backend)
    if ingest_path:
        missing = [str(path) for path in ingest_path if not path.exists()]
        if missing:
            raise typer.BadParameter(f"Ingest path(s) not found: {', '.join(missing)}")
        stats = engine.ingest_paths(list(ingest_path), collection=collection)
        typer.echo(
            "Pre-ingest completed: "
            f"files={stats['files']} chunks={stats['chunks']} "
            f"text={stats['text']} table={stats['table']} image={stats['image']}"
        )

    if not dataset.exists():
        raise typer.BadParameter(f"Dataset not found: {dataset}")

    try:
        parsed_k_values = parse_k_values(k_values)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    cases = load_eval_cases(dataset)
    report = run_evaluation(
        engine=engine,
        cases=cases,
        dataset_path=dataset,
        default_collection=collection,
        k_values=parsed_k_values,
    )

    report_path = output
    if report_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = engine.settings.storage_dir / "eval_reports" / f"report_{stamp}.json"
    save_evaluation_report(report, report_path)

    summary = report.summary
    typer.echo("\nEvaluation Summary")
    typer.echo(f"- Total cases: {summary.total_cases}")
    typer.echo(f"- Retrieval-evaluable cases: {summary.retrieval_evaluable_cases}")
    typer.echo(f"- Citation-evaluable cases: {summary.citation_evaluable_cases}")
    typer.echo(f"- Avg latency (ms): {summary.avg_latency_ms:.2f}")
    typer.echo(f"- P95 latency (ms): {summary.p95_latency_ms:.2f}")
    if summary.mean_mrr is not None:
        typer.echo(f"- Mean MRR: {summary.mean_mrr:.4f}")
    for key, value in summary.mean_recall_at.items():
        typer.echo(f"- Mean Recall@{key}: {value:.4f}")
    if summary.citation_hit_rate is not None:
        typer.echo(f"- Citation hit-rate: {summary.citation_hit_rate:.4f}")
    if summary.mean_citation_precision is not None:
        typer.echo(f"- Mean citation precision: {summary.mean_citation_precision:.4f}")
    typer.echo(f"- Report saved to: {report_path}")


if __name__ == "__main__":
    app()
