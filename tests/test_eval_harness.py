from pathlib import Path

import pytest

from multimodal_rag.eval import load_eval_cases, parse_k_values, run_evaluation
from multimodal_rag.eval.models import EvalCase
from multimodal_rag.models import Citation, Chunk, Modality, QueryAnswer, RetrievalHit


class EvalEngineStub:
    def query(self, question, collection=None, top_k=None, query_image_path=None):
        if question == "q1":
            hit = RetrievalHit(
                chunk=Chunk(
                    chunk_id="hit-1",
                    source_path="docs/report.pdf",
                    modality=Modality.TEXT,
                    content="Revenue was 25M",
                ),
                score=0.9,
                backend="stub",
            )
            citation = Citation(
                chunk_id="hit-1",
                source_path="docs/report.pdf",
                modality=Modality.TEXT,
                page_number=2,
                excerpt="Revenue was 25M",
            )
            return QueryAnswer(answer="stub-1", hits=[hit], citations=[citation])

        hit_1 = RetrievalHit(
            chunk=Chunk(
                chunk_id="miss-1",
                source_path="docs/other.pdf",
                modality=Modality.TEXT,
                content="Not relevant",
            ),
            score=0.8,
            backend="stub",
        )
        hit_2 = RetrievalHit(
            chunk=Chunk(
                chunk_id="hit-2",
                source_path="data/table.csv",
                modality=Modality.TABLE,
                content="Table row",
            ),
            score=0.7,
            backend="stub",
        )
        return QueryAnswer(answer="stub-2", hits=[hit_1, hit_2], citations=[])


def test_parse_k_values():
    assert parse_k_values("1, 3, 5,3") == [1, 3, 5]
    with pytest.raises(ValueError):
        parse_k_values("")
    with pytest.raises(ValueError):
        parse_k_values("0,2")


def test_load_eval_cases_jsonl_auto_case_id(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '\n'.join(
            [
                '{"question":"q1","expected_chunk_ids":["x"]}',
                '{"case_id":"custom_case","question":"q2","expected_source_paths":["report.pdf"]}',
            ]
        ),
        encoding="utf-8",
    )
    cases = load_eval_cases(dataset)
    assert len(cases) == 2
    assert cases[0].case_id == "case_001"
    assert cases[1].case_id == "custom_case"


def test_run_evaluation_metrics():
    engine = EvalEngineStub()
    cases = [
        EvalCase(case_id="c1", question="q1", expected_chunk_ids=["hit-1"]),
        EvalCase(case_id="c2", question="q2", expected_source_paths=["table.csv"]),
    ]
    report = run_evaluation(
        engine=engine,
        cases=cases,
        dataset_path=Path("eval/datasets/starter_eval.jsonl"),
        default_collection=None,
        k_values=[1, 2],
    )

    summary = report.summary
    assert summary.total_cases == 2
    assert summary.retrieval_evaluable_cases == 2
    assert summary.citation_evaluable_cases == 2
    assert summary.mean_mrr is not None
    assert summary.mean_recall_at["1"] == pytest.approx(0.5)
    assert summary.mean_recall_at["2"] == pytest.approx(1.0)
    assert summary.mean_mrr == pytest.approx(0.75)
    assert summary.citation_hit_rate == pytest.approx(0.5)
    assert summary.mean_citation_precision == pytest.approx(0.5)
    assert summary.avg_latency_ms >= 0.0
