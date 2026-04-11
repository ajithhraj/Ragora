from __future__ import annotations

from contextlib import contextmanager
import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from multimodal_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


class TelemetryManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(settings.observability_enabled)
        self._configured = False
        self._tracer: Any | None = None
        self._request_counter: Any | None = None
        self._request_latency_hist: Any | None = None
        self._query_latency_hist: Any | None = None

    def setup(self) -> None:
        if self._configured:
            return
        self._configured = True
        if not self.enabled:
            return

        try:
            from opentelemetry import metrics, trace
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import (
                ConsoleMetricExporter,
                PeriodicExportingMetricReader,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
            from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
        except Exception as exc:  # pragma: no cover - dependency/runtime branch
            logger.warning("OpenTelemetry dependencies unavailable; observability disabled: %s", exc)
            self.enabled = False
            return

        resource = Resource.create({"service.name": self.settings.observability_service_name})

        span_processors: list[Any] = []
        metric_readers: list[Any] = []

        if self.settings.observability_otlp_endpoint:
            endpoint = self.settings.observability_otlp_endpoint
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                span_processors.append(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
                metric_readers.append(
                    PeriodicExportingMetricReader(
                        OTLPMetricExporter(endpoint=endpoint),
                        export_interval_millis=5000,
                    )
                )
            except Exception as exc:  # pragma: no cover - dependency/runtime branch
                logger.warning("OTLP exporter unavailable; telemetry will continue without OTLP: %s", exc)

        if self.settings.observability_console_exporter:
            span_processors.append(BatchSpanProcessor(ConsoleSpanExporter()))
            metric_readers.append(
                PeriodicExportingMetricReader(
                    ConsoleMetricExporter(),
                    export_interval_millis=5000,
                )
            )

        tracer_provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(TraceIdRatioBased(self.settings.observability_trace_sample_ratio)),
        )
        for processor in span_processors:
            tracer_provider.add_span_processor(processor)

        meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)

        try:
            trace.set_tracer_provider(tracer_provider)
        except Exception:
            # Another tracer provider may already be configured in process.
            pass
        try:
            metrics.set_meter_provider(meter_provider)
        except Exception:
            # Another meter provider may already be configured in process.
            pass

        self._tracer = trace.get_tracer("multimodal_rag")
        meter = metrics.get_meter("multimodal_rag")
        self._request_counter = meter.create_counter(
            "mmrag.http.requests",
            unit="1",
            description="Total API requests",
        )
        self._request_latency_hist = meter.create_histogram(
            "mmrag.http.request.duration_ms",
            unit="ms",
            description="HTTP request duration",
        )
        self._query_latency_hist = meter.create_histogram(
            "mmrag.query.duration_ms",
            unit="ms",
            description="RAG query duration",
        )

    def generate_request_id(self, incoming_value: str | None = None) -> str:
        value = (incoming_value or "").strip()
        if value:
            return value[:128]
        return uuid4().hex

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None):
        if not self.enabled or not self._tracer:
            yield None
            return

        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    if value is None:
                        continue
                    span.set_attribute(key, value)
            yield span

    @contextmanager
    def timed_span(self, name: str, attributes: dict[str, Any] | None = None):
        start = perf_counter()
        with self.span(name, attributes=attributes) as span:
            yield span
        duration_ms = (perf_counter() - start) * 1000.0
        if span is not None:
            span.set_attribute("mmrag.duration_ms", duration_ms)

    def record_http(self, method: str, route: str, status_code: int, duration_ms: float) -> None:
        if not self.enabled:
            return
        attrs = {
            "http.method": method,
            "http.route": route,
            "http.status_code": status_code,
        }
        if self._request_counter is not None:
            self._request_counter.add(1, attrs)
        if self._request_latency_hist is not None:
            self._request_latency_hist.record(duration_ms, attrs)

    def record_query(
        self,
        route: str,
        retrieval_mode: str | None,
        corrected: bool,
        grounded: bool,
        duration_ms: float,
    ) -> None:
        if not self.enabled or self._query_latency_hist is None:
            return
        attrs = {
            "http.route": route,
            "mmrag.retrieval_mode": retrieval_mode or "unknown",
            "mmrag.corrected": corrected,
            "mmrag.grounded": grounded,
        }
        self._query_latency_hist.record(duration_ms, attrs)


_telemetry_instance: TelemetryManager | None = None


def get_telemetry(settings: Settings | None = None) -> TelemetryManager:
    global _telemetry_instance
    if _telemetry_instance is None:
        config = settings or get_settings()
        manager = TelemetryManager(config)
        manager.setup()
        _telemetry_instance = manager
    return _telemetry_instance
