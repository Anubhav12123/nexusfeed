"""OpenTelemetry distributed tracing setup.

Exports spans over OTLP so request traces can be correlated across the API,
Kafka consumers, and the training pipeline in a tool like Jaeger or Honeycomb.
Falls back to a no-op tracer if the collector endpoint is unreachable so local
dev never breaks on a missing OTel collector.
"""
from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from nexusfeed.config import Settings, get_settings

logger = logging.getLogger(__name__)

_configured = False


def configure_tracing(settings: Settings | None = None) -> trace.Tracer:
    global _configured
    settings = settings or get_settings()
    if not _configured:
        try:
            resource = Resource.create({"service.name": settings.app_name, "service.env": settings.env})
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            _configured = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("otel_configure_failed_using_noop", extra={"error": str(exc)})
    return trace.get_tracer(settings.app_name)
