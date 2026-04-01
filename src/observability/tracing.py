from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


def setup_tracing(app):
    resource = Resource.create({
        "service.name": "telco-churn-api"
    })

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    exporter = CloudTraceSpanExporter()
    span_processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(span_processor)

    # Auto-instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    return trace.get_tracer(__name__)