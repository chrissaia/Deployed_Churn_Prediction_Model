from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from langfuse.opentelemetry import LangfuseSpanProcessor


def setup_tracing(app):
    resource = Resource.create({
        "service.name": "telco-churn-api"
    })

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Export app spans to Google Cloud Trace
    cloud_exporter = CloudTraceSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(cloud_exporter))

    # Export app spans to Langfuse too
    provider.add_span_processor(LangfuseSpanProcessor())

    FastAPIInstrumentor.instrument_app(app)

    return trace.get_tracer(__name__)