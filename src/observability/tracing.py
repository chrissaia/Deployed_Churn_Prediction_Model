from google.auth.exceptions import DefaultCredentialsError

def setup_tracing(app):
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        if trace.get_tracer_provider().__class__.__name__ == "ProxyTracerProvider":
            trace.set_tracer_provider(TracerProvider())

        exporter = CloudTraceSpanExporter()
        span_processor = BatchSpanProcessor(exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        print("Cloud Trace enabled")

    except DefaultCredentialsError:
        print("Cloud Trace disabled (no credentials found)")
    except Exception as e:
        print(f"Cloud Trace disabled: {e}")