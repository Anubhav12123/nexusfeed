import json
import logging

from nexusfeed.observability.logging import JsonFormatter, configure_logging, request_id_ctx


def test_json_formatter_produces_valid_json_with_core_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="nexusfeed.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello world", args=(), exc_info=None,
    )
    output = formatter.format(record)
    payload = json.loads(output)
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "nexusfeed.test"


def test_json_formatter_includes_request_id_from_context():
    token = request_id_ctx.set("req-123")
    try:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="nexusfeed.test", level=logging.INFO, pathname=__file__, lineno=1,
            msg="with request id", args=(), exc_info=None,
        )
        payload = json.loads(formatter.format(record))
        assert payload["request_id"] == "req-123"
    finally:
        request_id_ctx.reset(token)


def test_json_formatter_merges_extra_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="nexusfeed.test", level=logging.WARNING, pathname=__file__, lineno=1,
        msg="event happened", args=(), exc_info=None,
    )
    record.custom_field = "custom_value"
    payload = json.loads(formatter.format(record))
    assert payload["custom_field"] == "custom_value"


def test_configure_logging_sets_root_handler():
    configure_logging(level="DEBUG", json_format=True)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)


def test_configure_logging_console_format():
    configure_logging(level="INFO", json_format=False)
    root = logging.getLogger()
    assert not isinstance(root.handlers[0].formatter, JsonFormatter)
