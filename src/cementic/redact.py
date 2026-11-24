"""Mask credentials in database URLs and in text that may echo one.

SQLAlchemy's ``hide_password=True`` masks only the authority password. A URL
such as ``postgresql+psycopg://u:A@h/d?password=B&sslpassword=C`` keeps ``B``
and ``C`` -- libpq reads both as credentials -- and a driver error can echo a
whole DSN. Every place that prints, logs or serialises a URL or a database
error goes through here.
"""

import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

if TYPE_CHECKING:
    from sqlalchemy.engine import URL

MASK = "***"

#: Query keys that carry a secret: libpq's ``password`` and ``sslpassword``,
#: plus the usual spellings other drivers use. ``passfile`` and ``sslkey``
#: name files, not secrets, so they stay readable.
_SECRET_KEY = re.compile(r"(?i)\w*pass(?:word|wd)|pwd|\w*secret|\w*token")

#: ``key=value`` in a URL query or a libpq conninfo string; the value may be
#: quoted, in which case it runs to the closing quote.
_KEY_VALUE = re.compile(
    r"""(?ix)
    (?P<key>\b\w*(?:pass(?:word|wd)|pwd|secret|token))
    =
    (?P<value>"[^"]*"|'[^']*'|[^\s&;,)\]"']+)
    """
)

#: ``scheme://user:password@`` inside free text.
_AUTHORITY = re.compile(r"(?P<head>\b[\w+.-]+://[^:/?#\s@]*):(?P<pw>[^@/\s]+)@")

#: Replacing a one- or two-character secret everywhere would destroy the
#: surrounding message, so such values are matched by shape only.
_MIN_LITERAL = 3


def _secret_values(url: "URL") -> list[str]:
    values: list[str] = []
    if url.password:
        values.append(url.password)
    for key, value in url.query.items():
        if _SECRET_KEY.fullmatch(key):
            values.extend([value] if isinstance(value, str) else list(value))
    return values


def redact_url(url: "str | URL") -> str:
    """Render ``url`` with the authority password and secret query values masked."""
    # Imported here: cli.py imports this module, and the CLI must start
    # without pulling in SQLAlchemy (tests/unit/test_cli_performance.py).
    from sqlalchemy.engine import make_url

    parsed = make_url(url) if isinstance(url, str) else url
    base = parsed.set(query={}).render_as_string(hide_password=True)
    pairs: list[str] = []
    for key, value in parsed.query.items():
        for item in [value] if isinstance(value, str) else value:
            shown = MASK if _SECRET_KEY.fullmatch(key) else quote_plus(item)
            pairs.append(f"{quote_plus(key)}={shown}")
    return base + ("?" + "&".join(pairs) if pairs else "")


def redact_text(text: str, url: "str | URL | None" = None) -> str:
    """Mask credentials in ``text``, such as a driver error echoing a DSN.

    With ``url``, its secret values are masked wherever they appear, even bare.
    Without it, only DSN shapes are masked: ``user:pw@`` and ``password=...``.
    """
    if url is not None:
        from sqlalchemy.engine import make_url

        try:
            parsed = make_url(url) if isinstance(url, str) else url
            literals = _secret_values(parsed)
        except Exception:
            literals = []
        for literal in sorted(literals, key=len, reverse=True):
            if len(literal) >= _MIN_LITERAL:
                text = text.replace(literal, MASK).replace(quote_plus(literal), MASK)
    text = _KEY_VALUE.sub(lambda m: f"{m['key']}={MASK}", text)
    return _AUTHORITY.sub(lambda m: f"{m['head']}:{MASK}@", text)


class RedactingFormatter(logging.Formatter):
    """Formatter that masks credentials in the message and in tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record))
