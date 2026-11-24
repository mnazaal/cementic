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
from urllib.parse import quote, quote_plus

if TYPE_CHECKING:
    from sqlalchemy.engine import URL

MASK = "***"

#: Key names that carry a secret, as a regex fragment: anything ending in
#: ``password``/``passwd`` (``sslpassword`` is libpq's), plus names that are
#: unambiguously credentials. A bare ``token`` or ``pwd`` is deliberately
#: absent: ``max_token=128`` and ``PWD=/home/u`` are ordinary log text.
#: ``passfile`` and ``sslkey`` name files, not secrets.
_KEY_NAMES = (
    r"\w*pass(?:word|wd)"
    r"|\w*(?:access_token|refresh_token|auth_token|id_token|api[-_]?key|secret|secret_key)"
)
_SECRET_KEY = re.compile(_KEY_NAMES, re.IGNORECASE)

#: A value: quoted (backslash escapes allowed), or unquoted up to whitespace.
#: An unquoted ``&`` ends the value only when a new ``key=`` follows, so a
#: libpq password such as ``ab&X`` is masked whole while ``?password=B&sslmode=x``
#: keeps ``sslmode``. Trailing ``) ] } , ; ' "`` punctuation is left outside, so
#: the closing quote of ``'host=h password=x'`` survives; a quote followed by
#: more value (``ab'x``) belongs to the value.
_VALUE = (
    r"""(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'"""
    r"""|(?:[^\s&]|&(?!\w+=))+?(?=[),;\]}'"]*(?:\s|$|&\w+=)))"""
)

#: ``key=value`` in a URL query or libpq conninfo (``password = x`` allowed),
#: and ``'key': 'value'`` / ``"key": "value"`` in dict and JSON reprs.
_KEY_VALUE = re.compile(
    rf"""(?ix)
    (?<!\w)
    (?P<q>["']?)
    (?P<key>{_KEY_NAMES})
    (?P=q)
    (?P<sep>[ \t]*[=:][ \t]*)
    (?P<value>{_VALUE})
    """
)

#: ``scheme://user:password@`` inside free text. The password runs to the last
#: ``@`` of the whitespace-free token, so a raw ``@`` or ``/`` inside it is
#: covered; a password containing whitespace is not recognised by shape.
#: ``http://host:8080/a@b`` therefore masks ``8080/a``: it is structurally
#: identical to a password containing ``/``, and masking is the safe side.
_AUTHORITY = re.compile(r"(?P<head>\b[\w+.-]+://[^:/?#\s@]*):(?P<pw>\S+)@")

#: Replacing a one- or two-character secret everywhere would destroy the
#: surrounding message, so such values are matched by shape only.
_MIN_LITERAL = 3


def _variants(literal: str) -> list[str]:
    return list({literal, quote_plus(literal), quote(literal, safe="")})


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
                for variant in _variants(literal):
                    text = text.replace(variant, MASK)
    text = _KEY_VALUE.sub(lambda m: f"{m['q']}{m['key']}{m['q']}{m['sep']}{MASK}", text)
    return _AUTHORITY.sub(lambda m: f"{m['head']}:{MASK}@", text)


def install_last_resort_redaction() -> None:
    """Make ``logging.lastResort`` (handlerless library loggers) mask secrets.

    Called by every process entry point: the CLI and the background runner.
    """
    if logging.lastResort is not None:
        logging.lastResort.setFormatter(RedactingFormatter())


class RedactingFormatter(logging.Formatter):
    """Formatter that masks credentials in the message and in tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record))
