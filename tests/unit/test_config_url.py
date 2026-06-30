"""Tests for DatabaseConfig URL security."""

from sqlalchemy.engine import URL

from cementic.config import Config, DatabaseConfig


class TestDatabaseUrlSecurity:
    """Verify password handling in database URL."""

    def test_url_does_not_leak_password_in_repr(self) -> None:
        cfg = DatabaseConfig(
            host="localhost",
            port=5432,
            name="mydb",
            user="admin",
            password="s3cret!@#",
        )
        url = cfg.url
        # repr() and str() redact the password via SQLAlchemy URL object
        assert isinstance(url, URL)
        assert "s3cret" not in repr(url)
        assert "s3cret" not in str(url)
        assert "***" in repr(url)

    def test_url_still_connectable(self) -> None:
        cfg = DatabaseConfig(
            host="localhost",
            port=5432,
            name="mydb",
            user="admin",
            password="s3cret!@#",
        )
        url = cfg.url
        # The URL object is valid for SQLAlchemy connections
        full = url.render_as_string(hide_password=False)
        assert "admin" in full
        assert "s3cret%21%40%23" in full  # URL-encoded
        assert "localhost:5432" in full
        assert "mydb" in full

    def test_password_with_special_characters_url_encoded(self) -> None:
        cfg = DatabaseConfig(
            host="localhost",
            port=5432,
            name="mydb",
            user="admin",
            password="p@ss:word/with#special%chars",
        )
        url = cfg.url
        assert "p@ss:word" not in repr(url)
        # Password is URL-encoded for special chars
        full = url.render_as_string(hide_password=False)
        assert "%40" in full  # @ encoded
        assert "%3A" in full  # : encoded
        assert "%2F" in full  # / encoded


class TestDatabaseUrlOverride:
    """A full URL override wins over the discrete host/port/... fields."""

    def test_url_override_takes_precedence(self) -> None:
        cfg = DatabaseConfig(
            host="localhost",
            port=5432,
            name="mydb",
            user="admin",
            password="secret",
            url_override="postgresql://other:pw@otherhost:6543/otherdb",
        )
        url = cfg.url
        assert url.host == "otherhost"
        assert url.port == 6543
        assert url.database == "otherdb"
        assert url.username == "other"

    def test_cementic_db_url_env_var_is_honored(self, monkeypatch) -> None:
        monkeypatch.delenv("CEMENTIC_CONFIG", raising=False)
        monkeypatch.setenv("CEMENTIC_DB_URL", "postgresql://u:pw@envhost:7000/envdb")
        url = Config().database.url
        assert url.host == "envhost"
        assert url.port == 7000
        assert url.database == "envdb"

    def test_no_override_builds_from_parts(self) -> None:
        cfg = DatabaseConfig(host="localhost", port=5432, name="mydb", user="admin")
        assert cfg.url.host == "localhost"
        assert cfg.url.database == "mydb"
