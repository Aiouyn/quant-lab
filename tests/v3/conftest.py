import pytest


@pytest.fixture(autouse=True)
def _no_alt_data_downloads(monkeypatch):
    """Store-build tests fake Yahoo; keep the macro alt-data step (FRED / CFTC) offline too."""
    monkeypatch.setenv('MACRO_ALT_DATA', '0')
