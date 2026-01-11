import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[3] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from parser.quotes_advanced import (  # type: ignore E402
    _extract_quotes,
    replace_quotes_with_placeholder,
)


def test_extract_quotes_order_and_placeholder():
    text = 'Он сказал: "Привет" и добавил: «До встречи».'
    quotes, spans = _extract_quotes(text)
    assert quotes == ["Привет", "До встречи"]
    assert len(spans) == 2

    replaced = replace_quotes_with_placeholder(text, placeholder="цитата")
    assert 'Он сказал: цитата и добавил: цитата.' == replaced


def test_unmatched_quotes_are_ignored():
    text = 'Открытая кавычка без закрытия: "незавершенная фраза.'
    quotes, spans = _extract_quotes(text)
    assert quotes == []
    assert spans == []
    replaced = replace_quotes_with_placeholder(text, placeholder="цитата")
    assert replaced == text
