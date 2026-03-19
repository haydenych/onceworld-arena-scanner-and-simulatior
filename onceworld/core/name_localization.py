"""Helpers for EN/JP monster-name display localization."""

import csv
from pathlib import Path


_DEFAULT_LANG = "EN"
_SUPPORTED_LANGS = {"EN", "JP"}
_TEXT_BY_LANG = {
    "EN": {
        "unknown": "unknown",
        "low_confidence": "low confidence",
        "na": "n/a",
    },
    "JP": {
        "unknown": "不明",
        "low_confidence": "低信頼度",
        "na": "n/a",
    },
}


def normalize_lang(language):
    """Normalize a language code to EN/JP with EN fallback."""
    lang = str(language or "").upper()
    if lang in _SUPPORTED_LANGS:
        return lang
    return _DEFAULT_LANG


def load_name_map(csv_path):
    """Load PET_NAME -> PET_NAME_JP map from monsters CSV."""
    mapping = {}
    path = Path(csv_path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                en_name = str(row.get("PET_NAME", "")).strip()
                jp_name = str(row.get("PET_NAME_JP", "")).strip()
                if en_name:
                    mapping[en_name] = jp_name or en_name
    except Exception:
        return {}
    return mapping


def localized_text(key, language):
    """Get a localized UI text token."""
    lang = normalize_lang(language)
    table = _TEXT_BY_LANG.get(lang, _TEXT_BY_LANG[_DEFAULT_LANG])
    return table.get(key, key)


def display_unit_name(unit_name, language, name_jp_by_en):
    """Return the display name for a unit in the selected language."""
    name = str(unit_name or "")
    if not name:
        return name

    lang = normalize_lang(language)
    if name == _TEXT_BY_LANG["EN"]["unknown"]:
        return localized_text("unknown", lang)

    if lang == "JP":
        return name_jp_by_en.get(name, name)
    return name
