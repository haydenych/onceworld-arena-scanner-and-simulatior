"""OCR helpers and Tesseract setup."""

import os
import re

import cv2
import pytesseract

import user_config
from onceworld.config.runtime import TESS_CONFIG_COIN, TESS_CONFIG_LEVEL
from onceworld.paths import TESSERACT_EXE_FALLBACK

_TESS_CONFIGURED = False


def configure_tesseract():
    global _TESS_CONFIGURED
    if _TESS_CONFIGURED:
        return

    tess_cmd = user_config.TESSERACT_CMD
    if not tess_cmd and os.path.isfile(str(TESSERACT_EXE_FALLBACK)):
        tess_cmd = str(TESSERACT_EXE_FALLBACK)

    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    _TESS_CONFIGURED = True


def preprocess_for_ocr(img_bgr):
    img = cv2.resize(img_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def read_level_text(img_bgr):
    proc = preprocess_for_ocr(img_bgr)
    text = pytesseract.image_to_string(proc, config=TESS_CONFIG_LEVEL).strip()
    match = re.search(r"Lv\.?\s*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), text
    return None, text


def read_coin_text(img_bgr):
    proc = preprocess_for_ocr(img_bgr)
    text = pytesseract.image_to_string(proc, config=TESS_CONFIG_COIN).strip()
    digits = re.sub(r"[^0-9]", "", text)
    if digits:
        return int(digits), text
    return None, text


configure_tesseract()
