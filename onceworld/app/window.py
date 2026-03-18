"""Qt main window for OnceWorld Arena detection UI."""

import pytesseract
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QImage,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from onceworld.config.runtime import UNIT_WARN_THRESHOLD
from onceworld.core.detector import ScreenDetector


def _fmt_pct(value):
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "n/a"


def _fmt_float(value, digits=3):
    try:
        return f"{float(value):.{int(digits)}f}"
    except Exception:
        return "n/a"


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, float(t)))
    r = int(c1.red() + (c2.red() - c1.red()) * t)
    g = int(c1.green() + (c2.green() - c1.green()) * t)
    b = int(c1.blue() + (c2.blue() - c1.blue()) * t)
    return QColor(r, g, b)


class MainWindow(QMainWindow):
    def __init__(self, save_unlabeled=False):
        super().__init__()
        self._base_window_width = 400
        self._base_window_height = 400
        self._font_targets = []
        self._enemy_text_boxes = []

        self.setWindowTitle("OnceWorld Arena")
        self.setMinimumSize(self._base_window_width, self._base_window_height)
        self.resize(self._base_window_width, self._base_window_height)
        self._set_app_icon()
        self.detector = None
        try:
            self.detector = ScreenDetector(save_unlabeled=save_unlabeled)
        except Exception as exc:
            QMessageBox.critical(self, "Start", str(exc))
            self.close()
            return

        self._check_ocr()
        self._init_palette()
        self._build_ui()
        self._apply_scaled_fonts()
        self._set_status("")

    def _set_app_icon(self):
        size = 64
        img = QImage(size, size, QImage.Format_ARGB32)
        img.fill(QColor("#0D1117"))

        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing, True)

        pen = QPen(QColor("#21262D"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QColor("#161B22"))
        painter.drawRoundedRect(4, 4, size - 8, size - 8, 12, 12)

        pen2 = QPen(QColor("#58A6FF"))
        pen2.setWidth(5)
        pen2.setCapStyle(Qt.RoundCap)
        painter.setPen(pen2)
        painter.drawLine(18, 38, 30, 26)
        painter.drawLine(30, 26, 46, 42)
        painter.end()

        self.setWindowIcon(QIcon(QPixmap.fromImage(img)))

    def _init_palette(self):
        self.c_bg = QColor("#0D1117")
        self.c_panel = QColor("#161B22")
        self.c_panel_soft = QColor("#0D1117")
        self.c_text = QColor("#C9D1D9")
        self.c_muted = QColor("#8B949E")
        self.c_accent = QColor("#58A6FF")
        self.c_good = QColor("#3FB950")
        self.c_warn = QColor("#D29922")
        self.c_bad = QColor("#F85149")
        self.c_team_a = QColor("#EF4444")
        self.c_team_b = QColor("#3B82F6")
        self.c_team_c = QColor("#22C55E")
        self.c_prob_lo = QColor("#22C55E")
        self.c_prob_hi = QColor("#EF4444")
        self.c_prob_gold = QColor("#FACC15")

        palette = self.palette()
        palette.setColor(QPalette.Window, self.c_bg)
        palette.setColor(QPalette.WindowText, self.c_text)
        palette.setColor(QPalette.Base, self.c_panel_soft)
        palette.setColor(QPalette.AlternateBase, self.c_panel)
        palette.setColor(QPalette.Text, self.c_text)
        palette.setColor(QPalette.Button, self.c_panel)
        palette.setColor(QPalette.ButtonText, self.c_text)
        self.setPalette(palette)

    def _card_frame(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Plain)
        frame.setStyleSheet(
            "QFrame { background-color: #161B22; border: 1px solid #21262D; border-radius: 6px; }"
        )
        return frame

    def _register_font_target(self, widget, base_pt, min_pt=7, max_pt=None):
        self._font_targets.append(
            {
                "widget": widget,
                "base": float(base_pt),
                "min": int(min_pt) if min_pt is not None else None,
                "max": int(max_pt) if max_pt is not None else None,
            }
        )

    def _label(self, text, muted=False, bold=False, large=False):
        label = QLabel(text)
        base_size = 14 if large else 8
        font = QFont("Segoe UI", base_size)
        if bold:
            font.setWeight(QFont.DemiBold)
        label.setFont(font)
        if large:
            self._register_font_target(label, base_size, min_pt=10, max_pt=24)
        else:
            self._register_font_target(label, base_size, min_pt=7, max_pt=14)
        if muted:
            label.setStyleSheet("color: #8B949E;")
        else:
            label.setStyleSheet("color: #C9D1D9;")
        return label

    def _font_scale_factor(self):
        w_scale = self.width() / float(self._base_window_width)
        h_scale = self.height() / float(self._base_window_height)
        scale = min(w_scale, h_scale)
        return max(0.85, min(1.8, scale))

    def _apply_scaled_fonts(self):
        if not self._font_targets:
            return

        scale = self._font_scale_factor()
        for spec in self._font_targets:
            widget = spec["widget"]
            if widget is None:
                continue
            font = widget.font()
            point_size = int(round(spec["base"] * scale))
            if spec["min"] is not None:
                point_size = max(spec["min"], point_size)
            if spec["max"] is not None:
                point_size = min(spec["max"], point_size)
            point_size = max(1, point_size)
            if font.pointSize() != point_size:
                font.setPointSize(point_size)
                widget.setFont(font)

        for text in self._enemy_text_boxes:
            fm = QFontMetrics(text.font())
            text.setMinimumHeight(int(fm.lineSpacing() * 4 + 8))

        if hasattr(self, "scan_btn"):
            btn_fm = QFontMetrics(self.scan_btn.font())
            self.scan_btn.setMinimumHeight(max(28, int(btn_fm.height() + 12)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled_fonts()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout()
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(3)
        central.setLayout(root)
        self.status_label = None

        top_cards = QHBoxLayout()
        top_cards.setSpacing(4)
        card_win = self._card_frame()
        card_win_layout = QVBoxLayout(card_win)
        card_win_layout.setContentsMargins(6, 4, 6, 4)
        card_win_layout.setSpacing(2)
        card_win_layout.addWidget(self._label("Win", muted=True))

        win_row = QHBoxLayout()
        win_row.setSpacing(4)
        self.win_team_label = self._label("-", bold=True, large=True)
        win_row.addWidget(self.win_team_label, 0, Qt.AlignLeft)

        self.win_prob_label = QLabel("-")
        prob_font = QFont("Segoe UI", 10)
        prob_font.setWeight(QFont.DemiBold)
        self.win_prob_label.setFont(prob_font)
        self._register_font_target(self.win_prob_label, 10, min_pt=8, max_pt=18)
        self.win_prob_label.setStyleSheet("color: #FACC15;")
        win_row.addWidget(self.win_prob_label, 0, Qt.AlignLeft)

        self.win_coin_label = self._label("-", muted=True, bold=True)
        win_row.addWidget(self.win_coin_label, 0, Qt.AlignLeft)
        win_row.addStretch()
        card_win_layout.addLayout(win_row)

        self.scan_btn = QPushButton("Scan")
        scan_font = QFont("Segoe UI", 8)
        scan_font.setWeight(QFont.DemiBold)
        self.scan_btn.setFont(scan_font)
        self._register_font_target(self.scan_btn, 8, min_pt=7, max_pt=14)
        self.scan_btn.setCursor(Qt.PointingHandCursor)
        self._scan_btn_style_normal = (
            "QPushButton { background-color: #238636; color: #FFFFFF; border-radius: 4px; padding: 4px 10px; }"
            "QPushButton:hover { background-color: #2ea043; }"
            "QPushButton:disabled { background-color: #30363D; color: #8B949E; }"
        )
        self._scan_btn_style_scanning = (
            "QPushButton { background-color: #30363D; color: #8B949E; border-radius: 4px; padding: 4px 10px; }"
        )
        self.scan_btn.setStyleSheet(self._scan_btn_style_normal)
        self.scan_btn.clicked.connect(self.on_scan)
        self.scan_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.scan_btn.setMinimumHeight(28)
        card_win_layout.addWidget(self.scan_btn)

        top_cards.addWidget(card_win)
        root.addLayout(top_cards)

        sides_frame = self._card_frame()
        sides_layout = QVBoxLayout(sides_frame)
        sides_layout.setContentsMargins(6, 2, 6, 2)
        sides_layout.setSpacing(2)
        self.side_units = {}

        for key, label in [("team_a", "A"), ("team_b", "B"), ("team_c", "C")]:
            row = QVBoxLayout()
            row.setSpacing(2)
            head = QHBoxLayout()
            side_lbl = self._label(label, muted=True, bold=True)
            head.addWidget(side_lbl, 0, Qt.AlignLeft)
            prob_lbl = self._label("-", muted=True)
            head.addWidget(prob_lbl, 0, Qt.AlignLeft)
            head.addStretch()
            coin_lbl = self._label("Coin -", muted=True)
            head.addWidget(coin_lbl, 0, Qt.AlignRight)
            row.addLayout(head)

            text = QTextEdit()
            text.setReadOnly(True)
            text.setFrameShape(QFrame.NoFrame)
            text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            text.setStyleSheet(
                "QTextEdit { background-color: #0D1117; color: #C9D1D9; border-radius: 4px; }"
            )
            mono = QFont("Cascadia Mono", 8)
            text.setFont(mono)
            self._register_font_target(text, 8, min_pt=7, max_pt=14)
            self._enemy_text_boxes.append(text)
            fm = QFontMetrics(mono)
            text.setMinimumHeight(int(fm.lineSpacing() * 4 + 8))
            text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            row.addWidget(text, 1)
            sides_layout.addLayout(row, 1)
            self.side_units[key] = {"coin": coin_lbl, "prob": prob_lbl, "list": text}

        root.addWidget(sides_frame, 1)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setVisible(False)

    def _check_ocr(self):
        try:
            _ = pytesseract.get_tesseract_version()
        except Exception:
            QMessageBox.warning(
                self,
                "OCR",
                "Tesseract OCR not found.\nInstall Tesseract and set user_config.TESSERACT_CMD.",
            )

    def _set_status(self, text):
        _ = text
        return

    def _append_log(self, text, level=None):
        color = "#C9D1D9"
        if level == "bad":
            color = "#F85149"
        elif level == "warn":
            color = "#D29922"
        self.log_text.setTextColor(QColor(color))
        self.log_text.append(text)
        self.log_text.moveCursor(QTextCursor.End)
        if len(self.log_text.toPlainText()) > 4000:
            txt = self.log_text.toPlainText()
            self.log_text.clear()
            self.log_text.setText(txt[-3000:])

    def on_scan(self):
        if not self.detector:
            return

        self.scan_btn.setStyleSheet(self._scan_btn_style_scanning)
        self.scan_btn.setEnabled(False)
        self.scan_btn.repaint()
        QApplication.processEvents()
        self._set_status("")
        self.log_text.clear()
        for key in self.side_units:
            self.side_units[key]["coin"].setText("Coin -")
            self.side_units[key]["list"].clear()

        try:
            results = self.detector.detect()
            self._render_results(results)
            self._set_status("")
        except Exception as exc:
            QMessageBox.critical(self, "Scan", str(exc))
            self._set_status("")
        finally:
            self.scan_btn.setStyleSheet(self._scan_btn_style_normal)
            self.scan_btn.setEnabled(True)
            self.scan_btn.repaint()

    def _render_results(self, results):
        battle = results.get("_battle", {})
        probs = {}

        if battle.get("available"):
            best_team = battle.get("best_team")
            best_prob = battle.get("best_prob", 0.0)
            probs = battle.get("team_probs", {}) or {}
            label = {"team_a": "A", "team_b": "B", "team_c": "C"}.get(best_team, "?")
            self.win_team_label.setText(label)
            if best_team == "team_a":
                self.win_team_label.setStyleSheet(f"color: {self.c_team_a.name()};")
            elif best_team == "team_b":
                self.win_team_label.setStyleSheet(f"color: {self.c_team_b.name()};")
            elif best_team == "team_c":
                self.win_team_label.setStyleSheet(f"color: {self.c_team_c.name()};")
            else:
                self.win_team_label.setStyleSheet("color: #C9D1D9;")

            self.win_prob_label.setText(f"{_fmt_pct(best_prob)}")
            if best_prob >= 0.999:
                color = self.c_prob_gold
            else:
                if best_prob <= 0.5:
                    t = 0.0
                elif best_prob >= 0.99:
                    t = 1.0
                else:
                    t = (best_prob - 0.5) / (0.99 - 0.5)
                color = _lerp_color(self.c_prob_hi, self.c_prob_lo, t)
            self.win_prob_label.setStyleSheet(f"color: {color.name()};")
            coins = results.get(best_team, {}).get("coins", None)
            self.win_coin_label.setText(f"Coin {coins if coins is not None else '-'}")
        else:
            self.win_team_label.setText("?")
            self.win_team_label.setStyleSheet("color: #8B949E;")
            self.win_prob_label.setText("n/a")
            self.win_coin_label.setText("Coin -")

        for team in ["team_a", "team_b", "team_c"]:
            team_result = results.get(team, {})
            side = self.side_units.get(team)
            if not side:
                continue

            side_prob = probs.get(team, None)
            if side.get("prob") is not None:
                side["prob"].setText(_fmt_pct(side_prob) if side_prob is not None else "-")

            box = side["list"]
            if not team_result.get("found"):
                side["coin"].setText("Coin -")
                box.clear()
                box.append("none")
                continue

            coins = team_result.get("coins")
            side["coin"].setText(f"Coin {coins if coins is not None else '-'}")
            box.clear()
            units = team_result.get("units", [])
            if not units:
                box.append("none")
                continue

            for idx, unit in enumerate(units, 1):
                name = unit.get("unit_name")
                level = unit.get("level")
                score = unit.get("score")
                score_txt = _fmt_float(score, 2) if score is not None else "n/a"
                label = {"team_a": "A", "team_b": "B", "team_c": "C"}.get(team, team)
                level_text = "???" if level is None else str(level)
                row_prefix = f"{label}{idx} Lv{level_text}-{name} "
                if name == "unknown":
                    box.setTextColor(self.c_bad)
                    box.insertPlainText(f"{row_prefix}{score_txt} (low confidence)\n")
                elif level is None:
                    box.setTextColor(self.c_bad)
                    box.insertPlainText(f"{row_prefix}{score_txt}\n")
                elif score is not None and score < UNIT_WARN_THRESHOLD:
                    box.setTextColor(self.c_text)
                    box.insertPlainText(row_prefix)
                    box.setTextColor(self.c_warn)
                    box.insertPlainText(f"{score_txt} (low confidence)\n")
                else:
                    box.setTextColor(self.c_text)
                    box.insertPlainText(f"{row_prefix}{score_txt}\n")
                box.setTextColor(self.c_text)
