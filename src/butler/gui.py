import json
import os
from typing import Callable

from PySide6.QtCore import QObject, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView


class ButlerAPI(QObject):
    """
    The Neural Bridge between Python and JavaScript.
    Methods marked with @Slot() are callable from index.html.
    """

    drag_requested = Signal()
    hit_regions_changed = Signal(str)

    def __init__(self, on_mic, on_wake, on_text, on_live, on_minimize, on_kill):
        super().__init__()
        self.on_mic = on_mic
        self.on_wake = on_wake
        self.on_text = on_text
        self.on_live = on_live
        self.on_minimize = on_minimize
        self.on_kill = on_kill
        self._debug_enabled = os.getenv("BUTLER_GUI_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    def _debug(self, name: str, value=None) -> None:
        if self._debug_enabled:
            suffix = "" if value is None else f"={value!r}"
            print(f"[BUTLER GUI] {name}{suffix}")

    @Slot()
    def on_mic_click(self):
        self._debug("on_mic_click")
        self.on_mic()

    @Slot(bool)
    def on_wake_toggle(self, enabled):
        self._debug("on_wake_toggle", enabled)
        self.on_wake(enabled)

    @Slot(str)
    def on_text_submit(self, text):
        self._debug("on_text_submit", text)
        self.on_text(text)

    @Slot(bool)
    def on_live_toggle(self, enabled):
        self._debug("on_live_toggle", enabled)
        self.on_live(enabled)

    @Slot(bool)
    def on_text_toggle(self, enabled):
        self._debug("on_text_toggle", enabled)

    @Slot()
    def start_drag(self):
        self._debug("start_drag")
        self.drag_requested.emit()

    @Slot()
    def minimize(self):
        self._debug("minimize")
        self.on_minimize()

    @Slot()
    def kill(self):
        self._debug("kill")
        self.on_kill()

    @Slot(str)
    def update_hit_regions(self, payload):
        self._debug("update_hit_regions", payload)
        self.hit_regions_changed.emit(payload)


class ButlerWidget(QMainWindow):
    status_signal = Signal(str)
    feed_signal = Signal(str, str)
    deferred_call_signal = Signal(int, object)
    MIN_WIDTH = 460
    MIN_HEIGHT = 132
    DEFAULT_WIDTH = 560
    DEFAULT_HEIGHT = 150
    MAX_WIDTH = 900
    MAX_HEIGHT = 260

    def __init__(
        self,
        on_mic_click: Callable[[], None],
        on_wake_toggle: Callable[[bool], None],
        on_text_submit: Callable[[str], None] = None,
        on_live_toggle: Callable[[bool], None] = None,
    ):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")

        self.on_mic_click = on_mic_click
        self.on_wake_toggle = on_wake_toggle
        self.on_text_submit = on_text_submit
        self.on_live_toggle = on_live_toggle

        self.api = ButlerAPI(
            on_mic=lambda: self.on_mic_click(),
            on_wake=lambda e: self.on_wake_toggle(e),
            on_text=lambda t: self.on_text_submit(t) if on_text_submit else None,
            on_live=lambda e: self.on_live_toggle(e) if on_live_toggle else None,
            on_minimize=self.minimize,
            on_kill=self.close_app,
        )
        self.api.drag_requested.connect(self.start_system_move)
        self.api.hit_regions_changed.connect(self._on_hit_regions_changed)
        self._hit_regions = {}

        self.view = QWebEngineView(self)
        self.view.page().setBackgroundColor(Qt.transparent)
        self.view.setAttribute(Qt.WA_NoSystemBackground)
        self.view.setAttribute(Qt.WA_TranslucentBackground)
        self.view.setStyleSheet("background: transparent; border: none; outline: none;")
        self.setContentsMargins(0, 0, 0, 0)
        self.view.setContentsMargins(0, 0, 0, 0)

        self.channel = QWebChannel()
        self.channel.registerObject("api", self.api)
        self.view.page().setWebChannel(self.channel)

        base_path = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(base_path, "web_ui", "index.html")
        self.view.setUrl(QUrl.fromLocalFile(html_path))

        self.setCentralWidget(self.view)

        screen = QApplication.primaryScreen().availableGeometry()
        initial_w = self.DEFAULT_WIDTH
        initial_h = self.DEFAULT_HEIGHT
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.setMaximumSize(self.MAX_WIDTH, self.MAX_HEIGHT)
        self.setGeometry(
            screen.width() - initial_w - 20,
            (screen.height() - initial_h) // 2,
            initial_w,
            initial_h,
        )

        self.status_signal.connect(self._do_set_status)
        self.feed_signal.connect(self._do_append_feed)
        self.deferred_call_signal.connect(self._do_after)

    def nativeEvent(self, eventType, message):
        """Keep the widget interactive inside its window bounds."""
        import ctypes
        import ctypes.wintypes

        WM_NCHITTEST = 0x0084
        HTTRANSPARENT = -1
        HTCLIENT = 1
        HTBOTTOMRIGHT = 17

        if eventType == b"windows_generic_MSG":
            msg = ctypes.wintypes.MSG.from_address(int(message))
            if msg.message == WM_NCHITTEST:
                raw_x = msg.lParam & 0xFFFF
                raw_y = (msg.lParam >> 16) & 0xFFFF
                if raw_x & 0x8000:
                    raw_x -= 0x10000
                if raw_y & 0x8000:
                    raw_y -= 0x10000
                
                scale = self.devicePixelRatioF() or 1.0
                x = int(round(raw_x / scale)) - self.x()
                y = int(round(raw_y / scale)) - self.y()

                if 0 <= x < self.width() and 0 <= y < self.height():
                    if self._point_in_region("resize", x, y):
                        return True, HTBOTTOMRIGHT
                    # All other clicks inside the window go to WebEngine.
                    # Button actions (mic, live, text, wake, kill, minimize)
                    # are handled by JS onclick -> QWebChannel slots.
                    return True, HTCLIENT
                # Outside the window bounds - click passes through to desktop
                return True, HTTRANSPARENT
        return super().nativeEvent(eventType, message)

    def _on_hit_regions_changed(self, payload: str):
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return

        regions = {}
        for name in ("dock", "drag", "resize"):
            rect = data.get(name)
            if not isinstance(rect, dict):
                continue
            try:
                regions[name] = {
                    "x": float(rect["x"]),
                    "y": float(rect["y"]),
                    "w": float(rect["w"]),
                    "h": float(rect["h"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
        self._hit_regions = regions

    def _point_in_region(self, name: str, x: int, y: int) -> bool:
        rect = self._hit_regions.get(name)
        if not rect:
            return False
        return (
            rect["x"] <= x <= rect["x"] + rect["w"]
            and rect["y"] <= y <= rect["y"] + rect["h"]
        )

    def minimize(self):
        self.showMinimized()

    def close_app(self):
        app = QApplication.instance()
        if app is not None:
            app.quit()
            return
        self.close()

    def closeEvent(self, event):
        event.accept()
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(0, app.quit)

    def start_system_move(self):
        """Bulletproof Windows native drag via Win32 API."""
        import ctypes

        hwnd = int(self.winId())
        ctypes.windll.user32.ReleaseCapture()
        ctypes.windll.user32.SendMessageW(hwnd, 0x00A1, 2, 0)

    def set_status(self, status: str):
        self.status_signal.emit(status)

    def _do_set_status(self, status: str):
        self.view.page().runJavaScript(f"window.updateStatus({json.dumps(status)})")

    def append_feed(self, role: str, text: str):
        self.feed_signal.emit(role, text)

    def _do_append_feed(self, role: str, text: str):
        role_map = {"Heard": "user", "You": "user", "Butler": "butler"}
        js_role = role_map.get(role, role.lower())
        self.view.page().runJavaScript(
            f"window.appendLog({json.dumps(js_role)}, {json.dumps(text)})"
        )

    def show_tool_confirmation(self, text: str):
        self.view.page().runJavaScript(f"console.log({json.dumps(f'Tool Confirmation: {text}')})")

    def show_plan_review(self, text: str):
        self.view.page().runJavaScript(f"console.log({json.dumps(f'Plan Review: {text}')})")

    def after(self, ms, func):
        self.deferred_call_signal.emit(int(ms), func)

    def _do_after(self, ms, func):
        QTimer.singleShot(max(0, int(ms)), func)

    def update(self):
        pass

    def mainloop(self):
        pass
