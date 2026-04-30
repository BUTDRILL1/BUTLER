import os
import sys
import threading
from typing import Callable, Any
from PySide6.QtCore import QRect
from PySide6.QtCore import Qt, QUrl, QObject, Slot, Signal, QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QCursor
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebChannel import QWebChannel

class ButlerAPI(QObject):
    """
    The Neural Bridge between Python and JavaScript.
    Methods marked with @Slot() are callable from index.html.
    """
    # Signals for thread-safe UI updates
    window_resize_requested = Signal(int, int)
    chat_state_changed = Signal(bool)
    text_state_changed = Signal(bool)
    expand_state_changed = Signal(bool)
    hit_regions_updated = Signal(str)
    drag_requested = Signal()
    
    def __init__(self, on_mic, on_wake, on_text, on_live, on_minimize, on_kill):
        super().__init__()
        self.on_mic = on_mic
        self.on_wake = on_wake
        self.on_text = on_text
        self.on_live = on_live
        self.on_minimize = on_minimize
        self.on_kill = on_kill

    @Slot()
    def on_mic_click(self):
        self.on_mic()

    @Slot(bool)
    def on_wake_toggle(self, enabled):
        self.on_wake(enabled)

    @Slot(str)
    def on_text_submit(self, text):
        self.on_text(text)

    @Slot(bool)
    def on_live_toggle(self, enabled):
        self.on_live(enabled)

    @Slot(bool)
    def on_chat_toggle(self, enabled):
        self.chat_state_changed.emit(enabled)

    @Slot(bool)
    def on_text_toggle(self, enabled):
        self.text_state_changed.emit(enabled)

    @Slot(bool)
    def on_expand_toggle(self, expanded):
        self.expand_state_changed.emit(expanded)

    @Slot(str)
    def sync_hit_regions(self, rects_json):
        self.hit_regions_updated.emit(rects_json)

    @Slot()
    def start_drag(self):
        self.drag_requested.emit()

    @Slot()
    def minimize(self):
        self.on_minimize()

    @Slot()
    def kill(self):
        self.on_kill()

class ButlerWidget(QMainWindow):
    # Signals for thread-safe status/feed updates
    status_signal = Signal(str)
    feed_signal = Signal(str, str)

    def __init__(
        self, 
        on_mic_click: Callable[[], None], 
        on_wake_toggle: Callable[[bool], None],
        on_text_submit: Callable[[str], None] = None,
        on_tool_confirm: Callable[[bool], None] = None,
        on_persona_change: Callable[[str], None] = None,
        on_plan_confirm: Callable[[bool], None] = None,
        on_live_toggle: Callable[[bool], None] = None,
        initial_persona: str = "Executive"
    ):
        super().__init__()
        
        # 1. Windows Ghosting Setup (Transparency)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        
        # 2. State & Callbacks
        self.on_mic_click = on_mic_click
        self.on_wake_toggle = on_wake_toggle
        self.on_text_submit = on_text_submit
        self.on_live_toggle = on_live_toggle
        self._expanded = False
        self._chat_open = False
        self._text_open = False
        
        # 3. The API Bridge
        self.api = ButlerAPI(
            on_mic=lambda: self.on_mic_click(),
            on_wake=lambda e: self.on_wake_toggle(e),
            on_text=lambda t: self.on_text_submit(t) if on_text_submit else None,
            on_live=lambda e: self.on_live_toggle(e) if on_live_toggle else None,
            on_minimize=self.minimize,
            on_kill=self.close_app
        )
        self.api.window_resize_requested.connect(self.resize_window)
        self.api.chat_state_changed.connect(self._on_chat_toggle)
        self.api.text_state_changed.connect(self._on_text_toggle)
        self.api.expand_state_changed.connect(self._on_expand_toggle)
        self.api.hit_regions_updated.connect(self._on_hit_regions)
        self.api.drag_requested.connect(self.start_system_move)
        
        # 4. Web Engine Setup
        self.view = QWebEngineView(self)
        self.view.page().setBackgroundColor(Qt.transparent)
        self.view.setAttribute(Qt.WA_NoSystemBackground)
        self.view.setAttribute(Qt.WA_TranslucentBackground)
        self.view.setStyleSheet("background: transparent; border: none; outline: none;")
        self.setContentsMargins(0, 0, 0, 0)
        self.view.setContentsMargins(0, 0, 0, 0)
        
        # Set up QWebChannel
        self.channel = QWebChannel()
        self.channel.registerObject("api", self.api)
        self.view.page().setWebChannel(self.channel)
        
        # Initialize hit regions
        self._hit_rects = []
        self._update_hit_regions_fallback(160)

        # 5. Load Content
        base_path = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(base_path, "web_ui", "index.html")
        self.view.setUrl(QUrl.fromLocalFile(html_path))
        
        # 6. Layout — transparent click-through
        self.setCentralWidget(self.view)
        
        # 7. Position: center-right of screen
        screen = QApplication.primaryScreen().geometry()
        initial_w = 160
        initial_h = 240
        # Right edge 20px from screen right edge
        self.setGeometry(screen.width() - initial_w - 20, (screen.height() - initial_h) // 2, initial_w, initial_h)
        
        # Connect signals
        self.status_signal.connect(self._do_set_status)
        self.feed_signal.connect(self._do_append_feed)

    def nativeEvent(self, eventType, message):
        """Per-pixel click-through: only interactive regions receive clicks."""
        import ctypes
        import ctypes.wintypes
        WM_NCHITTEST = 0x0084
        HTTRANSPARENT = -1
        HTCLIENT = 1
        
        if eventType == b'windows_generic_MSG':
            msg = ctypes.wintypes.MSG.from_address(int(message))
            if msg.message == WM_NCHITTEST:
                x = ctypes.c_int16(msg.lParam & 0xFFFF).value - self.x()
                y = ctypes.c_int16((msg.lParam >> 16) & 0xFFFF).value - self.y()
                for rect in self._hit_rects:
                    if rect.contains(x, y):
                        return True, HTCLIENT
                return True, HTTRANSPARENT
        return super().nativeEvent(eventType, message)

    def _on_expand_toggle(self, expanded):
        self._expanded = expanded
        self._update_window_size()

    def _on_chat_toggle(self, chat_open):
        self._chat_open = chat_open
        self._update_window_size()

    def _on_text_toggle(self, text_open):
        self._text_open = text_open
        self._update_window_size()

    def _update_window_size(self):
        w = 160
        h = 240
        
        if self._expanded:
            w = max(w, 280)
            h = max(h, 340)
            
        if self._text_open:
            w = max(w, 400)
            h = max(h, 380)
            
        if self._chat_open:
            w = max(w, 520)
            h = max(h, 440)
            
        geom = self.geometry()
        top_right_x = geom.x() + geom.width()
        top_y = geom.y()
        self.setFixedSize(w, h)
        self.move(top_right_x - w, top_y)
        # Fallback hit rect update just in case JS is slow
        self._update_hit_regions_fallback(w)

    def _on_hit_regions(self, rects_json):
        """Receive actual element bounding rects from JS — the only source of truth."""
        import json
        try:
            rects = json.loads(rects_json)
            self._hit_rects = [
                QRect(int(r['x']), int(r['y']), int(r['w']), int(r['h']))
                for r in rects
            ]
        except Exception:
            pass

    def _update_hit_regions_fallback(self, w):
        """Fallback hit region for startup before JS reports."""
        hub_cx = w - 80
        hub_cy = 160
        self._hit_rects = [QRect(hub_cx - 50, hub_cy - 50, 140, 110)]

    def _trigger_drag(self):
        pass

    def resize_window(self, w, h):
        self.setFixedSize(w, h)

    def minimize(self):
        self.showMinimized()

    def close_app(self):
        self.close()
        os._exit(0)

    def closeEvent(self, event):
        self.close_app()
        event.accept()

    def start_system_move(self):
        """Bulletproof Windows native drag via Win32 API."""
        import ctypes
        hwnd = int(self.winId())
        ctypes.windll.user32.ReleaseCapture()
        ctypes.windll.user32.SendMessageW(hwnd, 0x00A1, 2, 0)

    # Thread-safe methods
    def set_status(self, status: str):
        self.status_signal.emit(status)

    def _do_set_status(self, status: str):
        safe_status = status.replace("'", "\\'")
        self.view.page().runJavaScript(f"window.updateStatus('{safe_status}')")

    def append_feed(self, role: str, text: str):
        self.feed_signal.emit(role, text)

    def _do_append_feed(self, role: str, text: str):
        safe_text = text.replace("'", "\\'").replace("\n", " ")
        role_map = {"Heard": "user", "You": "user", "Butler": "butler"}
        js_role = role_map.get(role, role.lower())
        self.view.page().runJavaScript(f"window.appendLog('{js_role}', '{safe_text}')")

    def show_tool_confirmation(self, text: str):
        self.view.page().runJavaScript(f"console.log('Tool Confirmation: {text}')")

    def show_plan_review(self, text: str):
        self.view.page().runJavaScript(f"console.log('Plan Review: {text}')")

    # Compatibility methods
    def after(self, ms, func):
        threading.Timer(ms/1000.0, func).start()

    def update(self):
        pass
    
    def mainloop(self):
        pass
