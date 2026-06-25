"""
Diagnostic 5: Full pipeline with QWebChannel + click listener
"""
import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"

from PySide6.QtCore import QObject, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage

class TestAPI(QObject):
    drag_requested = Signal()
    hit_regions_changed = Signal(str)

    @Slot()
    def on_mic_click(self): print("[SLOT] on_mic_click", flush=True)
    @Slot(bool)
    def on_wake_toggle(self, e): print(f"[SLOT] on_wake_toggle: {e}", flush=True)
    @Slot(str)
    def on_text_submit(self, t): print(f"[SLOT] on_text_submit: {t}", flush=True)
    @Slot(bool)
    def on_live_toggle(self, e): print(f"[SLOT] on_live_toggle: {e}", flush=True)
    @Slot(bool)
    def on_text_toggle(self, e): print(f"[SLOT] on_text_toggle: {e}", flush=True)
    @Slot()
    def start_drag(self): print("[SLOT] start_drag", flush=True); self.drag_requested.emit()
    @Slot()
    def minimize(self): print("[SLOT] minimize", flush=True)
    @Slot()
    def kill(self): print("[SLOT] kill", flush=True)
    @Slot(str)
    def update_hit_regions(self, p): print("[SLOT] update_hit_regions", flush=True); self.hit_regions_changed.emit(p)


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, msg, line, src):
        print(f"[JS] {msg}", flush=True)


app = QApplication(sys.argv)
win = QMainWindow()
win.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Window)
win.setAttribute(Qt.WA_TranslucentBackground, True)
win.setStyleSheet("background: transparent;")

page = Page()
page.setBackgroundColor(Qt.transparent)

# Set up channel BEFORE loading URL
api = TestAPI()
channel = QWebChannel()
channel.registerObject("api", api)
page.setWebChannel(channel)

view = QWebEngineView()
view.setPage(page)
view.setAttribute(Qt.WA_TranslucentBackground)

base = os.path.dirname(os.path.abspath(__file__))
html = os.path.join(base, "src", "butler", "web_ui", "index.html")
view.setUrl(QUrl.fromLocalFile(html))

win.setCentralWidget(view)
win.setGeometry(500, 300, 560, 150)

def check():
    page.runJavaScript("console.log('api=' + !!window.api + ' btns=' + document.querySelectorAll('.control-btn').length);")

def inject_click_listener():
    page.runJavaScript("""
        document.addEventListener('click', function(e) {
            console.log('CLICK: ' + e.target.tagName + ' class=' + e.target.className.substring(0,50));
        }, true);
        console.log('click listener injected');
    """)

page.loadFinished.connect(lambda ok: print(f"[PY] loaded: {ok}", flush=True))
QTimer.singleShot(3000, check)
QTimer.singleShot(3500, inject_click_listener)

def native_event(eventType, message):
    import ctypes, ctypes.wintypes
    if eventType == b"windows_generic_MSG":
        msg = ctypes.wintypes.MSG.from_address(int(message))
        if msg.message == 0x0084:  # WM_NCHITTEST
            return True, 1  # HTCLIENT
    return QMainWindow.nativeEvent(win, eventType, message)

win.nativeEvent = native_event
win.show()
print("[PY] Window shown. Click buttons!", flush=True)
sys.exit(app.exec())
