import time
import logging
from pathlib import Path
from threading import Thread, Timer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class WorkspaceWatcher(FileSystemEventHandler):
    """Monitors allowed roots and triggers incremental indexing on changes."""
    
    def __init__(self, runtime, debounce_seconds: float = 2.0):
        self.runtime = runtime
        self.debounce_seconds = debounce_seconds
        self.observer = Observer()
        self._pending_files = set()
        self._timer = None

    def start(self):
        roots = self.runtime.config.allowed_roots
        if not roots:
            logger.info("watcher_no_roots skipping start")
            return

        for root in roots:
            p = Path(root)
            if p.exists() and p.is_dir():
                self.observer.schedule(self, str(p), recursive=True)
                logger.info("watcher_started path=%s", p)
        
        self.observer.start()

    def stop(self):
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        if self._timer:
            self._timer.cancel()

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule_sync(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._schedule_sync(event.src_path)

    def _schedule_sync(self, path: str):
        # Ignore common noise
        p = Path(path)
        if any(part.startswith(".") or part == "__pycache__" for part in p.parts):
            return
        
        self._pending_files.add(path)
        
        if self._timer:
            self._timer.cancel()
            
        self._timer = Timer(self.debounce_seconds, self._flush_sync)
        self._timer.start()

    def _flush_sync(self):
        files = list(self._pending_files)
        self._pending_files.clear()
        
        if not files:
            return
            
        logger.info("watcher_syncing count=%d", len(files))
        try:
            # Call index.sync tool incrementally
            self.runtime.tools.call("index.sync", {"paths": files})
        except Exception as e:
            logger.warning("watcher_sync_failed error=%s", e)
