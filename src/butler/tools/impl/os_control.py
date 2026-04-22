import time
import subprocess
from typing import Any, Literal
from pydantic import BaseModel, Field

from butler.tools.base import Tool, ToolContext

# ---------------------------------------------------------------------------
# Volume Control
# ---------------------------------------------------------------------------
class OSVolumeArgs(BaseModel):
    level: int | None = Field(default=None, description="Volume level from 0 to 100")
    mute: bool | None = Field(default=None, description="Set to true to mute, false to unmute")

def handle_os_volume(ctx: ToolContext, args: OSVolumeArgs) -> dict[str, Any]:
    level = args.level
    mute = args.mute
    
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        
        changes = []
        if mute is not None:
            volume.SetMute(1 if mute else 0, None)
            changes.append(f"Mute set to {mute}")
            
        if level is not None:
            level = max(0, min(100, level))
            scalar = level / 100.0
            volume.SetMasterVolumeLevelScalar(scalar, None)
            changes.append(f"Volume set to {level}%")
            
        new_vol = int(volume.GetMasterVolumeLevelScalar() * 100)
        new_mute = bool(volume.GetMute())
        
        return {
            "success": True,
            "output": " ".join(changes) if changes else "No action taken",
            "current_state": {"level": new_vol, "muted": new_mute}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Brightness Control
# ---------------------------------------------------------------------------
class OSBrightnessArgs(BaseModel):
    level: int = Field(description="Brightness level from 0 to 100")

def handle_os_brightness(ctx: ToolContext, args: OSBrightnessArgs) -> dict[str, Any]:
    level = max(0, min(100, args.level))
    ps_script = f"(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, {level})"
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True, capture_output=True)
        return {"success": True, "level": level}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": "Brightness control not supported on this monitor", "details": e.stderr.decode(errors='ignore')}

# ---------------------------------------------------------------------------
# Power Control
# ---------------------------------------------------------------------------
class OSPowerArgs(BaseModel):
    action: Literal["lock", "sleep", "shutdown", "restart"]

def handle_os_power(ctx: ToolContext, args: OSPowerArgs) -> dict[str, Any]:
    action = args.action
    try:
        if action == "lock":
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=True)
        elif action == "sleep":
            subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=True)
        elif action == "shutdown":
            subprocess.run(["shutdown", "/s", "/t", "0"], check=True)
        elif action == "restart":
            subprocess.run(["shutdown", "/r", "/t", "0"], check=True)
        return {"success": True, "action": action}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Media Control
# ---------------------------------------------------------------------------
class OSMediaArgs(BaseModel):
    action: Literal["play_pause", "next", "previous"]

def handle_os_media(ctx: ToolContext, args: OSMediaArgs) -> dict[str, Any]:
    action = args.action
    try:
        import keyboard
        key_map = {
            "play_pause": "play/pause media",
            "next": "next track",
            "previous": "previous track"
        }
        keyboard.send(key_map[action])
        return {"success": True, "action": action}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ---------------------------------------------------------------------------
# App Launch / Kill
# ---------------------------------------------------------------------------
class OSAppLaunchArgs(BaseModel):
    app_name: str = Field(description="Name of the application or executable to launch (e.g., 'notepad', 'chrome')")

def handle_os_launch(ctx: ToolContext, args: OSAppLaunchArgs) -> dict[str, Any]:
    app_name = args.app_name
    try:
        subprocess.Popen(f"start {app_name}", shell=True)
        return {"success": True, "app": app_name}
    except Exception as e:
        return {"success": False, "error": str(e)}

class OSAppKillArgs(BaseModel):
    app_name: str = Field(description="Name of the app process to forcefully close (e.g., 'chrome.exe', 'notepad')")

def handle_os_kill(ctx: ToolContext, args: OSAppKillArgs) -> dict[str, Any]:
    app_name = args.app_name
    if not app_name.lower().endswith(".exe"):
        app_name += ".exe"
        
    try:
        res = subprocess.run(["taskkill", "/F", "/IM", app_name], capture_output=True, text=True)
        if res.returncode == 0:
            return {"success": True, "output": f"Killed {app_name}"}
        else:
            return {"success": False, "error": res.stderr.strip() or res.stdout.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Wi-Fi Control
# ---------------------------------------------------------------------------
class OSWifiArgs(BaseModel):
    action: Literal["status", "disconnect"]

def handle_os_wifi(ctx: ToolContext, args: OSWifiArgs) -> dict[str, Any]:
    action = args.action
    try:
        if action == "status":
            res = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True)
            return {"success": True, "output": res.stdout}
        elif action == "disconnect":
            res = subprocess.run(["netsh", "wlan", "disconnect"], capture_output=True, text=True)
            return {"success": True, "output": res.stdout}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Clipboard Access
# ---------------------------------------------------------------------------
class OSClipboardArgs(BaseModel):
    action: Literal["read", "write"]
    text: str | None = Field(default=None, description="The string to copy to the clipboard if action is 'write'")

def handle_os_clipboard(ctx: ToolContext, args: OSClipboardArgs) -> dict[str, Any]:
    action = args.action
    text = args.text
    try:
        import pyperclip
        if action == "write":
            if text is None:
                return {"success": False, "error": "Missing 'text' parameter for write action"}
            pyperclip.copy(text)
            return {"success": True, "message": "Copied to clipboard"}
        else:
            return {"success": True, "clipboard_content": pyperclip.paste()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Registry Hook
# ---------------------------------------------------------------------------
def build() -> list[Tool]:
    return [
        Tool[OSVolumeArgs](
            name="os.volume",
            description="Control the Windows system volume. You can set a specific percentage (0-100) or toggle mute.",
            input_model=OSVolumeArgs,
            handler=handle_os_volume,
            side_effect=True
        ),
        Tool[OSBrightnessArgs](
            name="os.brightness",
            description="Control the Windows display brightness.",
            input_model=OSBrightnessArgs,
            handler=handle_os_brightness,
            side_effect=True
        ),
        Tool[OSPowerArgs](
            name="os.power",
            description="Control system power state (lock screen, sleep, shutdown, restart).",
            input_model=OSPowerArgs,
            handler=handle_os_power,
            side_effect=True
        ),
        Tool[OSMediaArgs](
            name="os.media",
            description="Send virtual media keys (play/pause, next track, previous track).",
            input_model=OSMediaArgs,
            handler=handle_os_media,
            side_effect=True
        ),
        Tool[OSAppLaunchArgs](
            name="os.launch_app",
            description="Launch a Windows application by its executable name or registered URI command.",
            input_model=OSAppLaunchArgs,
            handler=handle_os_launch,
            side_effect=True
        ),
        Tool[OSAppKillArgs](
            name="os.kill_app",
            description="Forcefully close a running Windows application by its process name.",
            input_model=OSAppKillArgs,
            handler=handle_os_kill,
            side_effect=True
        ),
        Tool[OSWifiArgs](
            name="os.wifi_control",
            description="Check Wi-Fi connection status or disconnect from the current network.",
            input_model=OSWifiArgs,
            handler=handle_os_wifi,
            side_effect=True
        ),
        Tool[OSClipboardArgs](
            name="os.clipboard",
            description="Read the current text from the Windows clipboard or write new text to it.",
            input_model=OSClipboardArgs,
            handler=handle_os_clipboard,
            side_effect=True
        )
    ]
