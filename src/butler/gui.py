import customtkinter as ctk
import threading

class ButlerWidget(ctk.CTk):
    def __init__(self, on_mic_click, on_wake_toggle):
        super().__init__()
        
        self.title("Butler")
        self.geometry("300x120")
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        self.configure(fg_color="#1e1e1e")
        
        self.on_mic_click = on_mic_click
        self.on_wake_toggle = on_wake_toggle
        
        # Center the window on screen to start
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = int(screen_width/2 - 150)
        y = int(screen_height - 150)
        self.geometry(f"+{x}+{y}")
        
        self.setup_ui()
        self.bind_movement()
        
    def setup_ui(self):
        # Top bar with dragging
        self.top_frame = ctk.CTkFrame(self, fg_color="#2a2a2a", corner_radius=0, height=24)
        self.top_frame.pack(fill="x")
        
        self.status_label = ctk.CTkLabel(self.top_frame, text="Asleep", text_color="#aaaaaa", font=("Arial", 12))
        self.status_label.pack(side="left", padx=10)
        
        self.close_btn = ctk.CTkButton(self.top_frame, text="X", width=20, height=20, 
                                       fg_color="transparent", hover_color="#ff4444",
                                       command=self.destroy)
        self.close_btn.pack(side="right", padx=5)
        
        # Main controls
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, pady=10)
        
        self.wake_var = ctk.StringVar(value="off")
        self.wake_switch = ctk.CTkSwitch(self.main_frame, text="Hi Butler", 
                                         command=self._handle_toggle,
                                         variable=self.wake_var, onvalue="on", offvalue="off")
        self.wake_switch.pack(side="left", padx=20)
        
        self.mic_btn = ctk.CTkButton(self.main_frame, text="🎙", width=50, height=50,
                                     corner_radius=25, font=("Arial", 24),
                                     command=self.on_mic_click)
        self.mic_btn.pack(side="right", padx=20)
        
    def set_status(self, text: str):
        self.status_label.configure(text=text)
        self.update_idletasks()
        
    def _handle_toggle(self):
        self.on_wake_toggle(self.wake_var.get() == "on")
        
    def bind_movement(self):
        self.top_frame.bind("<ButtonPress-1>", self.start_move)
        self.top_frame.bind("<ButtonRelease-1>", self.stop_move)
        self.top_frame.bind("<B1-Motion>", self.do_move)
        self.status_label.bind("<ButtonPress-1>", self.start_move)
        self.status_label.bind("<B1-Motion>", self.do_move)

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def stop_move(self, event):
        self.x = None
        self.y = None

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        self.geometry(f"+{x}+{y}")
