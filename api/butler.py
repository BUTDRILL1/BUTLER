from http.server import BaseHTTPRequestHandler
import json
import re
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import re

from butler.config import load_config
from butler.db import ButlerDB
from butler.agent.memory import MemoryStore
from butler.tools.registry import build_default_tool_registry
from butler.agent.loop import AgentRuntime

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "online",
            "message": "BUTLER is online and listening. Please send POST requests with {\"query\": \"...\"} to interact."
        }).encode('utf-8'))

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))
            
            # Check if this is a Telegram Webhook payload
            is_telegram = False
            chat_id = None
            if "message" in body and "chat" in body.get("message", {}):
                is_telegram = True
                chat_id = body["message"]["chat"]["id"]
                username = body["message"].get("from", {}).get("username", "")
                
                # Check for photo
                if "photo" in body["message"]:
                    import requests
                    import base64
                    config = load_config()
                    token = config.telegram_bot_token
                    caption = body["message"].get("caption", "")
                    # Get the highest resolution photo (last in the array)
                    photo_id = body["message"]["photo"][-1]["file_id"]
                    
                    # Fetch file path from Telegram
                    file_resp = requests.get(f"https://api.telegram.org/bot{token}/getFile?file_id={photo_id}").json()
                    if file_resp.get("ok"):
                        file_path = file_resp["result"]["file_path"]
                        # Download the image bytes
                        img_bytes = requests.get(f"https://api.telegram.org/file/bot{token}/{file_path}").content
                        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                        
                        # Process with Vision model
                        from butler.agent.provider import NvidiaProvider
                        vision_prov = NvidiaProvider(api_keys=config.nvidia_api_keys, model=config.vision_model)
                        prompt_text = caption if caption else "Describe this image in detail."
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                                ]
                            }
                        ]
                        try:
                            vision_desc = vision_prov.chat(messages)
                            query = f"[System Note: The user uploaded an image. Visual analysis: {vision_desc}]"
                            if caption:
                                query += f"\nUser's caption: {caption}"
                        except Exception as e:
                            query = f"[System Note: The user uploaded an image, but the vision model failed to process it: {str(e)}]"
                            if caption:
                                query += f"\nUser's caption: {caption}"
                    else:
                        query = "[System Note: The user uploaded an image, but I could not download it from Telegram.]"
                        
                else:
                    query = body["message"].get("text", "")
                
            else:
                query = body.get("query", "")
                
            if not query:
                # If there's no query, just return OK (Telegram sometimes sends edits or status updates)
                self.send_response(200)
                self.end_headers()
                return

            config = load_config()
            
            # Security check for Telegram
            if is_telegram and config.telegram_allowed_user:
                allowed = config.telegram_allowed_user.lower().lstrip("@")
                if str(chat_id) != allowed and username.lower() != allowed:
                    self.send_response(200)
                    self.end_headers()
                    return
            db = ButlerDB.open(config)

            from butler.agent.provider import AnthropicProvider, GeminiProvider, NvidiaProvider, OllamaProvider
            if config.provider == "gemini":
                prov = GeminiProvider(api_keys=config.gemini_api_keys, model=config.model)
            elif config.provider == "claude":
                prov = AnthropicProvider(api_keys=config.claude_api_keys, model=config.model)
            elif config.provider == "nvidia":
                prov = NvidiaProvider(api_keys=config.nvidia_api_keys, model=config.model)
            else:
                prov = OllamaProvider(base_url=config.ollama_url, model=config.model)

            from butler.paths import butler_home_dir
            mem_db = butler_home_dir() / "memory.db"
            memory = MemoryStore(str(mem_db), prov)
            
            tools = build_default_tool_registry(config, db, memory)

            # Retrieve conversation history
            cid = db.get_last_conversation()
            if not cid:
                cid = db.new_conversation()

            # Initialize the runtime
            runtime = AgentRuntime(
                config=config,
                db=db,
                tools=tools,
                memory=memory,
                conversation_id=cid
            )

            # Run the agent loop. It will automatically call tools (like adding skills)
            response_text = ""
            for token in runtime.chat_once_stream(query, auto_approve=True):
                response_text += token
                
            def clean_for_speech(text):
                # Remove markdown asterisks, backticks, hashes
                t = re.sub(r'[*`#]', '', text)
                # Remove markdown links
                t = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', t)
                return t.strip()

            final_clean = clean_for_speech(response_text)
            
            # Simple exit check
            exit_phrases = ["bye", "stop", "close butler", "goodbye", "sign off"]
            is_exit = any(p in query.lower() for p in exit_phrases)

            import requests
            
            # If this request came from Telegram, send the response back via Telegram API
            if is_telegram and config.telegram_bot_token:
                # Send back to Telegram
                tg_url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
                # Telegram has a 4096 char limit, but we'll assume responses are standard length
                for i in range(0, len(response_text), 4000):
                    chunk = response_text[i:i+4000]
                    requests.post(tg_url, json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"})
                
                # Telegram requires a 200 OK immediately
                self.send_response(200)
                self.end_headers()
                return
            else:
                # Original Siri Shortcut behavior
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                self.wfile.write(json.dumps({
                    "response": final_clean,
                    "exit": is_exit
                }).encode('utf-8'))
            
        except Exception as e:
            self.send_error_response(f"Sorry, something went wrong on my end: {str(e)}")

    def send_error_response(self, message):
        self.send_response(200) # Always return 200 for Siri TTS to speak the error
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            "response": message,
            "error": True
        }).encode('utf-8'))
