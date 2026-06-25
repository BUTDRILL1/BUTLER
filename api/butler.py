from http.server import BaseHTTPRequestHandler
import json
import re

from butler.config import load_config
from butler.db import ButlerDB
from butler.agent.memory import MemoryStore
from butler.tools.registry import build_default_tool_registry
from butler.agent.loop import AgentRuntime

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))
            
            query = body.get("query", "")
            if not query:
                self.send_error_response("Missing query")
                return

            config = load_config()
            db = ButlerDB.open(config)
            memory = MemoryStore(db)
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
