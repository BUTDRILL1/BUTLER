import json
from http.server import BaseHTTPRequestHandler
from butler.config import load_config
from butler.db import ButlerDB

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        import requests
        try:
            config = load_config()
            db = ButlerDB.open(config)
            
            chat_id_to_use = config.telegram_chat_id

            due = db.get_pending_reminders()
            fired_count = 0
            if due and config.telegram_bot_token and chat_id_to_use:
                tg_url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
                for r in due:
                    try:
                        text = f"🔔 *REMINDER:* {r['message']}"
                        requests.post(tg_url, json={"chat_id": chat_id_to_use, "text": text, "parse_mode": "Markdown"})
                        
                        recurrence = r.get("recurrence_minutes")
                        if recurrence:
                            db.reschedule_recurring_reminder(r["id"], recurrence)
                        else:
                            db.mark_reminder_sent(r["id"])
                        fired_count += 1
                    except Exception as e:
                        print(f"Failed to fire reminder {r['id']}: {e}")

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "cron_executed",
                "reminders_fired": fired_count
            }).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
