import requests
import json
from butler.tools.base import Tool, Parameter

class SendTelegramMessage(Tool):
    """
    Sends a message to the user via Telegram.
    This is extremely useful for sending long outputs like recipes, code snippets, or lists to the user's phone so they can read it later.
    """
    
    @property
    def name(self) -> str:
        return "send_telegram_message"

    @property
    def description(self) -> str:
        return "Sends a message to the user via Telegram. Use this when the user asks you to send something to their Telegram, or if the output is too long to speak over voice."

    @property
    def parameters(self) -> list[Parameter]:
        return [
            Parameter(
                name="message",
                type="string",
                description="The text message to send to the user's Telegram. Supports basic markdown.",
                required=True
            )
        ]

    def execute(self, message: str) -> str:
        token = self.config.telegram_bot_token
        if not token:
            return "Error: BUTLER_TELEGRAM_BOT_TOKEN is not configured."
            
        chat_id = self.config.telegram_chat_id
        
        # If we don't have the chat_id yet, try to fetch it from recent bot updates
        if not chat_id:
            try:
                updates_url = f"https://api.telegram.org/bot{token}/getUpdates"
                resp = requests.get(updates_url).json()
                if resp.get("ok") and resp.get("result"):
                    # Get the most recent message's chat ID
                    chat_id = resp["result"][-1]["message"]["chat"]["id"]
            except Exception as e:
                pass
                
        if not chat_id:
            return "Error: Could not determine Telegram Chat ID. Tell the user they need to send at least one message to the Telegram bot first before I can reply to them."
            
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, json=payload).json()
        if response.get("ok"):
            return "Successfully sent message to user's Telegram."
        else:
            return f"Failed to send Telegram message: {response.get('description')}"
