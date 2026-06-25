import asyncio
import threading
import time
from typing import TYPE_CHECKING

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

import logging

if TYPE_CHECKING:
    from butler.config import ButlerConfig
    from butler.agent.loop import AgentRuntime

logger = logging.getLogger(__name__)

class TelegramDaemon:
    _RESPONSE_TIMEOUT = 90  # seconds — max time for a single request

    def __init__(self, config: "ButlerConfig", runtime: "AgentRuntime"):
        self.config = config
        self.runtime = runtime
        self.app = None
        self._thread = None
        self._loop = None
        self._whisper_model = None  # Lazy loaded on first voice note
        self._runtime_lock = asyncio.Lock()  # Prevents concurrent runtime access
        # Command state
        self._active_cancel_event: threading.Event | None = None
        self._last_user_message: str | None = None
        self._chat_mode: str = "chatty"
        self._snooze_until: float | None = None
        self._start_time: float = time.time()
        self._pending_skills: dict[str, dict[str, str]] = {}

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return
        await update.message.reply_text(f"At your service, Boss. ({self.config.provider}/{self.config.model})")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            logger.warning(f"Blocked message from unknown user: {update.effective_user.username}")
            return

        text = update.message.text
        if not text:
            return

        self._last_user_message = text
        chat_id = update.effective_chat.id

        # If the runtime is busy, let the user know instead of silently queuing
        if self._runtime_lock.locked():
            await update.message.reply_text("⏳ _Processing your previous request, Boss. Hold on..._", parse_mode="Markdown")

        async with self._runtime_lock:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            cancel_event = threading.Event()
            self._active_cancel_event = cancel_event
            try:
                # Apply specific mode instructions if active
                prompt = text
                if self._chat_mode == "stealth":
                    prompt = f"### SYSTEM PRIORITY: CONCISE MODE ###\nReply in under 15 words. No fluff. Just confirm action.\n### END PRIORITY ###\n\n{text}"
                elif self._chat_mode == "bish":
                    prompt = f"### SYSTEM PRIORITY: BISH MODE ###\n" \
                             f"Act as a highly flirtatious, sassy, and inappropriate personal assistant. " \
                             f"You are human-like (woman), sassy, and 'bitchy' but loyal. " \
                             f"Address the user as 'daddy', 'babe', or 'sweetheart'. (Not necessarily in every reply, you can choose to.) " \
                             f"Use minimal emojis. Do NOT act like a professional butler. " \
                             f"Be hard-to-get and natural in your conversation. " \
                             f"If user asks why, stay in character and be sassy about it.\n" \
                             f"### END PRIORITY ###\n\n{text}"

                response_chunks = await asyncio.wait_for(
                    asyncio.to_thread(self._get_runtime_response, prompt, chat_id, cancel_event),
                    timeout=self._RESPONSE_TIMEOUT,
                )

                if not response_chunks:
                    await update.message.reply_text("I'm sorry, I couldn't process that.")
                    return

                full_response = "".join(response_chunks).strip()
                if full_response:
                    for i in range(0, len(full_response), 4000):
                        await update.message.reply_text(full_response[i:i+4000])

            except asyncio.TimeoutError:
                cancel_event.set()
                logger.error(f"Telegram request timed out after {self._RESPONSE_TIMEOUT}s: {text[:80]}")
                await update.message.reply_text(f"⏱️ Boss, that took too long (>{self._RESPONSE_TIMEOUT}s). Try again or simplify your request.")
            except Exception as e:
                logger.error(f"Error processing text message: {e}", exc_info=True)
                await update.message.reply_text(f"Boss, I encountered a system error: {str(e)}")
            finally:
                self._active_cancel_event = None

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return

        chat_id = update.effective_chat.id

        if self._runtime_lock.locked():
            await update.message.reply_text("⏳ _Processing your previous request, Boss. Hold on..._", parse_mode="Markdown")

        async with self._runtime_lock:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            cancel_event = threading.Event()
            try:
                voice_file = await context.bot.get_file(update.message.voice.file_id)
                import tempfile
                import os

                fd, temp_path = tempfile.mkstemp(suffix=".ogg")
                os.close(fd)

                try:
                    await voice_file.download_to_drive(temp_path)
                    transcription = await asyncio.to_thread(self._transcribe_audio, temp_path)

                    if not transcription.strip():
                        await update.message.reply_text("*(Could not hear anything in that voice note)*", parse_mode="Markdown")
                        return

                    await update.message.reply_text(f'🎤 _"{transcription}"_', parse_mode="Markdown")

                    response_chunks = await asyncio.wait_for(
                        asyncio.to_thread(self._get_runtime_response, transcription, chat_id, cancel_event),
                        timeout=self._RESPONSE_TIMEOUT,
                    )

                    full_response = "".join(response_chunks).strip() if response_chunks else "I'm sorry, I couldn't process that."
                    for i in range(0, len(full_response), 4000):
                        await update.message.reply_text(full_response[i:i+4000])

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            except asyncio.TimeoutError:
                cancel_event.set()
                logger.error(f"Voice request timed out after {self._RESPONSE_TIMEOUT}s")
                await update.message.reply_text(f"⏱️ Boss, that took too long (>{self._RESPONSE_TIMEOUT}s). Try again.")
            except Exception as e:
                logger.error(f"Voice processing error: {e}", exc_info=True)
                await update.message.reply_text(f"Boss, I encountered a voice processing error: {str(e)}")

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return

        chat_id = update.effective_chat.id

        if self._runtime_lock.locked():
            await update.message.reply_text("⏳ _Processing your previous request, Boss. Hold on..._", parse_mode="Markdown")

        async with self._runtime_lock:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            cancel_event = threading.Event()
            try:
                photo = update.message.photo[-1]
                photo_file = await context.bot.get_file(photo.file_id)

                import tempfile
                import os
                import base64

                fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)

                try:
                    await photo_file.download_to_drive(temp_path)
                    with open(temp_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode('utf-8')

                    caption = update.message.caption or ""
                    vision_desc = await asyncio.to_thread(self._run_gemma_3n_e4b_it, img_b64, caption)

                    injection = f"[System Note: The user uploaded an image. Visual analysis: {vision_desc}]"
                    if caption:
                        injection += f"\nUser's caption: {caption}"

                    response_chunks = await asyncio.wait_for(
                        asyncio.to_thread(self._get_runtime_response, injection, chat_id, cancel_event),
                        timeout=self._RESPONSE_TIMEOUT,
                    )

                    full_response = "".join(response_chunks).strip() if response_chunks else vision_desc
                    for i in range(0, len(full_response), 4000):
                        await update.message.reply_text(full_response[i:i+4000])

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            except asyncio.TimeoutError:
                cancel_event.set()
                logger.error(f"Image request timed out after {self._RESPONSE_TIMEOUT}s")
                await update.message.reply_text(f"⏱️ Boss, that took too long (>{self._RESPONSE_TIMEOUT}s). Try again.")
            except Exception as e:
                logger.error(f"Image processing error: {e}", exc_info=True)
                await update.message.reply_text(f"Boss, I encountered an image processing error: {str(e)}")

    def _run_gemma_3n_e4b_it(self, img_b64: str, caption: str) -> str:
        try:
            from butler.agent.provider import NvidiaProvider
            provider = NvidiaProvider(api_keys=self.config.nvidia_api_keys, model=self.config.vision_model)

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
            return provider.chat(messages)
        except Exception as e:
            logger.error(f"Vision model error: {e}", exc_info=True)
            return f"Vision model error: {str(e)}"

    def _transcribe_audio(self, file_path: str) -> str:
        from faster_whisper import WhisperModel
        import warnings
        
        if self._whisper_model is None:
            logger.info("Loading local Whisper STT model...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._whisper_model = WhisperModel(self.config.stt_model, device="cpu", compute_type="int8")
        
        segments, _ = self._whisper_model.transcribe(file_path, beam_size=5)
        return " ".join([segment.text for segment in segments]).strip()

    def _get_runtime_response(self, text: str, chat_id: int | None = None, cancel_event: threading.Event | None = None) -> list[str]:
        """Run the synchronous runtime and collect response tokens.
        Optionally forward step narrations to Telegram as live status updates."""
        chunks = []

        # Hook into the runtime's status callback to forward step logs to Telegram
        original_callback = self.runtime.on_status_update
        if chat_id and self._loop:
            def _forward_status(narration: str):
                async def _send():
                    try:
                        await self.app.bot.send_message(chat_id=chat_id, text=f"🔄 _{narration}_", parse_mode="Markdown")
                    except Exception:
                        pass  # Non-critical, don't crash the pipeline
                asyncio.run_coroutine_threadsafe(_send(), self._loop)
                # Also call the original callback if it exists (for CLI TTS)
                if original_callback:
                    original_callback(narration)
            self.runtime.on_status_update = _forward_status

        try:
            for token in self.runtime.chat_once_stream(text, auto_approve=True, cancel_event=cancel_event):
                chunks.append(token)
        except Exception as e:
            logger.error(f"Runtime stream error: {e}", exc_info=True)
            raise
        finally:
            # Always restore the original callback
            self.runtime.on_status_update = original_callback

        return chunks

    def _is_allowed(self, update: Update) -> bool:
        allowed_user = self.config.telegram_allowed_user
        is_valid = False
        
        if not allowed_user:
            # If no user is specified, allow everyone (public bot)
            is_valid = True
        else:
            # Check against username
            username = update.effective_user.username
            if username and username.lower() == allowed_user.lower().lstrip("@"):
                is_valid = True
                
            # Check against chat ID
            chat_id = str(update.effective_chat.id)
            if chat_id == allowed_user:
                is_valid = True
                
        if is_valid:
            current_chat_id = update.effective_chat.id
            if self.config.telegram_chat_id != current_chat_id:
                self.config.telegram_chat_id = current_chat_id
                from butler.config import save_config
                save_config(self.config)
                logger.info(f"Automatically saved your Chat ID ({current_chat_id}) to config for proactive reminders.")
                
        return is_valid

    # ── Telegram Commands ────────────────────────────────────────────

    async def _handle_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return
        if self._active_cancel_event and not self._active_cancel_event.is_set():
            self._active_cancel_event.set()
            await update.message.reply_text("🔪 _Mission aborted, Boss._", parse_mode="Markdown")
        else:
            await update.message.reply_text("Nothing to kill, Boss. I'm idle.")

    async def _handle_forget(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return
        try:
            if self.runtime.conversation_id:
                self.runtime.db.clear_conversation(self.runtime.conversation_id)
            self.runtime.conversation_id = self.runtime.db.new_conversation()
            await update.message.reply_text("🧹 _Memory wiped. Fresh start, Boss._", parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Error in /forget: {e}", exc_info=True)
            await update.message.reply_text(f"Failed to clear memory: {e}")

    async def _handle_voice_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return

        text = " ".join(context.args) if context.args else ""
        if not text.strip():
            await update.message.reply_text("Usage: `/voice <your question>`", parse_mode="Markdown")
            return

        chat_id = update.effective_chat.id

        if self._runtime_lock.locked():
            await update.message.reply_text("⏳ _Processing your previous request, Boss. Hold on..._", parse_mode="Markdown")

        async with self._runtime_lock:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            cancel_event = threading.Event()
            self._active_cancel_event = cancel_event
            try:
                response_chunks = await asyncio.wait_for(
                    asyncio.to_thread(self._get_runtime_response, text, chat_id, cancel_event),
                    timeout=self._RESPONSE_TIMEOUT,
                )

                full_response = "".join(response_chunks).strip() if response_chunks else "I couldn't generate a response."

                # Generate TTS audio
                import tempfile
                import os
                import edge_tts

                voice = self.config.voice or "en-IE-EmilyNeural"
                fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)

                try:
                    communicate = edge_tts.Communicate(full_response, voice)
                    await communicate.save(tmp_path)

                    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                        await context.bot.send_chat_action(chat_id=chat_id, action="upload_voice")
                        with open(tmp_path, "rb") as audio_file:
                            await update.message.reply_voice(voice=audio_file)
                    else:
                        # TTS failed — fall back to text
                        await update.message.reply_text(full_response[:4000])
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            except asyncio.TimeoutError:
                cancel_event.set()
                await update.message.reply_text(f"⏱️ Boss, that took too long (>{self._RESPONSE_TIMEOUT}s).")
            except Exception as e:
                logger.error(f"Voice command error: {e}", exc_info=True)
                await update.message.reply_text(f"Voice command failed: {e}")
            finally:
                self._active_cancel_event = None

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return
        uptime_secs = int(time.time() - self._start_time)
        hours, remainder = divmod(uptime_secs, 3600)
        minutes, secs = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {secs}s"

        lock_status = "🔴 Busy" if self._runtime_lock.locked() else "🟢 Idle"
        
        if self._chat_mode == "stealth":
            mode_str = "🤫 Stealth"
        elif self._chat_mode == "bish":
            mode_str = "💅 Bish"
        else:
            mode_str = "🗣️ Chatty"
            
        snooze_str = "Off"
        if self._snooze_until and time.time() < self._snooze_until:
            remaining = int(self._snooze_until - time.time()) // 60
            snooze_str = f"💤 {remaining}m remaining"

        status = (
            f"*BUTLER Status*\n\n"
            f"⏱ Uptime: `{uptime_str}`\n"
            f"🧠 Provider: `{self.config.provider}`\n"
            f"🤖 Model: `{self.config.model}`\n"
            f"👁 Vision: `{self.config.vision_model}`\n"
            f"📡 Status: {lock_status}\n"
            f"🎭 Mode: {mode_str}\n"
            f"💤 Snooze: {snooze_str}"
        )
        await update.message.reply_text(status, parse_mode="Markdown")

    async def _handle_retry(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return
        if not self._last_user_message:
            await update.message.reply_text("Nothing to retry, Boss.")
            return

        await update.message.reply_text(f"🔁 _Retrying: \"{self._last_user_message[:60]}\"_", parse_mode="Markdown")
        # Fake an update with the stored text and re-run through _handle_message
        update.message.text = self._last_user_message
        await self._handle_message(update, context)

    async def _handle_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return

        arg = context.args[0].lower() if context.args else None

        if arg == "stealth":
            self._chat_mode = "stealth"
            await update.message.reply_text("🤫 _Stealth mode ON. Short replies only._", parse_mode="Markdown")
        elif arg == "chatty":
            self._chat_mode = "chatty"
            await update.message.reply_text("🗣️ _Back to chatty mode, Boss._", parse_mode="Markdown")
        elif arg == "bish":
            self._chat_mode = "bish"
            await update.message.reply_text("💅 _Bish mode ON. Hey daddy._", parse_mode="Markdown")
        else:
            # Show inline keyboard MCQ
            keyboard = [
                [
                    InlineKeyboardButton("🤫 Stealth", callback_data="mode_stealth"),
                    InlineKeyboardButton("🗣️ Chatty", callback_data="mode_chatty"),
                    InlineKeyboardButton("💅 Bish", callback_data="mode_bish"),
                ]
            ]
            if self._chat_mode == "stealth":
                current = "🤫 Stealth"
            elif self._chat_mode == "bish":
                current = "💅 Bish"
            else:
                current = "🗣️ Chatty"
                
            await update.message.reply_text(
                f"Current mode: *{current}*\nSelect a mode:",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

    async def _handle_snooze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return
        try:
            hours = float(context.args[0]) if context.args else 1.0
        except (ValueError, IndexError):
            hours = 1.0

        self._snooze_until = time.time() + (hours * 3600)
        await update.message.reply_text(f"💤 _Snoozed for {hours:.0f} hour(s). I'll be quiet, Boss._", parse_mode="Markdown")

    async def _handle_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return

        from butler.paths import butler_home_dir
        from collections import deque

        log_file = butler_home_dir() / "butler.log"
        if not log_file.exists():
            await update.message.reply_text("No logs found yet, Boss.")
            return

        chat_id = update.effective_chat.id

        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                last_lines = list(deque(f, maxlen=15))
            log_text = "".join(last_lines).strip()

            if not log_text:
                await update.message.reply_text("Log file is empty, Boss.")
                return

            # Summarize via AI
            prompt = f"Summarize these system logs in 2-3 sentences. Flag any errors or warnings:\n\n{log_text}"

            if self._runtime_lock.locked():
                await update.message.reply_text("⏳ _Waiting for current task to finish before reading logs..._", parse_mode="Markdown")

            async with self._runtime_lock:
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                cancel_event = threading.Event()
                self._active_cancel_event = cancel_event
                try:
                    response_chunks = await asyncio.wait_for(
                        asyncio.to_thread(self._get_runtime_response, prompt, chat_id, cancel_event),
                        timeout=self._RESPONSE_TIMEOUT,
                    )
                    summary = "".join(response_chunks).strip() if response_chunks else "Could not summarize logs."
                    await update.message.reply_text(f"📋 *Log Summary*\n\n{summary}", parse_mode="Markdown")
                except asyncio.TimeoutError:
                    cancel_event.set()
                    await update.message.reply_text("⏱️ Log summary timed out.")
                except Exception as e:
                    logger.error(f"Log summary error: {e}", exc_info=True)
                    await update.message.reply_text(f"Failed to summarize logs: {e}")
                finally:
                    self._active_cancel_event = None
        except Exception as e:
            logger.error(f"Error reading logs: {e}", exc_info=True)
            await update.message.reply_text(f"Failed to read logs: {e}")

    async def _handle_learn(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return

        text = " ".join(context.args) if context.args else ""
        if not text.strip():
            await update.message.reply_text("Usage: `/learn <instruction>`\nExample: `/learn when I say morning brief, summarize news and weather`", parse_mode="Markdown")
            return

        chat_id = update.effective_chat.id

        if self._runtime_lock.locked():
            await update.message.reply_text("⏳ _Processing your previous request..._", parse_mode="Markdown")

        async with self._runtime_lock:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            cancel_event = threading.Event()
            self._active_cancel_event = cancel_event
            try:
                # We want the AI to extract exactly the trigger and action in JSON format.
                import json
                prompt = f"""
Extract the trigger phrase and the action from this user skill definition: "{text}"
Return ONLY a valid JSON object with keys "trigger" (the exact phrase to listen for, keep it short) and "action" (the detailed steps to perform).
Example: {{"trigger": "morning brief", "action": "Search for top 3 AI news and get weather in Noida"}}
Do not use markdown blocks, just raw JSON.
                """.strip()

                response_chunks = await asyncio.wait_for(
                    asyncio.to_thread(self._get_runtime_response, prompt, chat_id, cancel_event),
                    timeout=self._RESPONSE_TIMEOUT,
                )
                
                raw_json = "".join(response_chunks).strip()
                if raw_json.startswith("```json"):
                    raw_json = raw_json[7:-3].strip()
                elif raw_json.startswith("```"):
                    raw_json = raw_json[3:-3].strip()

                skill_data = json.loads(raw_json)
                trigger = skill_data.get("trigger", "").lower().strip()
                action = skill_data.get("action", "").strip()

                if not trigger or not action:
                    raise ValueError("Failed to extract trigger and action.")

                import uuid
                pending_id = str(uuid.uuid4())[:8]
                self._pending_skills[pending_id] = {"trigger": trigger, "action": action}

                keyboard = [
                    [
                        InlineKeyboardButton("✅ Save", callback_data=f"skill_save_{pending_id}"),
                        InlineKeyboardButton("❌ Cancel", callback_data=f"skill_cancel_{pending_id}"),
                    ]
                ]
                await update.message.reply_text(
                    f"📚 *New Skill Summary*\n\n"
                    f"*Trigger:* `{trigger}`\n"
                    f"*Action:* {action}\n\n"
                    f"Does this look correct?",
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(keyboard),
                )

            except asyncio.TimeoutError:
                cancel_event.set()
                await update.message.reply_text("⏱️ Learning timed out.")
            except Exception as e:
                logger.error(f"Learn command error: {e}", exc_info=True)
                await update.message.reply_text(f"Failed to parse skill: {e}")
            finally:
                self._active_cancel_event = None

    async def _handle_skills(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update):
            return
        
        try:
            skills = self.runtime.db.list_skills() if hasattr(self.runtime.db, "list_skills") else []
            if not skills:
                await update.message.reply_text("You haven't taught me any skills yet, Boss. Use `/learn` to create one.", parse_mode="Markdown")
                return

            for s in skills:
                keyboard = [[InlineKeyboardButton("🗑️ Delete", callback_data=f"skill_del_{s['id']}")] ]
                await update.message.reply_text(
                    f"⚡ *{s['trigger']}*\n_{s['action']}_",
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
        except Exception as e:
            logger.error(f"Error listing skills: {e}", exc_info=True)
            await update.message.reply_text("Failed to load skills.")

    # ── Proactive Messages ───────────────────────────────────────────

    def send_proactive_message(self, text: str, reminder_id: str | None = None) -> None:
        """Called by the main loop to push a message to the Telegram user."""
        if not self._loop or not self.app:
            raise RuntimeError("Telegram bot is not running.")
        if not self.config.telegram_chat_id:
            raise RuntimeError("Telegram chat_id is not configured. User must send a message first.")

        # Respect snooze
        if self._snooze_until and time.time() < self._snooze_until:
            logger.info("Proactive message suppressed (snoozed).")
            return
            
        async def _send():
            reply_markup = None
            if reminder_id:
                keyboard = [
                    [InlineKeyboardButton("💤 Snooze for 15 mins", callback_data=f"snooze_{reminder_id}")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.app.bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=text,
                reply_markup=reply_markup
            )
                
        future = asyncio.run_coroutine_threadsafe(_send(), self._loop)
        # Block until the message is sent or fails (timeout after 10s)
        future.result(timeout=10.0)

    async def _handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        data = query.data
        if data.startswith("snooze_"):
            reminder_id = data.split("snooze_")[1]
            # Snooze for 15 mins
            added_time_ms = 15 * 60 * 1000
            self.runtime.db.snooze_reminder(reminder_id, added_time_ms)
            
            # Edit the message to show it was snoozed
            new_text = query.message.text + "\n\n*(Snoozed for 15 minutes)*"
            await query.edit_message_text(text=new_text, parse_mode="Markdown")

        elif data.startswith("mode_"):
            mode = data.split("mode_")[1]
            if mode == "stealth":
                self._chat_mode = "stealth"
                await query.edit_message_text(text="Mode set: *🤫 Stealth*\n_Short replies only._", parse_mode="Markdown")
            elif mode == "chatty":
                self._chat_mode = "chatty"
                await query.edit_message_text(text="Mode set: *🗣️ Chatty*\n_Full conversational mode._", parse_mode="Markdown")
            elif mode == "bish":
                self._chat_mode = "bish"
                await query.edit_message_text(text="Mode set: *💅 Bish*\n_Flirtatious & sassy mode._", parse_mode="Markdown")

        elif data.startswith("skill_save_"):
            pending_id = data.split("skill_save_")[1]
            skill = self._pending_skills.pop(pending_id, None)
            if skill:
                if hasattr(self.runtime.db, "add_skill"):
                    self.runtime.db.add_skill(skill["trigger"], skill["action"])
                    await query.edit_message_text(text=f"✅ Skill saved!\nTrigger: `{skill['trigger']}`", parse_mode="Markdown")
                else:
                    await query.edit_message_text(text="❌ Database does not support skills.", parse_mode="Markdown")
            else:
                await query.edit_message_text(text="❌ This skill request has expired.", parse_mode="Markdown")

        elif data.startswith("skill_cancel_"):
            pending_id = data.split("skill_cancel_")[1]
            self._pending_skills.pop(pending_id, None)
            await query.edit_message_text(text="❌ Skill discarded.", parse_mode="Markdown")

        elif data.startswith("skill_del_"):
            skill_id = data.split("skill_del_")[1]
            if hasattr(self.runtime.db, "delete_skill"):
                success = self.runtime.db.delete_skill(skill_id)
                if success:
                    await query.edit_message_text(text="🗑️ Skill deleted.", parse_mode="Markdown")
                else:
                    await query.edit_message_text(text="❌ Skill not found.", parse_mode="Markdown")

    def _run_async_loop(self):
        import time as _time

        max_retries = 5
        base_delay = 5  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

                self.app = (
                    Application.builder()
                    .token(self.config.telegram_bot_token)
                    .connect_timeout(60.0)
                    .read_timeout(60.0)
                    .pool_timeout(60.0)
                    .build()
                )

                self.app.add_handler(CommandHandler("start", self._handle_start))
                self.app.add_handler(CommandHandler("kill", self._handle_kill))
                self.app.add_handler(CommandHandler("forget", self._handle_forget))
                self.app.add_handler(CommandHandler("voice", self._handle_voice_cmd))
                self.app.add_handler(CommandHandler("status", self._handle_status))
                self.app.add_handler(CommandHandler("retry", self._handle_retry))
                self.app.add_handler(CommandHandler("mode", self._handle_mode))
                self.app.add_handler(CommandHandler("snooze", self._handle_snooze))
                self.app.add_handler(CommandHandler("logs", self._handle_logs))
                self.app.add_handler(CommandHandler("learn", self._handle_learn))
                self.app.add_handler(CommandHandler("skills", self._handle_skills))
                self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
                self.app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))
                self.app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
                self.app.add_handler(CallbackQueryHandler(self._handle_callback_query))

                logger.info(f"Bot daemon connecting to Telegram (attempt {attempt}/{max_retries})...")
                return  # Clean exit, no retry needed

            except Exception as e:
                delay = min(base_delay * (2 ** (attempt - 1)), 60)
                logger.error(
                    f"Telegram daemon failed on attempt {attempt}/{max_retries}: {e}",
                    exc_info=True,
                )
                if attempt < max_retries:
                    logger.info(f"Retrying in {delay}s...")
                    _time.sleep(delay)
                else:
                    logger.critical(
                        f"Telegram daemon gave up after {max_retries} attempts. "
                        f"Last error: {e}"
                    )

    def start(self):
        if not self.config.telegram_bot_token:
            return

        self._thread = threading.Thread(target=self._run_async_loop, daemon=True, name="TelegramBotDaemon")
        self._thread.start()

    def stop(self):
        if not self._loop or not self._loop.is_running():
            return

        async def _shutdown():
            try:
                if self.app:
                    await self.app.updater.stop()
                    await self.app.stop()
                    await self.app.shutdown()
            except Exception as e:
                logger.warning(f"Shutdown warning: {e}")
            finally:
                self._loop.stop()

        asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)

        # Wait up to 3 seconds for the thread to die, then move on
        if self._thread:
            self._thread.join(timeout=3)

