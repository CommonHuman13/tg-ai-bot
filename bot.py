import os
import asyncio
import logging
import re
from collections import defaultdict, deque
from typing import Deque, Tuple
from google.genai import types
from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from google import genai


# ----------------------------
# Config (env only)
# ----------------------------
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

# Prefer правильное имя:
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY (preferred) or OPENAI_API_KEY (fallback) in environment variables.")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
MAX_TURNS = int(os.getenv("MAX_TURNS", "12"))  # turns = user+assistant pairs
PORT = int(os.getenv("PORT", "10000"))  # Render sets PORT for Web Services


SYSTEM_PROMPT = (
    "Ты полезный ассистент в Telegram. Отвечай по делу, понятным языком. "
    "Если вопрос неполный — задай уточняющий вопрос."
)

# Telegram hard limit is 4096 chars; keep margin
TG_CHUNK = 3800


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("tg-ai-bot")


# ----------------------------
# Gemini client (sync SDK -> run in thread)
# ----------------------------
client = genai.Client(api_key=GEMINI_API_KEY)


# ----------------------------
# Simple per-chat memory (in RAM)
# ----------------------------
History = Deque[Tuple[str, str]]  # (role, text)
history: defaultdict[int, History] = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))


def build_prompt(chat_id: int, user_text: str) -> str:
    lines = ["Ты ассистент в Telegram. Отвечай кратко и по делу.", ""]
    for role, text in history[chat_id]:
        lines.append(f"{role}: {text}")
    lines.append(f"User: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)



GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.6"))

def _is_quota_error(e: Exception) -> bool:
    s = str(e)
    return ("RESOURCE_EXHAUSTED" in s) or ("quota" in s.lower()) or ("429" in s)

async def call_gemini(chat_id: int, user_text: str) -> str:
    prompt = build_prompt(chat_id, user_text)

    def _sync_call() -> str:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            ),
        )
        txt = getattr(resp, "text", None)
        return (txt or "").strip()

    try:
        return await asyncio.to_thread(_sync_call)
    except Exception as e:
        if _is_quota_error(e):
            return (
                "⚠️ Уперлись в лимит Gemini (квота Free tier закончилась или слишком много запросов).\n"
                "Попробуй:\n"
                "• подождать 5–30 минут и повторить\n"
                "• либо завтра (если дневная квота)\n"
                "• либо включить Billing/другой ключ\n"
                "• либо уменьшить MAX_TURNS/макс. токены ответа."
            )
        log.exception("Gemini error: %s", e)
        return "Упс, ошибка при запросе к модели. Попробуй ещё раз."


async def send_long(message: Message, text: str) -> None:
    text = text.strip()
    if not text:
        return
    for i in range(0, len(text), TG_CHUNK):
        await message.answer(text[i:i + TG_CHUNK])


# ----------------------------
# Render health server (so Web Service sees an open port)
# ----------------------------
async def start_health_server() -> web.AppRunner:
    app = web.Application()

    async def ok(_request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app.router.add_get("/", ok)
    app.router.add_get("/healthz", ok)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
    await site.start()
    log.info("Health server started on :%s", PORT)
    return runner


# ----------------------------
# Telegram bot handlers
# ----------------------------
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def on_start(message: Message):
    await message.answer("Привет! Напиши сообщение — отвечу как ИИ 🙂\nКоманда: /reset (сбросить контекст)")


@dp.message(Command("reset"))
async def on_reset(message: Message):
    history[message.chat.id].clear()
    await message.answer("Ок, сбросил контекст. Пиши заново 🙂")


@dp.message(F.text)
async def on_text(message: Message):
    user_text = (message.text or "").strip()
    if not user_text:
        return

    chat_id = message.chat.id

    # сохраняем user
    history[chat_id].append(("User", user_text))

    thinking = await message.answer("Думаю…")
    answer = await call_gemini(chat_id, user_text)

    # сохраняем assistant
    history[chat_id].append(("Assistant", answer))

    # заменяем "Думаю…" на ответ
    try:
        await thinking.edit_text(answer[:4096])
        # если длиннее лимита — докидываем остаток
        if len(answer) > 4096:
            await send_long(message, answer[4096:])
    except Exception:
        # если edit не вышел — просто отправляем
        await send_long(message, answer)


async def main():
    log.info("Bot starting... model=%s", GEMINI_MODEL)
    await start_health_server()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())


