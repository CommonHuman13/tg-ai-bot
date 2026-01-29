import os
import re
import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

from openai import OpenAI

import dateparser
from dateparser.search import search_dates

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tg-ai-bot")

# ===== ENV =====
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # хранить в env, не в коде :contentReference[oaicite:1]{index=1}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

PUBLIC_URL = os.environ["PUBLIC_URL"].rstrip("/")
WEBHOOK_PATH = os.environ.get("WEBHOOK_PATH", "/tg-webhook")
WEBHOOK_URL = f"{PUBLIC_URL}{WEBHOOK_PATH}"

PORT = int(os.getenv("PORT", "10000"))
TZ = os.getenv("TZ", "Europe/Moscow")

# ===== OpenAI client =====
# SDK читает ключ из env, но мы явно задаём для ясности
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== Bot/Dispatcher =====
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# ===== Memory =====
MAX_TURNS = int(os.getenv("MAX_TURNS", "12"))  # 12 пар user/assistant
history = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))

# ===== Concurrency limit (чтобы не убить лимиты) =====
llm_sem = asyncio.Semaphore(int(os.getenv("LLM_CONCURRENCY", "2")))

SYSTEM_PROMPT = (
    "Ты полезный и дружелюбный ассистент в Telegram. "
    "Отвечай кратко и по делу. Если не уверен — скажи об этом."
)

# ===== Helpers =====
def build_prompt(chat_id: int, user_text: str) -> str:
    """Надёжно и просто: собираем историю в один текстовый prompt."""
    parts = [f"System: {SYSTEM_PROMPT}"]
    for role, text in history[chat_id]:
        if role == "user":
            parts.append(f"User: {text}")
        else:
            parts.append(f"Assistant: {text}")
    parts.append(f"User: {user_text}")
    parts.append("Assistant:")
    return "\n".join(parts)

def split_telegram(text: str, chunk_size: int = 3800):
    text = text.strip()
    if len(text) <= 4096:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]
    return chunks

def try_parse_reminder_ru(text: str):
    """
    Пытаемся распознать напоминание из обычной фразы:
    "напомни завтра распечатать документы"
    Возвращает (when_dt, reminder_text) или None.
    """
    t = text.strip()
    if not re.match(r"^(напомни|напомнить)\b", t, flags=re.I):
        return None

    # ищем дату/время в тексте
    matches = search_dates(
        t,
        languages=["ru"],
        settings={
            "TIMEZONE": TZ,
            "RETURN_AS_TIMEZONE_AWARE": False,  # проще для APScheduler
            "PREFER_DATES_FROM": "future",
        },
    )
    if not matches:
        return None

    date_text, when = matches[0]
    if not when:
        return None

    # если распознало "завтра" как 00:00 — ставим дефолт 10:00
    if when.hour == 0 and when.minute == 0:
        when = when.replace(hour=10, minute=0)

    # на всякий: если вдруг получилось прошлое — сдвинем на +1 день
    if when < datetime.now():
        when = when + timedelta(days=1)

    # вырезаем "напомни( мне)?", и кусок даты
    rem_text = re.sub(r"^(напомни|напомнить)(\s+мне)?\s*", "", t, flags=re.I).strip()
    rem_text = rem_text.replace(date_text, "").strip(" ,.-")
    if not rem_text:
        rem_text = "напоминание"

    return when, rem_text

async def call_openai(chat_id: int, user_text: str) -> str:
    prompt = build_prompt(chat_id, user_text)

    async with llm_sem:
        # OpenAI quickstart: client.responses.create(...), response.output_text :contentReference[oaicite:2]{index=2}
        resp = await asyncio.to_thread(
            client.responses.create,
            model=OPENAI_MODEL,
            input=prompt,
        )
    out = (resp.output_text or "").strip()
    return out if out else "Не получилось сгенерировать ответ. Попробуй переформулировать."

# ===== Reminders =====
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

scheduler = AsyncIOScheduler(timezone=TZ)

def schedule_reminder(chat_id: int, when: datetime, text: str):
    job_id = f"rem_{chat_id}_{int(when.timestamp())}"

    async def _send():
        try:
            await bot.send_message(chat_id, f"⏰ Напоминание: {text}")
        except Exception:
            log.exception("Failed to send reminder")

    scheduler.add_job(lambda: asyncio.create_task(_send()), trigger=DateTrigger(run_date=when), id=job_id, replace_existing=True)
    return job_id

# ===== Handlers =====
@dp.message(CommandStart())
async def on_start(message: Message):
    await message.answer(
        "Привет! Напиши сообщение — отвечу как ИИ 🙂\n"
        "Команды:\n"
        "/reset — сбросить контекст\n"
        "/model — показать модель\n"
        "Можно просто написать: «напомни завтра распечатать документы»"
    )

@dp.message(Command("model"))
async def on_model(message: Message):
    await message.answer(f"Модель: {OPENAI_MODEL}")

@dp.message(Command("reset"))
async def on_reset(message: Message):
    history[message.chat.id].clear()
    await message.answer("Ок, контекст сброшен 👌")

@dp.message(F.text)
async def on_text(message: Message):
    chat_id = message.chat.id
    text = (message.text or "").strip()
    if not text:
        return

    # 1) Напоминалка (из обычной фразы)
    parsed = try_parse_reminder_ru(text)
    if parsed:
        when, reminder_text = parsed
        schedule_reminder(chat_id, when, reminder_text)
        await message.answer(f"Записал ✅\nНапомню: {when.strftime('%Y-%m-%d %H:%M')} — {reminder_text}")
        return

    # 2) Обычный вопрос-ответ
    # пишем "печатает..."
    try:
        await bot.send_chat_action(chat_id, action="typing")
    except Exception:
        pass

    # добавляем пользователя в историю
    history[chat_id].append(("user", text))

    try:
        answer = await call_openai(chat_id, text)
    except Exception as e:
        # 429/квоты/прочее — покажем аккуратно
        log.exception("OpenAI error")
        answer = f"Ошибка при запросе к модели: {type(e).__name__}. Попробуй позже."

    # добавляем ассистента в историю
    history[chat_id].append(("assistant", answer))

    for chunk in split_telegram(answer):
        await message.answer(chunk)

# ===== Webhook app =====
async def on_startup(app: web.Application):
    scheduler.start()
    # важное: выставляем webhook и сбрасываем старые апдейты
    await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)
    log.info("Webhook set to %s", WEBHOOK_URL)

async def on_shutdown(app: web.Application):
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass
    scheduler.shutdown(wait=False)

def main():
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    # healthcheck
    async def health(_):
        return web.Response(text="ok")

    app.router.add_get("/", health)

    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)

    web.run_app(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()

