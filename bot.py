import os
import re
import asyncio
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from zoneinfo import ZoneInfo

from aiohttp import web

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from aiogram.exceptions import TelegramConflictError

from google import genai
from google.genai import types as genai_types


# -------------------- Logging --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("tg-ai-bot")


# -------------------- Env --------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var")

# Gemini key: основной GEMINI_API_KEY, fallback на OPENAI_API_KEY (на случай старой настройки)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY env var (or OPENAI_API_KEY fallback)")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.6"))

MAX_TURNS = int(os.getenv("MAX_TURNS", "6"))  # хранится user+assistant => *2
REQUEST_COOLDOWN_SEC = float(os.getenv("REQUEST_COOLDOWN_SEC", "1.5"))

TZ_NAME = os.getenv("TZ", "Europe/Moscow")
TZ = ZoneInfo(TZ_NAME)

PORT = int(os.getenv("PORT", "10000"))


# -------------------- Gemini client --------------------
client = genai.Client(api_key=GEMINI_API_KEY)


# -------------------- Memory (history) --------------------
# history[chat_id] = deque([(role, text), ...]) role: "User"/"Assistant"
history = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))

# simple cooldown per chat
last_request_ts = defaultdict(lambda: 0.0)


# -------------------- Reminders --------------------
reminders = defaultdict(dict)  # reminders[chat_id][rem_id] = {"at": dt, "text": str, "task": asyncio.Task}
rem_counter = defaultdict(lambda: 0)


def _now() -> datetime:
    return datetime.now(TZ)


def _parse_time_hhmm(text: str):
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return hh, mm
    return None


def parse_reminder(text: str):
    """
    Поддержка:
    - "напомни завтра распечатать документы"
    - "напомни завтра в 18:30 распечатать документы"
    - "напомни в 18:30 распечатать документы"
    - "напомни через 20 минут сделать чай"
    - "напомни через 2 часа позвонить"
    """
    t = text.strip()
    if not re.search(r"\bнапомни\b", t, re.IGNORECASE):
        return None

    # вытащим "тело" после "напомни (мне)?"
    body = re.split(r"\bнапомни(?:\s+мне)?\b", t, flags=re.IGNORECASE, maxsplit=1)
    if len(body) < 2:
        return None
    body = body[1].strip()
    if not body:
        return None

    now = _now()

    # "через N ..."
    m = re.search(r"\bчерез\s+(\d+)\s*(минут|мин|час|часа|часов|день|дня|дней)\b", body, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if "мин" in unit:
            dt = now + timedelta(minutes=n)
        elif "час" in unit:
            dt = now + timedelta(hours=n)
        else:
            dt = now + timedelta(days=n)

        task_text = re.sub(r"\bчерез\s+\d+\s*(?:минут|мин|час|часа|часов|день|дня|дней)\b", "", body, flags=re.IGNORECASE).strip()
        task_text = task_text.lstrip(",.:-— ").strip()
        if not task_text:
            task_text = "напоминание"
        return dt, task_text

    # завтра / сегодня + (в HH:MM)
    hhmm = _parse_time_hhmm(body)

    if re.search(r"\bзавтра\b", body, re.IGNORECASE):
        base = (now + timedelta(days=1)).date()
        if hhmm:
            hh, mm = hhmm
            dt = datetime(base.year, base.month, base.day, hh, mm, tzinfo=TZ)
            task_text = re.sub(r"\bзавтра\b", "", body, flags=re.IGNORECASE)
            task_text = re.sub(r"\bв\s*\d{1,2}:\d{2}\b", "", task_text, flags=re.IGNORECASE).strip()
        else:
            # если не указано время — ставим 09:00
            dt = datetime(base.year, base.month, base.day, 9, 0, tzinfo=TZ)
            task_text = re.sub(r"\bзавтра\b", "", body, flags=re.IGNORECASE).strip()

        task_text = task_text.lstrip(",.:-— ").strip() or "напоминание"
        return dt, task_text

    if re.search(r"\bсегодня\b", body, re.IGNORECASE):
        base = now.date()
        if hhmm:
            hh, mm = hhmm
            dt = datetime(base.year, base.month, base.day, hh, mm, tzinfo=TZ)
            # если время уже прошло — на завтра
            if dt <= now:
                dt = dt + timedelta(days=1)
            task_text = re.sub(r"\bсегодня\b", "", body, flags=re.IGNORECASE)
            task_text = re.sub(r"\bв\s*\d{1,2}:\d{2}\b", "", task_text, flags=re.IGNORECASE).strip()
        else:
            # если без времени — через 1 час
            dt = now + timedelta(hours=1)
            task_text = re.sub(r"\bсегодня\b", "", body, flags=re.IGNORECASE).strip()

        task_text = task_text.lstrip(",.:-— ").strip() or "напоминание"
        return dt, task_text

    # "в HH:MM ..."
    if hhmm:
        hh, mm = hhmm
        base = now.date()
        dt = datetime(base.year, base.month, base.day, hh, mm, tzinfo=TZ)
        if dt <= now:
            dt = dt + timedelta(days=1)
        task_text = re.sub(r"\bв\s*\d{1,2}:\d{2}\b", "", body, flags=re.IGNORECASE).strip()
        task_text = task_text.lstrip(",.:-— ").strip() or "напоминание"
        return dt, task_text

    # если есть "напомни ..." но времени нет — не создаём
    return None


async def reminder_job(bot: Bot, chat_id: int, rem_id: int, when_dt: datetime, text: str):
    try:
        delay = (when_dt - _now()).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        await bot.send_message(chat_id, f"⏰ Напоминание: {text}")
    except Exception as e:
        log.exception("Reminder job error: %s", e)
    finally:
        # очистка
        reminders[chat_id].pop(rem_id, None)


def build_prompt(chat_id: int, user_text: str) -> str:
    # короткая системка (экономит квоту)
    lines = [
        "Ты полезный ассистент в Telegram. Отвечай кратко, по делу, без воды.",
        "",
    ]
    for role, txt in history[chat_id]:
        lines.append(f"{role}: {txt}")
    lines.append(f"User: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _is_quota_error(e: Exception) -> bool:
    s = str(e)
    return ("RESOURCE_EXHAUSTED" in s) or ("quota" in s.lower()) or ("429" in s)


async def call_gemini(chat_id: int, user_text: str) -> str:
    prompt = build_prompt(chat_id, user_text)

    def _sync_call() -> str:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
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
                "⚠️ Лимит Gemini Free tier закончился (квота/429).\n"
                "Попробуй позже (5–30 минут) или завтра, либо поменяй ключ/включи Billing.\n"
                "Чтобы квота жила дольше — уменьши MAX_TURNS и GEMINI_MAX_OUTPUT_TOKENS."
            )
        log.exception("Gemini error: %s", e)
        return "Упс, ошибка при запросе к модели. Попробуй ещё раз."


# -------------------- Telegram bot --------------------
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def on_start(message: types.Message):
    await message.answer(
        "Привет! Напиши сообщение — отвечу как ИИ 🙂\n"
        "Команды:\n"
        "/reset — сбросить контекст\n"
        "/reminders — список напоминаний\n"
        "/cancel <id> — отменить напоминание\n\n"
        "Можно так: «напомни завтра в 18:30 распечатать документы»"
    )


@dp.message(Command("help"))
async def on_help(message: types.Message):
    await on_start(message)


@dp.message(Command("reset"))
async def on_reset(message: types.Message):
    history[message.chat.id].clear()
    await message.answer("Ок, контекст сброшен. Пиши заново 🙂")


@dp.message(Command("reminders"))
async def on_reminders(message: types.Message):
    chat_id = message.chat.id
    if not reminders[chat_id]:
        await message.answer("Напоминаний нет.")
        return
    items = []
    for rid, data in sorted(reminders[chat_id].items(), key=lambda x: x[0]):
        when_dt = data["at"]
        items.append(f"{rid}) {when_dt.strftime('%Y-%m-%d %H:%M')} — {data['text']}")
    await message.answer("Твои напоминания:\n" + "\n".join(items))


@dp.message(Command("cancel"))
async def on_cancel(message: types.Message):
    chat_id = message.chat.id
    parts = (message.text or "").split()
    if len(parts) < 2 or not parts[1].isdigit():
        await message.answer("Используй: /cancel <id>")
        return
    rid = int(parts[1])
    data = reminders[chat_id].get(rid)
    if not data:
        await message.answer("Не нашёл такое напоминание.")
        return
    task = data.get("task")
    if task:
        task.cancel()
    reminders[chat_id].pop(rid, None)
    await message.answer(f"Ок, отменил напоминание #{rid}.")


@dp.message()
async def on_text(message: types.Message):
    chat_id = message.chat.id
    text = (message.text or "").strip()
    if not text:
        return

    # 1) Напоминания (если распознали — не дергаем ИИ)
    parsed = parse_reminder(text)
    if parsed:
        when_dt, rem_text = parsed
        rem_counter[chat_id] += 1
        rid = rem_counter[chat_id]

        task = asyncio.create_task(reminder_job(bot, chat_id, rid, when_dt, rem_text))
        reminders[chat_id][rid] = {"at": when_dt, "text": rem_text, "task": task}

        await message.answer(
            f"✅ Запомнил. Напомню #{rid}: {when_dt.strftime('%Y-%m-%d %H:%M')} — {rem_text}"
        )
        return

    # 2) Cooldown (экономит квоту)
    now_ts = asyncio.get_running_loop().time()
    if now_ts - last_request_ts[chat_id] < REQUEST_COOLDOWN_SEC:
        await message.answer("Подожди секундочку 🙂")
        return
    last_request_ts[chat_id] = now_ts

    # 3) Ответ ИИ
    thinking = await message.answer("Думаю…")

    # добавляем user в историю
    history[chat_id].append(("User", text))

    answer = await call_gemini(chat_id, text)
    answer = answer[:4000] if answer else "…"

    # добавляем assistant в историю
    history[chat_id].append(("Assistant", answer))

    await thinking.edit_text(answer)


# -------------------- Health server (Render) --------------------
async def start_health_server():
    async def health(_):
        return web.Response(text="ok")

    app = web.Application()
    app.add_routes([web.get("/", health), web.get("/healthz", health)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log.info("Health server started on :%s", PORT)


async def main():
    # на всякий случай убираем webhook, иначе polling может конфликтовать
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass

    await start_health_server()

    # ВАЖНО: если где-то запущен второй экземпляр (локально или ещё один хостинг),
    # Telegram будет кидать 409 Conflict: terminated by other getUpdates request.
    while True:
        try:
            log.info("Bot starting... model=%s tz=%s", GEMINI_MODEL, TZ_NAME)
            await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
        except TelegramConflictError:
            log.error(
                "409 Conflict: другой экземпляр бота уже делает getUpdates.\n"
                "Останови бота на ПК/другом хостинге или оставь только один Render-сервис.\n"
                "Повторю попытку через 30 секунд."
            )
            await asyncio.sleep(30)
        except Exception as e:
            log.exception("Polling crashed: %s", e)
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())

