import os
import re
import json
import time
import heapq
import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict, deque
from typing import Optional, List, Tuple

import dateparser
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.exceptions import TelegramBadRequest

from aiohttp import web

from google import genai
from google.genai import types
from google.genai.errors import ClientError


# =========================
# CONFIG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("tg-ai-bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var")

# Gemini ключ лучше хранить в GEMINI_API_KEY, но поддержим и старые названия:
GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
).strip()
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY / OPENAI_API_KEY) env var")

# Модель: ставим стабильный дефолт. Пример использования есть в официальных доках. :contentReference[oaicite:4]{index=4}
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

# Если ловил 404 на v1beta — используй v1 (через HttpOptions). :contentReference[oaicite:5]{index=5}
GEMINI_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1").strip()

TIMEZONE = os.getenv("TIMEZONE", "Europe/Moscow").strip()
TZ = ZoneInfo(TIMEZONE)

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты полезный, дружелюбный ассистент в Telegram. Отвечай кратко и по делу, если не просят иначе.",
).strip()

# История (сколько последних «сообщений» хранить на чат)
MAX_TURNS = int(os.getenv("MAX_TURNS", "12"))  # 12 пар = 24 сообщений
HISTORY_MAXLEN = MAX_TURNS * 2

# Антиспам/нагрузка
COOLDOWN_SEC = float(os.getenv("COOLDOWN_SEC", "1.2"))  # минимальная пауза между запросами от одного юзера
MODEL_TIMEOUT_SEC = float(os.getenv("MODEL_TIMEOUT_SEC", "40"))

# Ответ Telegram ограничен ~4096
TG_LIMIT = 3900

# Напоминания: файл (частичная «живучесть» при рестарте)
REMINDERS_FILE = os.getenv("REMINDERS_FILE", "reminders.json").strip()

# Если хочешь, чтобы бот отвечал только тебе: поставь свой user_id (можно узнать командой /myid)
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID", "").strip()
ALLOWED_USER_ID_INT = int(ALLOWED_USER_ID) if ALLOWED_USER_ID.isdigit() else None


# =========================
# GEMINI CLIENT
# =========================
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(api_version=GEMINI_API_VERSION),
)


# =========================
# STATE
# =========================
# chat_id -> deque[{"role": "user"|"model", "text": "..."}]
history = defaultdict(lambda: deque(maxlen=HISTORY_MAXLEN))

# user_id -> last_ts
last_request_ts = defaultdict(lambda: 0.0)

# chat_id -> lock (чтобы не было гонок, если юзер спамит сообщениями)
chat_locks = defaultdict(asyncio.Lock)


# =========================
# REMINDERS
# =========================
@dataclass
class Reminder:
    rid: str
    chat_id: int
    user_id: int
    when_ts: float  # unix timestamp
    text: str

# min-heap by when_ts
reminder_heap: List[Tuple[float, str]] = []  # (when_ts, rid)
reminders: dict[str, Reminder] = {}


def _now_ts() -> float:
    return time.time()


def _dt_to_ts(dt: datetime) -> float:
    return dt.timestamp()


def _ts_to_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, TZ)


def save_reminders() -> None:
    try:
        payload = [asdict(r) for r in reminders.values()]
        with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("Failed to save reminders")


def load_reminders() -> None:
    if not os.path.exists(REMINDERS_FILE):
        return
    try:
        with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for item in payload:
            r = Reminder(**item)
            reminders[r.rid] = r
            heapq.heappush(reminder_heap, (r.when_ts, r.rid))
        log.info("Loaded reminders: %d", len(reminders))
    except Exception:
        log.exception("Failed to load reminders")


def gen_rid() -> str:
    # достаточно для одного процесса
    return f"r{int(_now_ts()*1000)}_{os.getpid()}"


def split_text(text: str, limit: int = TG_LIMIT) -> List[str]:
    text = text.strip()
    if not text:
        return ["(пустой ответ)"]
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < 200:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    parts.append(text)
    return parts


def is_time_explicit(s: str) -> bool:
    # грубо: "в 12", "12:30", "19.45"
    return bool(re.search(r"\b\d{1,2}([:.]\d{2})?\b", s))


def parse_reminder(text: str) -> Optional[Tuple[datetime, str]]:
    """
    Понимает фразы типа:
    - "напомни завтра распечатать документы"
    - "напомни мне завтра в 18:30 позвонить маме"
    - "напомни через 2 часа проверить почту"
    """
    t = text.strip()

    if not re.search(r"^\s*напомни", t, flags=re.IGNORECASE):
        return None

    # пытаемся вытащить дату/время из всей строки
    settings = {
        "TIMEZONE": TIMEZONE,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": datetime.now(TZ),
    }

    # dateparser хорошо понимает RU
    # 1) попробуем найти datetime в строке после "напомни"
    after = re.sub(r"^\s*напомни(\s+мне)?\s*", "", t, flags=re.IGNORECASE).strip()
    if not after:
        return None

    # эвристика: разделим "когда" и "что" — по первому глаголу/тексту может не сработать,
    # поэтому: сначала парсим datetime прямо из after.
    dt = dateparser.parse(after, languages=["ru"], settings=settings)

    # Если dt не получился — попробуем парсить первые 60 символов как "когда"
    if dt is None:
        dt = dateparser.parse(after[:60], languages=["ru"], settings=settings)
        if dt is None:
            return None

    # Если время не указано явно — поставим дефолт 10:00
    if not is_time_explicit(after):
        dt = dt.replace(hour=10, minute=0, second=0, microsecond=0)

    # Что напомнить: пытаемся убрать «датовую» часть простым способом:
    # если пользователь писал "завтра ...", "через ...", "в 19:00 ..." — часто это в начале.
    # Берём "что" как текст после найденной даты (эвристика по ключевым словам).
    what = after

    # убираем частые маркеры времени в начале
    what = re.sub(r"^(завтра|послезавтра|сегодня)\b", "", what, flags=re.IGNORECASE).strip()
    what = re.sub(r"^через\s+\d+\s*(минут|мин|час|часа|часов|день|дня|дней)\b", "", what, flags=re.IGNORECASE).strip()
    what = re.sub(r"^в\s+\d{1,2}([:.]\d{2})?\b", "", what, flags=re.IGNORECASE).strip()

    # если так и осталось пусто — попросим уточнить
    if not what:
        what = "Напоминание"

    return dt, what


async def reminder_loop(bot: Bot) -> None:
    while True:
        try:
            if not reminder_heap:
                await asyncio.sleep(1.0)
                continue

            when_ts, rid = reminder_heap[0]
            now = _now_ts()

            if when_ts > now:
                await asyncio.sleep(min(30.0, when_ts - now))
                continue

            heapq.heappop(reminder_heap)
            r = reminders.pop(rid, None)
            save_reminders()
            if not r:
                continue

            dt = _ts_to_dt(r.when_ts).strftime("%d.%m.%Y %H:%M")
            await bot.send_message(r.chat_id, f"⏰ Напоминание ({dt}): {r.text}")

        except Exception:
            log.exception("Reminder loop error")
            await asyncio.sleep(2.0)


# =========================
# HEALTH SERVER (для Render Web Service)
# =========================
async def start_health_server() -> None:
    """
    Если деплоишь как Render Web Service, он ожидает, что процесс откроет порт ($PORT),
    иначе пишет "No open ports detected...".
    Делаем крошечный HTTP сервер.
    """
    port = os.getenv("PORT")
    if not port:
        return
    port_i = int(port)

    app = web.Application()

    async def health(_):
        return web.json_response({"ok": True})

    app.router.add_get("/", health)
    app.router.add_get("/healthz", health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port_i)
    await site.start()
    log.info("Health server started on :%d", port_i)


# =========================
# AI CALL
# =========================
def build_contents(chat_id: int, user_text: str) -> List[types.Content]:
    contents: List[types.Content] = []

    for m in history[chat_id]:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(m["text"])]))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(user_text)]))
    return contents


async def call_gemini(chat_id: int, user_text: str) -> str:
    contents = build_contents(chat_id, user_text)

    config = types.GenerateContentConfig(
        system_instruction=[SYSTEM_PROMPT],
        temperature=0.6,
        max_output_tokens=1024,
    )

    # 1) пробуем выбранную модель
    try_models = [GEMINI_MODEL, "gemini-2.0-flash", "gemini-1.5-flash-001"]

    last_err = None
    for model_name in try_models:
        try:
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                ),
                timeout=MODEL_TIMEOUT_SEC,
            )
            # Обычно есть resp.text
            text = getattr(resp, "text", None)
            if not text:
                # fallback: попытка достать вручную
                try:
                    text = resp.candidates[0].content.parts[0].text
                except Exception:
                    text = ""
            text = (text or "").strip()
            if text:
                return text
            return "Пустой ответ от модели. Попробуй переформулировать."

        except ClientError as e:
            last_err = e
            # 404 по модели — пробуем следующий вариант
            log.warning("Model failed (%s): %s", model_name, str(e))
            continue
        except asyncio.TimeoutError as e:
            last_err = e
            log.warning("Model timeout (%s)", model_name)
            continue
        except Exception as e:
            last_err = e
            log.exception("Model error (%s)", model_name)
            continue

    return f"Не смог получить ответ от модели 😕 (ошибка: {last_err})"


# =========================
# TELEGRAM BOT
# =========================
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


def allowed(message: Message) -> bool:
    if ALLOWED_USER_ID_INT is None:
        return True
    return message.from_user and message.from_user.id == ALLOWED_USER_ID_INT


@dp.message(Command("start"))
async def cmd_start(message: Message):
    if not allowed(message):
        return
    await message.answer(
        "Привет! Напиши сообщение — отвечу как ИИ 🙂\n"
        "Команды:\n"
        "• /reset — сбросить контекст\n"
        "• /myid — показать твой user_id\n"
        "• /remind <когда> <что> — напоминание (или просто: «напомни завтра …»)\n"
        "• /reminds — список напоминаний\n"
        "• /delremind <id> — удалить напоминание"
    )


@dp.message(Command("myid"))
async def cmd_myid(message: Message):
    if not allowed(message):
        return
    uid = message.from_user.id if message.from_user else "unknown"
    await message.answer(f"Твой user_id: <code>{uid}</code>")


@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    if not allowed(message):
        return
    history[message.chat.id].clear()
    await message.answer("Ок, сбросил контекст. Пиши заново 🙂")


@dp.message(Command("reminds"))
async def cmd_reminds(message: Message):
    if not allowed(message):
        return
    uid = message.from_user.id if message.from_user else 0
    user_items = [r for r in reminders.values() if r.user_id == uid and r.chat_id == message.chat.id]
    if not user_items:
        await message.answer("У тебя нет активных напоминаний.")
        return

    lines = []
    for r in sorted(user_items, key=lambda x: x.when_ts):
        dt = _ts_to_dt(r.when_ts).strftime("%d.%m %H:%M")
        lines.append(f"• <code>{r.rid}</code> — {dt} — {r.text}")
    await message.answer("Твои напоминания:\n" + "\n".join(lines))


@dp.message(Command("delremind"))
async def cmd_delremind(message: Message):
    if not allowed(message):
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Использование: /delremind <id>")
        return
    rid = parts[1].strip()
    r = reminders.get(rid)
    if not r:
        await message.answer("Не нашёл такое напоминание.")
        return
    if message.from_user and r.user_id != message.from_user.id:
        await message.answer("Это не твоё напоминание 🙂")
        return

    reminders.pop(rid, None)
    # heap чистить лениво не будем — loop сам пропустит отсутствующий rid
    save_reminders()
    await message.answer("Удалил ✅")


@dp.message(Command("remind"))
async def cmd_remind(message: Message):
    if not allowed(message):
        return
    text = (message.text or "").strip()
    arg = text.split(maxsplit=1)
    if len(arg) < 2:
        await message.answer("Пример: /remind завтра в 18:30 распечатать документы")
        return

    parsed = parse_reminder("напомни " + arg[1])
    if not parsed:
        await message.answer("Не понял когда напомнить. Пример: /remind завтра в 18:30 распечатать документы")
        return

    dt, what = parsed
    rid = gen_rid()
    r = Reminder(
        rid=rid,
        chat_id=message.chat.id,
        user_id=message.from_user.id if message.from_user else 0,
        when_ts=_dt_to_ts(dt),
        text=what,
    )
    reminders[rid] = r
    heapq.heappush(reminder_heap, (r.when_ts, rid))
    save_reminders()

    await message.answer(f"Ок! Поставил напоминание ✅\nID: <code>{rid}</code>\nКогда: {_ts_to_dt(r.when_ts).strftime('%d.%m.%Y %H:%M')}\nЧто: {what}")


@dp.message(F.text)
async def on_text(message: Message):
    if not allowed(message):
        return

    uid = message.from_user.id if message.from_user else 0
    now = _now_ts()
    if now - last_request_ts[uid] < COOLDOWN_SEC:
        return
    last_request_ts[uid] = now

    text = (message.text or "").strip()
    if not text:
        return

    # 1) Натуральные напоминания без команды
    parsed = parse_reminder(text)
    if parsed:
        dt, what = parsed
        rid = gen_rid()
        r = Reminder(
            rid=rid,
            chat_id=message.chat.id,
            user_id=uid,
            when_ts=_dt_to_ts(dt),
            text=what,
        )
        reminders[rid] = r
        heapq.heappush(reminder_heap, (r.when_ts, rid))
        save_reminders()

        await message.answer(
            f"Ок! Напомню ✅\nID: <code>{rid}</code>\n"
            f"Когда: {_ts_to_dt(r.when_ts).strftime('%d.%m.%Y %H:%M')}\n"
            f"Что: {what}"
        )
        return

    # 2) AI ответ
    async with chat_locks[message.chat.id]:
        thinking = await message.answer("🤔 Думаю...")

        try:
            answer = await call_gemini(message.chat.id, text)

            # обновляем историю только после успешного ответа
            history[message.chat.id].append({"role": "user", "text": text})
            history[message.chat.id].append({"role": "model", "text": answer})

            parts = split_text(answer)
            # первая часть — редактируем "Думаю..."
            try:
                await thinking.edit_text(parts[0])
            except TelegramBadRequest:
                # если нельзя отредактировать — просто отправим
                await thinking.delete()
                await message.answer(parts[0])

            # остальные части — отдельными сообщениями
            for p in parts[1:]:
                await message.answer(p)

        except Exception:
            log.exception("Handler error")
            try:
                await thinking.edit_text("Сорян, что-то сломалось 😕 Попробуй ещё раз.")
            except Exception:
                pass


async def main():
    load_reminders()
    asyncio.create_task(reminder_loop(bot))
    asyncio.create_task(start_health_server())

    log.info("Bot starting... model=%s api_version=%s tz=%s", GEMINI_MODEL, GEMINI_API_VERSION, TIMEZONE)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
