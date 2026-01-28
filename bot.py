import os
import re
import json
import time
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from typing import Optional, Tuple, Dict, Deque, Any, List

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command

from zoneinfo import ZoneInfo

import aiosqlite
from google import genai


# ======================
# Config
# ======================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Важно: Render чаще всего в UTC. Поставь Europe/Moscow или свой.
BOT_TIMEZONE = os.getenv("BOT_TIMEZONE", "Europe/Moscow")
TZ = ZoneInfo(BOT_TIMEZONE)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

MAX_TURNS = int(os.getenv("MAX_TURNS", "24"))          # сколько “реплик” (user+assistant) хранить
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "12000"))  # чтобы не улетать в лимиты
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "4000"))     # лимит Telegram на одно сообщение

# Антифлуд (на пользователя)
USER_COOLDOWN_SEC = float(os.getenv("USER_COOLDOWN_SEC", "1.2"))

# SQLite файл (на Render без диска может сбрасываться при redeploy — это нормально для free)
DB_PATH = os.getenv("DB_PATH", "bot.db")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты полезный ассистент в Telegram. Отвечай кратко, по делу, дружелюбно. "
    "Если пользователь просит напоминание, помогай сформулировать и подтверждай."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("tg-ai-bot")


# ======================
# Helpers: memory
# ======================
HistoryItem = Dict[str, str]  # {"role": "user"/"assistant", "content": "..."}

history: Dict[int, Deque[HistoryItem]] = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))
chat_locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

# антифлуд
last_user_call: Dict[int, float] = {}


def build_prompt(chat_id: int) -> str:
    """
    Собираем историю в один prompt (Gemini принимает plain text).
    """
    lines: List[str] = [SYSTEM_PROMPT, ""]
    for item in history[chat_id]:
        role = "Пользователь" if item["role"] == "user" else "Ассистент"
        lines.append(f"{role}: {item['content']}")
    lines.append("Ассистент:")
    prompt = "\n".join(lines)

    # подрезаем, если слишком большой
    if len(prompt) > MAX_PROMPT_CHARS:
        # режем старые сообщения, пока не влезем
        while len(prompt) > MAX_PROMPT_CHARS and len(history[chat_id]) > 2:
            history[chat_id].popleft()
            prompt = "\n".join([SYSTEM_PROMPT, ""] + [
                f"{'Пользователь' if i['role']=='user' else 'Ассистент'}: {i['content']}"
                for i in history[chat_id]
            ] + ["Ассистент:"])
    return prompt


def split_text(s: str, chunk: int = MAX_REPLY_CHARS) -> List[str]:
    s = (s or "").strip()
    if not s:
        return ["(пустой ответ)"]
    parts = []
    while len(s) > chunk:
        parts.append(s[:chunk])
        s = s[chunk:]
    parts.append(s)
    return parts


# ======================
# Helpers: reminders
# ======================
@dataclass
class Reminder:
    id: int
    chat_id: int
    due_utc: int  # unix seconds UTC
    text: str
    created_utc: int


class ReminderStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._inited = False

    async def init(self):
        if self._inited:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    due_utc INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    created_utc INTEGER NOT NULL
                )
            """)
            await db.commit()
        self._inited = True

    async def add(self, chat_id: int, due_utc: int, text: str) -> int:
        await self.init()
        created = int(time.time())
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "INSERT INTO reminders(chat_id, due_utc, text, created_utc) VALUES (?, ?, ?, ?)",
                (chat_id, due_utc, text, created)
            )
            await db.commit()
            return int(cur.lastrowid)

    async def delete(self, reminder_id: int, chat_id: int) -> bool:
        await self.init()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "DELETE FROM reminders WHERE id = ? AND chat_id = ?",
                (reminder_id, chat_id)
            )
            await db.commit()
            return cur.rowcount > 0

    async def list_for_chat(self, chat_id: int) -> List[Reminder]:
        await self.init()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT id, chat_id, due_utc, text, created_utc FROM reminders WHERE chat_id = ? ORDER BY due_utc ASC",
                (chat_id,)
            )
            rows = await cur.fetchall()
        return [Reminder(*row) for row in rows]

    async def due_after_now(self) -> List[Reminder]:
        await self.init()
        now = int(time.time())
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT id, chat_id, due_utc, text, created_utc FROM reminders WHERE due_utc >= ? ORDER BY due_utc ASC",
                (now,)
            )
            rows = await cur.fetchall()
        return [Reminder(*row) for row in rows]


store = ReminderStore(DB_PATH)
scheduled_tasks: Dict[int, asyncio.Task] = {}  # reminder_id -> task


def parse_reminder_ru(text: str, now_local: datetime) -> Optional[Tuple[datetime, str]]:
    """
    Простейший разбор русских фраз:
    - "напомни завтра распечатать документы"
    - "напомни завтра в 10:30 распечатать документы"
    - "напомни через 15 минут выпить воды"
    - "напомни через 2 часа позвонить"
    Возвращает (due_local_datetime, reminder_text)
    """

    t = text.strip()

    # вытащим "напомни мне" / "напомни"
    m = re.match(r"(?i)^\s*напомни(?:\s+мне)?\s+(.*)$", t)
    if not m:
        return None

    rest = m.group(1).strip()

    # через N минут/часов/дней
    m2 = re.match(r"(?i)^через\s+(\d+)\s*(минут|мин|минуты|минута|час|часа|часов|день|дня|дней)\s+(.*)$", rest)
    if m2:
        n = int(m2.group(1))
        unit = m2.group(2).lower()
        msg = m2.group(3).strip()
        delta = None
        if "мин" in unit:
            delta = timedelta(minutes=n)
        elif "час" in unit:
            delta = timedelta(hours=n)
        elif "ден" in unit or "дн" in unit:
            delta = timedelta(days=n)

        if delta is None or not msg:
            return None

        return (now_local + delta, msg)

    # завтра/сегодня/послезавтра (+ время)
    day_shift = None
    if re.search(r"(?i)\bпослезавтра\b", rest):
        day_shift = 2
        rest = re.sub(r"(?i)\bпослезавтра\b", "", rest).strip()
    elif re.search(r"(?i)\bзавтра\b", rest):
        day_shift = 1
        rest = re.sub(r"(?i)\bзавтра\b", "", rest).strip()
    elif re.search(r"(?i)\bсегодня\b", rest):
        day_shift = 0
        rest = re.sub(r"(?i)\bсегодня\b", "", rest).strip()

    if day_shift is not None:
        # время: "в 10:30" или "в 10"
        time_h, time_m = 10, 0  # дефолт: 10:00
        mt = re.search(r"(?i)\bв\s*(\d{1,2})(?::(\d{2}))?\b", rest)
        if mt:
            time_h = int(mt.group(1))
            time_m = int(mt.group(2) or "0")
            rest = re.sub(r"(?i)\bв\s*\d{1,2}(?::\d{2})?\b", "", rest).strip()

        msg = rest.strip(" ,.-")
        if not msg:
            msg = "напоминание"

        due = (now_local + timedelta(days=day_shift)).replace(
            hour=time_h, minute=time_m, second=0, microsecond=0
        )
        # если “сегодня” и время уже прошло — сдвинем на +1 час, чтобы не было “в прошлом”
        if due <= now_local:
            due = now_local + timedelta(hours=1)
            due = due.replace(second=0, microsecond=0)

        return due, msg

    return None


async def schedule_reminder(bot: Bot, rem: Reminder):
    """
    Ждём до времени и отправляем сообщение.
    """
    now = int(time.time())
    wait_sec = max(0, rem.due_utc - now)

    async def runner():
        try:
            await asyncio.sleep(wait_sec)
            await bot.send_message(rem.chat_id, f"⏰ Напоминание: {rem.text}")
        finally:
            # чистим из БД и из задач
            try:
                async with aiosqlite.connect(DB_PATH) as db:
                    await db.execute("DELETE FROM reminders WHERE id = ?", (rem.id,))
                    await db.commit()
            except Exception:
                log.exception("Failed to delete reminder from DB")
            scheduled_tasks.pop(rem.id, None)

    task = asyncio.create_task(runner())
    scheduled_tasks[rem.id] = task


async def restore_scheduled(bot: Bot):
    """
    При старте поднимаем напоминания из БД и планируем снова.
    """
    reminders = await store.due_after_now()
    for rem in reminders:
        if rem.id not in scheduled_tasks:
            await schedule_reminder(bot, rem)


# ======================
# Gemini call (без блокировки event loop)
# ======================
def _gemini_generate_sync(prompt: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    txt = getattr(resp, "text", None) or ""
    return txt.strip()


async def gemini_generate(prompt: str, retries: int = 3) -> str:
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            return await asyncio.to_thread(_gemini_generate_sync, prompt)
        except Exception as e:
            log.warning("Gemini error attempt %s/%s: %s", attempt, retries, e)
            if attempt == retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2
    return ""


# ======================
# Bot handlers
# ======================
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Напиши сообщение — отвечу как ИИ 🙂\n"
        "Команды:\n"
        "• /reset — сбросить контекст\n"
        "• /remind <текст> — поставить напоминание\n"
        "• /reminders — список напоминаний\n"
        "• /cancel <id> — отменить напоминание\n\n"
        "Можно и без команд: напиши, например:\n"
        "«напомни мне завтра в 10:30 распечатать документы»"
    )


async def cmd_reset(message: Message):
    history[message.chat.id].clear()
    await message.answer("Ок, сбросил контекст. Пиши заново 🙂")


async def cmd_reminders(message: Message):
    items = await store.list_for_chat(message.chat.id)
    if not items:
        await message.answer("Напоминаний нет.")
        return

    lines = ["📌 Твои напоминания:"]
    for r in items:
        dt_local = datetime.fromtimestamp(r.due_utc, tz=timezone.utc).astimezone(TZ)
        lines.append(f"• id={r.id} — {dt_local:%Y-%m-%d %H:%M} — {r.text}")
    await message.answer("\n".join(lines))


async def cmd_cancel(message: Message):
    # /cancel 123
    parts = (message.text or "").split()
    if len(parts) < 2 or not parts[1].isdigit():
        await message.answer("Формат: /cancel <id>")
        return
    rid = int(parts[1])

    ok = await store.delete(rid, message.chat.id)
    task = scheduled_tasks.pop(rid, None)
    if task:
        task.cancel()

    await message.answer("✅ Отменил." if ok else "Не нашёл такое напоминание.")


async def handle_remind_text(message: Message, text: str):
    now_local = datetime.now(TZ)
    parsed = parse_reminder_ru(text, now_local)
    if not parsed:
        await message.answer(
            "Не понял время 😅\n"
            "Примеры:\n"
            "• напомни завтра в 10:30 распечатать документы\n"
            "• напомни через 15 минут размяться\n"
            "• /remind завтра 18:00 позвонить"
        )
        return

    due_local, msg = parsed
    due_utc = int(due_local.astimezone(timezone.utc).timestamp())

    rid = await store.add(message.chat.id, due_utc, msg)
    rem = Reminder(id=rid, chat_id=message.chat.id, due_utc=due_utc, text=msg, created_utc=int(time.time()))
    await schedule_reminder(message.bot, rem)

    await message.answer(f"✅ Ок! Напомню {due_local:%Y-%m-%d %H:%M}: {msg}")


async def cmd_remind(message: Message):
    # /remind <что-то>
    txt = (message.text or "")
    rest = txt[len("/remind"):].strip()
    if not rest:
        await message.answer("Напиши так: /remind завтра в 10:30 распечатать документы")
        return
    # поддержим “/remind завтра 10:30 …” без слова “в”
    # превратим в форму “напомни …”
    fake = "напомни " + rest
    await handle_remind_text(message, fake)


async def chat(message: Message):
    # Игнорим пустые/не текстовые
    text = (message.text or "").strip()
    if not text:
        return

    # 1) если это “напомни …” — делаем напоминание
    if re.match(r"(?i)^\s*напомни", text):
        await handle_remind_text(message, text)
        return

    # 2) антифлуд на пользователя
    uid = message.from_user.id if message.from_user else 0
    now = time.time()
    prev = last_user_call.get(uid, 0.0)
    if now - prev < USER_COOLDOWN_SEC:
        await message.answer("Секунду 🙂")
        return
    last_user_call[uid] = now

    chat_id = message.chat.id

    # 3) блокируем чат, чтобы не было гонок (2 запроса одновременно)
    async with chat_locks[chat_id]:
        # добавляем user в память
        history[chat_id].append({"role": "user", "content": text})

        thinking = await message.answer("Думаю…")

        try:
            prompt = build_prompt(chat_id)
            answer = await gemini_generate(prompt)
        except Exception:
            log.exception("Failed to generate")
            # откатим последнее сообщение пользователя, чтобы не ломать память мусором
            if history[chat_id] and history[chat_id][-1]["role"] == "user":
                history[chat_id].pop()
            await thinking.edit_text("Ошибка при обращении к модели. Попробуй ещё раз позже 🙏")
            return

        if not answer:
            answer = "Похоже, я не получил ответ. Попробуй переформулировать."

        # добавляем assistant в память
        history[chat_id].append({"role": "assistant", "content": answer})

        parts = split_text(answer, MAX_REPLY_CHARS)
        await thinking.edit_text(parts[0])
        for p in parts[1:]:
            await message.answer(p)


async def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing")

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_reset, Command("reset"))
    dp.message.register(cmd_remind, Command("remind"))
    dp.message.register(cmd_reminders, Command("reminders"))
    dp.message.register(cmd_cancel, Command("cancel"))

    dp.message.register(chat, F.text)

    await store.init()
    await restore_scheduled(bot)

    log.info("Bot started. TZ=%s model=%s", BOT_TIMEZONE, MODEL_NAME)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
