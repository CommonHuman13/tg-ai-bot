import os
import asyncio
import logging
from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from openai import AsyncOpenAI

# ---------- ЛОГИ ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tg-ai-bot")

# ---------- ENV ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# сколько "ходов" держим в памяти (user+assistant)
MAX_TURNS = int(os.getenv("MAX_TURNS", "12"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

# ---------- ПАМЯТЬ ----------
# На каждый chat_id: deque из сообщений вида {"role": "...", "content": "..."}
history = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))

SYSTEM_PROMPT = (
    "Ты полезный ассистент в Telegram. "
    "Отвечай понятно, по делу, без воды. "
    "Если не уверен — скажи, что не уверен."
)

# ---------- КЛИЕНТЫ ----------
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def on_start(m: types.Message):
    await m.answer(
        "Привет! Напиши вопрос — отвечу 🙂\n"
        "Команды:\n"
        "/reset — очистить память чата\n"
        "/model — показать текущую модель\n"
    )


@dp.message(Command("model"))
async def on_model(m: types.Message):
    await m.answer(f"Текущая модель: {OPENAI_MODEL}\nПамять: {MAX_TURNS} turns")


@dp.message(Command("reset"))
async def on_reset(m: types.Message):
    history[m.chat.id].clear()
    await m.answer("Ок, контекст чата очищен ✅")


@dp.message()
async def on_text(m: types.Message):
    text = (m.text or "").strip()
    if not text:
        return

    chat_id = m.chat.id

    # добавляем user в историю
    history[chat_id].append({"role": "user", "content": text})

    thinking = await m.answer("Думаю…")

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += list(history[chat_id])

        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
        )

        answer = (resp.choices[0].message.content or "").strip()
        if not answer:
            answer = "Пустой ответ 😅"

        # добавляем assistant в историю
        history[chat_id].append({"role": "assistant", "content": answer})

        # Telegram лимит на сообщение — примерно 4096
        await thinking.edit_text(answer[:4000])

    except Exception as e:
        log.exception("OpenAI error")
        await thinking.edit_text(f"Ошибка запроса к модели: {e}")


async def main():
    # важное: на всякий убираем webhook, чтобы не было конфликтов
    await bot.delete_webhook(drop_pending_updates=True)

    log.info("Bot starting… model=%s max_turns=%s", OPENAI_MODEL, MAX_TURNS)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

