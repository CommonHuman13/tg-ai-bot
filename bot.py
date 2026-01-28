import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from google import genai

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def start(m: types.Message):
    await m.answer("Привет! Напиши сообщение — отвечу как ИИ 🙂\nКоманда: /reset (позже добавим)")

@dp.message()
async def chat(m: types.Message):
    text = (m.text or "").strip()
    if not text:
        return
    thinking = await m.answer("Думаю…")
    resp = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=text
    )
    await thinking.edit_text((resp.text or "…")[:4000])

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
