import asyncio
import json
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from openai import AsyncOpenAI
from google.cloud import firestore
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai-bot")

# ---------- الإعدادات من متغيّرات البيئة ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "أنت مساعد ذكي ودود تتحدث العربية بطلاقة. أجب باختصار ووضوح ومن دون حشو.",
)
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))  # عدد أزواج (سؤال+جواب) المحفوظة في السياق

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")
WEBHOOK_PATH = f"/webhook/{WEBHOOK_SECRET}"
BASE_URL = (os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL", "")).rstrip("/")
PORT = int(os.getenv("PORT", "10000"))

# ---------- العملاء ----------
oai = AsyncOpenAI(api_key=OPENAI_API_KEY)

_creds_info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
_credentials = service_account.Credentials.from_service_account_info(_creds_info)
db = firestore.AsyncClient(project=_creds_info["project_id"], credentials=_credentials)

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
dp = Dispatcher()

# ---------- تخزين المحادثة في Firestore ----------
def _doc(user_id: int):
    return db.collection("chats").document(str(user_id))

async def load_history(user_id: int):
    snap = await _doc(user_id).get()
    if snap.exists:
        return snap.to_dict().get("history", [])
    return []

async def save_history(user_id: int, history):
    trimmed = history[-(MAX_TURNS * 2):]
    await _doc(user_id).set({"history": trimmed})

async def clear_history(user_id: int):
    await _doc(user_id).set({"history": []})

# ---------- استدعاء OpenAI ----------
async def ask_ai(history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    resp = await oai.chat.completions.create(model=OPENAI_MODEL, messages=messages)
    return (resp.choices[0].message.content or "").strip()

# ---------- تقسيم الرسائل الطويلة (حد تيليجرام 4096) ----------
def split_text(text, limit=4000):
    return [text[i:i + limit] for i in range(0, len(text), limit)] or ["..."]

# ---------- الأوامر ----------
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "أهلاً! 👋 أنا بوت ذكاء اصطناعي مدعوم بـ "
        f"{OPENAI_MODEL}.\n"
        "اكتب أي شي وراح أرد عليك، وأتذكّر سياق محادثتنا.\n\n"
        "/reset — مسح المحادثة والبدء من جديد\n"
        "/model — معرفة الموديل الحالي"
    )

@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    await clear_history(message.from_user.id)
    await message.answer("تم مسح المحادثة. 🧹 نبدأ من جديد.")

@dp.message(Command("model"))
async def cmd_model(message: Message):
    await message.answer(f"الموديل الحالي: {OPENAI_MODEL}")

# ---------- الرسائل النصية ----------
@dp.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message):
    user_id = message.from_user.id
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        history = await load_history(user_id)
        history.append({"role": "user", "content": message.text})
        answer = await ask_ai(history)
        history.append({"role": "assistant", "content": answer})
        await save_history(user_id, history)
        for chunk in split_text(answer):
            await message.answer(chunk)
    except Exception:
        log.exception("خطأ أثناء المعالجة")
        await message.answer("صار خطأ بسيط، جرّب مرة ثانية بعد شوي. 🙏")

# ---------- الويب هوك ----------
async def on_startup(app: web.Application):
    if not BASE_URL:
        log.warning("ما في WEBHOOK_URL / RENDER_EXTERNAL_URL — الويب هوك ما راح ينضبط.")
        return
    url = BASE_URL + WEBHOOK_PATH
    await bot.set_webhook(url, secret_token=WEBHOOK_SECRET, drop_pending_updates=True)
    log.info("Webhook set: %s", url)

async def on_shutdown(app: web.Application):
    await bot.delete_webhook()
    await bot.session.close()

async def health(request):
    return web.Response(text="OK")

def main():
    app = web.Application()
    app.router.add_get("/", health)
    SimpleRequestHandler(dispatcher=dp, bot=bot, secret_token=WEBHOOK_SECRET).register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
