import asyncio
import logging
import json
import os
from datetime import datetime, timezone, timedelta

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

# ---------- الإعدادات ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "أنت مساعد ذكي ودود تتحدث العربية بطلاقة. أجب باختصار ووضوح ومن دون حشو.",
)
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8000"))
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "1") == "1"
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT", "90"))

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")
WEBHOOK_PATH = f"/webhook/{WEBHOOK_SECRET}"
BASE_URL = (os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL", "")).rstrip("/")
PORT = int(os.getenv("PORT", "10000"))

KSA = timezone(timedelta(hours=3))
AR_DAYS = ["الإثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]

# ---------- العملاء ----------
oai = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=float(AI_TIMEOUT), max_retries=1)

_creds_info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
_credentials = service_account.Credentials.from_service_account_info(_creds_info)
db = firestore.AsyncClient(project=_creds_info["project_id"], credentials=_credentials)

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
dp = Dispatcher()

# ---------- Firestore ----------
def _doc(user_id: int):
    return db.collection("chats").document(str(user_id))

async def load_history(user_id: int):
    snap = await _doc(user_id).get()
    return snap.to_dict().get("history", []) if snap.exists else []

async def save_history(user_id: int, history):
    await _doc(user_id).set({"history": history[-(MAX_TURNS * 2):]})

async def clear_history(user_id: int):
    await _doc(user_id).set({"history": []})

# ---------- تعليمات النظام مع الوقت ----------
def build_instructions():
    now = datetime.now(KSA)
    stamp = f"{AR_DAYS[now.weekday()]} {now.strftime('%Y-%m-%d %H:%M')}"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"معلومة آنية موثوقة: الوقت الآن {stamp} بتوقيت السعودية (UTC+3). "
        "استخدم هذا الوقت مباشرة لأي سؤال عن الوقت أو التاريخ ولا تقل إنك لا تعرفه. "
        "إذا احتاج السؤال معلومات حديثة استخدم أداة البحث، واكتفِ بعملية بحث واحدة لتقليل الانتظار."
    )

# ---------- نداء OpenAI ----------
async def ask_ai(history):
    kwargs = dict(
        model=OPENAI_MODEL,
        instructions=build_instructions(),
        input=history,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    if ENABLE_WEB_SEARCH:
        kwargs["tools"] = [{
            "type": "web_search",
            "user_location": {"type": "approximate", "country": "SA", "timezone": "Asia/Riyadh"},
        }]
    resp = await asyncio.wait_for(oai.responses.create(**kwargs), timeout=AI_TIMEOUT)
    text = (resp.output_text or "").strip()
    return text or "ما قدرت أطلّع جواب كامل، جرّب تعيد صياغة السؤال. 🙏"

# ---------- إبقاء مؤشر "يكتب…" ----------
async def keep_typing(chat_id):
    try:
        while True:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass

def split_text(text, limit=4000):
    return [text[i:i + limit] for i in range(0, len(text), limit)] or ["..."]

# ---------- الأوامر ----------
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        f"أهلاً! 👋 بوت ذكاء اصطناعي ({OPENAI_MODEL}) مع بحث في الإنترنت.\n"
        "/reset — مسح المحادثة\n/model — معلومات الموديل"
    )

@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    await clear_history(message.from_user.id)
    await message.answer("تم مسح المحادثة. 🧹")

@dp.message(Command("model"))
async def cmd_model(message: Message):
    state = "مفعّل" if ENABLE_WEB_SEARCH else "معطّل"
    await message.answer(f"الموديل: {OPENAI_MODEL}\nبحث الإنترنت: {state}\nالحالة: شغّال ✅")

# ---------- الرسائل ----------
@dp.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message):
    user_id = message.from_user.id
    typing = asyncio.create_task(keep_typing(message.chat.id))
    try:
        history = await load_history(user_id)
        history.append({"role": "user", "content": message.text})
        answer = await ask_ai(history)
        history.append({"role": "assistant", "content": answer})
        await save_history(user_id, history)
    except asyncio.TimeoutError:
        log.warning("OpenAI timeout")
        answer = "السؤال أخذ وقت أطول من اللازم وانقطع. جرّب مرة ثانية. ⏳"
    except Exception as e:
        log.exception("خطأ أثناء المعالجة")
        answer = f"⚠️ خطأ للتشخيص:\n{type(e).__name__}: {e}"
    finally:
        typing.cancel()
    for chunk in split_text(answer):
        await message.answer(chunk)

# ---------- الويب هوك ----------
async def on_startup(app: web.Application):
    if not BASE_URL:
        log.warning("ما في WEBHOOK_URL / RENDER_EXTERNAL_URL.")
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
