"""Roleplay chat bot: watches chat and occasionally speaks in character.

Run:            python bot.py
List models:    python bot.py --models
Env vars:       DISCORD_TOKEN, GROQ_API_KEY (plus any key_env used in config.json)
"""

import asyncio
import datetime
import json
import os
import random
import re
import sys
import time
from collections import defaultdict, deque

import aiohttp
import discord

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))


def read(name):
    with open(os.path.join(BASE, name), encoding="utf-8") as f:
        return f.read().strip()


CONFIG = json.loads(read("config.json"))
NAME = CONFIG["character_name"]
CHARACTER = read("character.md").replace("{name}", NAME)
THEMES = [
    t.strip()
    for t in read("themes.txt").splitlines()
    if t.strip() and not t.startswith("#")
]

history = defaultdict(lambda: deque(maxlen=CONFIG["context_messages"]))
own_recent = deque(maxlen=6)
last_spoke = {}  # channel_id -> time the bot last posted
last_checked = {}  # channel_id -> time the bot last called the model
busy = set()  # channels with a reply already in progress
usage = {"date": None, "calls": 0}
NO_MENTIONS = discord.AllowedMentions.none()

MEMORY_DIR = os.path.join(BASE, CONFIG.get("memory_dir", "memory"))
os.makedirs(MEMORY_DIR, exist_ok=True)
MEMORY_TAG = re.compile(r"(?im)^\s*MEMORY:\s*(.+?)\s*$")

memories = {}  # guild_id -> list[str], loaded lazily and cached per server


def _memory_path(guild_id):
    return os.path.join(MEMORY_DIR, f"{guild_id}.json")


def get_memory(guild_id):
    if guild_id not in memories:
        try:
            with open(_memory_path(guild_id), encoding="utf-8") as f:
                data = json.load(f)
                memories[guild_id] = list(data) if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            memories[guild_id] = []
    return memories[guild_id]


def save_memory(guild_id):
    try:
        with open(_memory_path(guild_id), "w", encoding="utf-8") as f:
            json.dump(memories[guild_id], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[memory:{guild_id}] failed to save:", e)


def remember(guild_id, point):
    point = point.strip()
    mem = get_memory(guild_id)
    if not point or point in mem:
        return
    mem.append(point)
    cap = CONFIG.get("memory_cap", 10)
    while len(mem) > cap:
        dropped = mem.pop(0)
        print(f"[memory:{guild_id}] full, forgot: {dropped!r}")
    save_memory(guild_id)
    print(f"[memory:{guild_id}] added: {point!r}")


def extract_memory(text, guild_id):
    """Pull out a MEMORY: line if present, save it, and return the cleaned reply text."""
    m = MEMORY_TAG.search(text)
    if not m:
        return text
    remember(guild_id, m.group(1))
    return MEMORY_TAG.sub("", text).strip()


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


# ---------- budget and gate ----------


def budget_left():
    today = datetime.date.today()
    if usage["date"] != today:
        usage["date"], usage["calls"] = today, 0
    return usage["calls"] < CONFIG["daily_cap"]


def is_direct(message):
    """True if someone mentioned the bot or replied to one of its messages."""
    if not CONFIG["reply_when_mentioned"]:
        return False
    if client.user in message.mentions:
        return True
    ref = message.reference.resolved if message.reference else None
    return isinstance(ref, discord.Message) and ref.author == client.user


def should_speak(message, direct):
    cid = message.channel.id
    now = time.time()
    if not budget_left():
        return False
    if direct:
        return now - last_spoke.get(cid, 0) > CONFIG["direct_cooldown_seconds"]
    if now - last_spoke.get(cid, 0) < CONFIG["channel_cooldown_seconds"]:
        return False
    if now - last_checked.get(cid, 0) < CONFIG["check_cooldown_seconds"]:
        return False
    if len(message.clean_content) < CONFIG["min_message_length"]:
        return False
    chance = CONFIG["base_chance"]
    if "?" in message.content:
        chance += CONFIG["question_bonus"]
    return random.random() < chance


# ---------- model calls ----------


def build_prompt(channel_name, channel_id, direct, guild):
    chat = "\n".join(f"{who}: {text}" for who, text in history[channel_id])
    lines = [f"Recent chat in #{channel_name}:", chat, ""]
    if own_recent:
        lines.append("Your own recent messages (don't repeat these or their jokes):")
        lines += [f"- {t}" for t in own_recent]
        lines.append("")
    if THEMES and random.random() < CONFIG["theme_chance"]:
        picks = random.sample(THEMES, k=min(len(THEMES), random.randint(1, 2)))
        lines.append(
            "Optional inspiration, use only if it fits naturally, otherwise ignore: "
            + "; ".join(picks)
        )
        lines.append("")
    emojis = list_custom_emojis(guild)
    if emojis:
        lines.append(
            "Available server emojis (name -> exact code to paste if you use one):"
        )
        lines += emojis
        lines.append("")
    mem = get_memory(guild.id) if guild else []
    if mem:
        lines.append("Things you remember about this server:")
        lines += [f"- {m}" for m in mem]
        lines.append("")
    if direct:
        lines.append("Someone is talking to you directly. Reply to the latest message.")
    else:
        lines.append(
            "Nobody addressed you. Only chime in if you have something genuinely "
            "funny or fitting to add. If not, reply with exactly: SKIP"
        )
    lines.append("Reply with only the message you would send.")
    return "\n".join(lines)


def list_custom_emojis(guild):
    """Format a guild's custom emojis as 'name -> <code>' lines, for the prompt."""
    if guild is None or not guild.emojis:
        return None
    emojis = list(guild.emojis)
    limit = CONFIG.get("max_custom_emojis_shown", 30)
    if len(emojis) > limit:
        emojis = random.sample(emojis, limit)
    lines = []
    for e in emojis:
        code = f"<a:{e.name}:{e.id}>" if e.animated else f"<:{e.name}:{e.id}>"
        lines.append(f"{e.name} -> {code}")
    return lines


GIF_TAG = re.compile(r"\[gif\s*:\s*(.*?)\]", re.IGNORECASE)


async def resolve_gifs(session, text):
    """Replace [gif: query] tags with a real Klipy GIF link, or drop the tag if lookup fails."""
    key = os.environ.get(CONFIG.get("gif_key_env", "KLIPY_API_KEY"))
    matches = list(GIF_TAG.finditer(text))
    if not matches:
        return text
    if not key:
        print("[gif] KLIPY_API_KEY not set, dropping gif tag(s)")
        return GIF_TAG.sub("", text).strip()

    for m in matches:
        query = m.group(1).strip() or "funny"
        url = await search_gif(session, query, key)
        replacement = url if url else ""
        text = text.replace(m.group(0), replacement, 1)
    return text.strip()


def _first_media_url(file_data):
    """Klipy nests format variants (gif/webp/jpg/mp4/webm) under a size tier
    (hd/md/sm/...) inside 'file'; grab any usable url."""
    if not isinstance(file_data, dict):
        return None
    for size in file_data.values():
        if not isinstance(size, dict):
            continue
        for fmt_key in ("gif", "webp", "mp4"):
            entry = size.get(fmt_key)
            if isinstance(entry, dict) and entry.get("url"):
                return entry["url"]
        for entry in size.values():
            if isinstance(entry, dict) and entry.get("url"):
                return entry["url"]
    return None


async def search_gif(session, query, key):
    url = f"https://api.klipy.com/api/v1/{key}/gifs/search"
    params = {"q": query, "per_page": 8, "customer_id": "furina-bot"}
    try:
        async with session.get(url, params=params) as r:
            if r.status != 200:
                print(f"[gif] Klipy HTTP {r.status}: {(await r.text())[:200]}")
                return None
            data = await r.json()
            items = ((data.get("data") or {}).get("data")) or []
            if not items:
                return None
            pick = random.choice(items)
            return _first_media_url(pick.get("file"))
    except Exception as e:
        print("[gif] Klipy error:", e)
        return None


async def ask_model(system, user):
    """Try each provider in order; move on if one is rate-limited or errors."""
    base = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": CONFIG["temperature"],
        "max_tokens": CONFIG["max_tokens"],
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for p in CONFIG["providers"]:
            key = os.environ.get(p["key_env"])
            if not key:
                continue
            body = {**base, "model": p["model"], **p.get("extra_params", {})}
            try:
                async with session.post(
                    p["base_url"] + "/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {key}"},
                ) as r:
                    if r.status != 200:
                        print(
                            f"[{p['name']}] HTTP {r.status}: {(await r.text())[:200]}"
                        )
                        continue
                    data = await r.json()
                    text = (data["choices"][0]["message"].get("content") or "").strip()
                    if text:
                        print(
                            f"[{p['name']}] model output before gif resolve: {text!r}"
                        )
                        return await resolve_gifs(session, text)
                    print(f"[{p['name']}] empty reply (raise max_tokens?)")
            except Exception as e:
                print(f"[{p['name']}] error: {e}")
    return None


async def list_models():
    async with aiohttp.ClientSession() as session:
        for p in CONFIG["providers"]:
            key = os.environ.get(p["key_env"])
            if not key:
                print(f"{p['name']}: {p['key_env']} not set")
                continue
            async with session.get(
                p["base_url"] + "/models", headers={"Authorization": f"Bearer {key}"}
            ) as r:
                data = await r.json()
                print(p["name"], sorted(m["id"] for m in data.get("data", [])))


# ---------- discord events ----------


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author.bot or message.guild is None:
        return
    allowed = CONFIG["channel_ids"]
    if allowed and message.channel.id not in allowed:
        return

    cid = message.channel.id
    history[cid].append((message.author.display_name, message.clean_content[:300]))

    direct = is_direct(message)
    if cid in busy or not should_speak(message, direct):
        return

    busy.add(cid)
    try:
        await asyncio.sleep(random.uniform(*CONFIG["delay_range_seconds"]))
        usage["calls"] += 1
        last_checked[cid] = time.time()
        async with message.channel.typing():
            prompt = build_prompt(message.channel.name, cid, direct, message.guild)
            reply = await ask_model(CHARACTER, prompt)
        print(f"[{NAME}] raw reply: {reply!r}")
        if not reply or reply.strip().upper().startswith("SKIP"):
            return

        text = extract_memory(reply.strip(), message.guild.id)
        if not text:
            return
        gif_match = re.search(
            r"https://\S+\.(?:gif|webp|mp4)(?:\?\S*)?", text, re.IGNORECASE
        )
        if gif_match:
            head, gif_url = text[: gif_match.start()].strip(), gif_match.group(0)
            text = (head[: CONFIG["max_chars"]] + "\n" + gif_url) if head else gif_url
        else:
            text = text[: CONFIG["max_chars"]]
        if direct or random.random() < CONFIG["reply_style_chance"]:
            await message.reply(
                text, mention_author=False, allowed_mentions=NO_MENTIONS
            )
        else:
            await message.channel.send(text, allowed_mentions=NO_MENTIONS)

        last_spoke[cid] = time.time()
        own_recent.append(text)
        history[cid].append((NAME, text))
    except discord.HTTPException as e:
        print("Discord error:", e)
    finally:
        busy.discard(cid)


if __name__ == "__main__":
    if "--models" in sys.argv:
        asyncio.run(list_models())
    else:
        client.run(os.environ["DISCORD_TOKEN"])
