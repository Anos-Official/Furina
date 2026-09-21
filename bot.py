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
THEMES = [t.strip() for t in read("themes.txt").splitlines()
          if t.strip() and not t.startswith("#")]

history = defaultdict(lambda: deque(maxlen=CONFIG["context_messages"]))
own_recent = deque(maxlen=6)
last_spoke = {}     # channel_id -> time the bot last posted
last_checked = {}   # channel_id -> time the bot last called the model
busy = set()        # channels with a reply already in progress
usage = {"date": None, "calls": 0}
NO_MENTIONS = discord.AllowedMentions.none()

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

def build_prompt(channel_name, channel_id, direct):
    chat = "\n".join(f"{who}: {text}" for who, text in history[channel_id])
    lines = [f"Recent chat in #{channel_name}:", chat, ""]
    if own_recent:
        lines.append("Your own recent messages (don't repeat these or their jokes):")
        lines += [f"- {t}" for t in own_recent]
        lines.append("")
    if THEMES and random.random() < CONFIG["theme_chance"]:
        picks = random.sample(THEMES, k=min(len(THEMES), random.randint(1, 2)))
        lines.append("Optional inspiration, use only if it fits naturally, otherwise ignore: "
                     + "; ".join(picks))
        lines.append("")
    if direct:
        lines.append("Someone is talking to you directly. Reply to the latest message.")
    else:
        lines.append("Nobody addressed you. Only chime in if you have something genuinely "
                     "funny or fitting to add. If not, reply with exactly: SKIP")
    lines.append("Reply with only the message you would send.")
    return "\n".join(lines)


async def ask_model(system, user):
    """Try each provider in order; move on if one is rate-limited or errors."""
    base = {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": CONFIG["temperature"],
        "max_tokens": CONFIG["max_tokens"],
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for p in CONFIG["providers"]:
            key = os.environ.get(p["key_env"])
            if not key:
                print(f"[{p['name']}] {p['key_env']} is not set, skipping")
            body = {**base, "model": p["model"], **p.get("extra_params", {})}
            try:
                async with session.post(p["base_url"] + "/chat/completions", json=body,
                                        headers={"Authorization": f"Bearer {key}"}) as r:
                    if r.status != 200:
                        print(f"[{p['name']}] HTTP {r.status}: {(await r.text())[:200]}")
                        continue
                    data = await r.json()
                    text = (data["choices"][0]["message"].get("content") or "").strip()
                    if text:
                        return text
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
            async with session.get(p["base_url"] + "/models",
                                    headers={"Authorization": f"Bearer {key}"}) as r:
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
            reply = await ask_model(CHARACTER, build_prompt(message.channel.name, cid, direct))
            print(f"[debug] model reply: {reply!r}")
        if not reply or reply.strip().upper().startswith("SKIP"):
            return

        text = reply.strip()[: CONFIG["max_chars"]]
        if direct or random.random() < CONFIG["reply_style_chance"]:
            await message.reply(text, mention_author=False, allowed_mentions=NO_MENTIONS)
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
