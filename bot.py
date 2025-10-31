import asyncio
import json
import logging
import os
import time
import struct
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

import discord
from discord.ext import commands
from discord.sinks import Filters, Sink
import numpy as np
from dotenv import load_dotenv
from vosk import KaldiRecognizer, Model




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("blad-bot")

# Load configuration from .env (optional).
load_dotenv()


def _load_env_list(var_name: str, *, default: str = "") -> set[str]:
    raw_value = os.getenv(var_name, default)
    if not raw_value:
        return set()
    return {item.strip().lower() for item in raw_value.split(",") if item.strip()}


DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise SystemExit(
        "DISCORD_TOKEN is not set. Create a bot token at https://discord.com/developers "
        "and place it in a .env file or environment variable."
    )

MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "./model")
MODEL_PATH = Path(MODEL_PATH).expanduser().resolve()
if not MODEL_PATH.exists():
    raise SystemExit(
        f"Vosk model directory '{MODEL_PATH}' is missing. "
        "Download a model (e.g. Russian: vosk-model-small-ru-0.22) "
        "and unpack it into this path."
    )

TARGET_WORDS = _load_env_list("TARGET_WORDS", default="блад,blad")
if not TARGET_WORDS:
    raise SystemExit(
        "TARGET_WORDS is empty. Set TARGET_WORDS=word1,word2 in .env or environment."
    )

DETECTION_MESSAGE = os.getenv(
    "DETECTION_MESSAGE",
    '🚨 {user_tag} ({user}) произнёс слово "{word}" в {channel}. Фраза: "{transcript}"',
)

DETECTION_COOLDOWN = float(os.getenv("DETECTION_COOLDOWN_SECONDS", "3.0"))
DETECTION_CONTEXT_SECONDS = float(os.getenv("DETECTION_CONTEXT_SECONDS", "1.0"))

# Prepare Vosk speech recognition model.
try:
    VOSK_MODEL = Model(str(MODEL_PATH))
except Exception as exc:  # pragma: no cover - dependency failure
    raise SystemExit(f"Не удалось загрузить модель Vosk: {exc}") from exc

RECOGNIZER_SAMPLE_RATE = 16_000  # Hz, expected input sample rate for the recognizer.

# Discord bot setup.
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents)


def _create_recognizer() -> KaldiRecognizer:
    recognizer = KaldiRecognizer(VOSK_MODEL, RECOGNIZER_SAMPLE_RATE)
    recognizer.SetWords(True)
    return recognizer


def _normalize_text(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


class KeywordDetectSink(Sink):
    """Custom audio sink that looks for predefined keywords and notifies a callback."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        target_words: Iterable[str],
        on_detect: Callable[[int, str, str], Awaitable[None]],
        cooldown: float,
        context_seconds: float,
    ) -> None:
        super().__init__()
        self.loop = loop
        self.on_detect = on_detect
        self.target_words = {word.lower() for word in target_words}
        self.cooldown = cooldown
        self.context_seconds = context_seconds
        self._recognizers: dict[int, KaldiRecognizer] = {}
        self._last_trigger: dict[int, float] = {}

    @Filters.container  # type: ignore[attr-defined]
    def write(self, data: bytes, user: Any) -> None:
        try:
            user_id = int(user)
        except (TypeError, ValueError):
            return

        pcm = self._convert_and_resample(data)
        if not pcm:
            return

        recognizer = self._recognizers.setdefault(user_id, _create_recognizer())
        try:
            if recognizer.AcceptWaveform(pcm):
                result = json.loads(recognizer.Result())
                self._handle_final_result(user_id, recognizer, result)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Ошибка распознавания речи: %s", exc)

    def cleanup(self) -> None:
        # Override to avoid keeping audio buffers alive.
        self.finished = True

    def _convert_and_resample(self, pcm_bytes: bytes) -> bytes:
        if not pcm_bytes:
            return b""
        data = np.frombuffer(pcm_bytes, dtype=np.int16)
        if data.size == 0:
            return b""
        if data.size % 2 != 0:
            data = data[:-1]  # ensure stereo pairs are complete
        if data.size == 0:
            return b""
        stereo = data.reshape(-1, 2)
        mono = stereo.mean(axis=1).astype(np.int16)
        downsampled = mono[::3]  # 48000 -> 16000 Hz
        return downsampled.tobytes()

    def _handle_final_result(
        self, user_id: int, recognizer: KaldiRecognizer, result: dict[str, Any]
    ) -> None:
        transcript = (result.get("text") or "").strip()
        words = result.get("result") or []

        match = self._find_match(words, transcript)
        if match is None:
            recognizer.Reset()
            return

        matched_word, start, end = match
        now = time.perf_counter()
        if now - self._last_trigger.get(user_id, 0.0) < self.cooldown:
            recognizer.Reset()
            return

        snippet = self._build_context_snippet(words, start, end, transcript)
        snippet_text = snippet or transcript or matched_word

        self._last_trigger[user_id] = now
        asyncio.run_coroutine_threadsafe(
            self.on_detect(user_id, matched_word, snippet_text), self.loop
        )
        recognizer.Reset()

    def _find_match(
        self, words: list[dict[str, Any]], transcript: str
    ) -> tuple[str, float, float] | None:
        for entry in words:
            token = entry.get("word", "").lower()
            if self._matches(token):
                start = float(entry.get("start", 0.0))
                end = float(entry.get("end", start))
                return token, start, end

        for token in _normalize_text(transcript):
            if self._matches(token):
                return token, 0.0, 0.0
        return None

    def _matches(self, token: str) -> bool:
        for target in self.target_words:
            if target in token:
                return True
        return False

    def _build_context_snippet(
        self,
        words: list[dict[str, Any]],
        target_start: float,
        target_end: float,
        fallback: str,
    ) -> str:
        if words:
            window_start = max(0.0, target_start - self.context_seconds)
            window_end = target_end + self.context_seconds
            snippet_words: list[str] = []
            for entry in words:
                word = entry.get("word")
                if not word:
                    continue
                start = float(entry.get("start", target_start))
                end = float(entry.get("end", start))
                if end >= window_start and start <= window_end:
                    snippet_words.append(word)
            snippet = " ".join(snippet_words).strip()
            if snippet:
                return snippet
        return fallback.strip()


active_sinks: dict[int, KeywordDetectSink] = {}


async def _send_detection_message(
    text_channel: discord.abc.Messageable,
    voice_channel: discord.abc.Connectable | None,
    *,
    user_mention: str,
    user_tag: str,
    word: str,
    transcript: str,
) -> None:
    channel_label = (
        getattr(voice_channel, "mention", None)
        or getattr(voice_channel, "name", "голосовом канале")
    )
    message = DETECTION_MESSAGE.format(
        word=word,
        transcript=transcript,
        user=user_mention,
        user_tag=user_tag,
        channel=channel_label,
    )
    await text_channel.send(message)


async def _on_recording_stopped(
    sink: KeywordDetectSink,
    ctx: commands.Context,
) -> None:
    active_sinks.pop(ctx.guild.id, None)
    await ctx.send("Остановил прослушивание голосового канала.")


@bot.command(name="join")
async def join(ctx: commands.Context) -> None:
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Сначала зайдите в голосовой канал, чтобы я мог присоединиться.")
        return

    voice_channel = ctx.author.voice.channel
    voice_client = ctx.voice_client

    if voice_client and voice_client.channel != voice_channel:
        await voice_client.move_to(voice_channel)
    elif not voice_client:
        voice_client = await voice_channel.connect()

    if voice_client is None:
        await ctx.send("Не удалось подключиться к голосовому каналу.")
        return

    if voice_client.recording:
        await ctx.send("Я уже слушаю этот голосовой канал.")
        return

    async def detection_callback(user_id: int, word: str, transcript: str) -> None:
        member = ctx.guild.get_member(user_id) if ctx.guild else None
        user_mention = member.mention if member else f"<@{user_id}>"
        if member:
            display = member.global_name or member.display_name
            user_tag = str(member)
            if display and display not in user_tag:
                user_tag = f"{display} / {user_tag}"
        else:
            user_tag = f"ID {user_id}"

        voice = ctx.voice_client.channel if ctx.voice_client else voice_channel
        await _send_detection_message(
            ctx.channel,
            voice,
            user_mention=user_mention,
            user_tag=user_tag,
            word=word,
            transcript=transcript,
        )

    sink = KeywordDetectSink(
        loop=bot.loop,
        target_words=TARGET_WORDS,
        on_detect=detection_callback,
        cooldown=DETECTION_COOLDOWN,
        context_seconds=DETECTION_CONTEXT_SECONDS,
    )

    voice_client.start_recording(sink, _on_recording_stopped, ctx)
    active_sinks[ctx.guild.id] = sink
    await ctx.send(f"Подключился к {voice_channel.name} и начал слушать.")


@bot.command(name="leave")
async def leave(ctx: commands.Context) -> None:
    if ctx.voice_client is None:
        await ctx.send("Я не подключен к голосовому каналу.")
        return

    if ctx.voice_client.recording:
        ctx.voice_client.stop_recording()

    await ctx.voice_client.disconnect(force=True)
    await ctx.send("Отключился от голосового канала.")


@bot.command(name="stop")
async def stop(ctx: commands.Context) -> None:
    voice_client = ctx.voice_client
    if not voice_client or not voice_client.recording:
        await ctx.send("Я сейчас не записываю голосовой канал.")
        return

    voice_client.stop_recording()
    await ctx.send("Остановил распознавание, но остаюсь в канале. Используйте !leave для отключения.")


@bot.event
async def on_ready() -> None:
    logger.info("Бот подключился как %s (ID: %s)", bot.user, bot.user.id)


def main() -> None:
    try:
        bot.run(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("Останавливаю бота по запросу пользователя.")


if __name__ == "__main__":
    main()
