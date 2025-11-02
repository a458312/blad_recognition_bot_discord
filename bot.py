import asyncio
import json
import logging
import os
import time
import struct
import uuid
import wave
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

import discord
from discord.ext import commands
from discord.sinks import Filters, Sink
import numpy as np
from dotenv import load_dotenv


load_dotenv()

SPEECH_BACKEND = os.getenv("SPEECH_BACKEND", "vosk").strip().lower()
KaldiRecognizer = None
VoskModel = None
WhisperModel = None

if SPEECH_BACKEND == "vosk":
    try:
        from vosk import KaldiRecognizer as _KaldiRecognizer, Model as _VoskModel
    except ImportError as exc:  # pragma: no cover - environment validation
        raise SystemExit(
            "Vosk backend selected but 'vosk' package is not installed. "
            "Install it with 'pip install vosk' or switch SPEECH_BACKEND."
        ) from exc
    KaldiRecognizer = _KaldiRecognizer
    VoskModel = _VoskModel
elif SPEECH_BACKEND == "whisper":
    try:
        from faster_whisper import WhisperModel as _WhisperModel
    except ImportError as exc:  # pragma: no cover - environment validation
        raise SystemExit(
            "Whisper backend selected but 'faster-whisper' is not installed. "
            "Install it with 'pip install faster-whisper'."
        ) from exc
    WhisperModel = _WhisperModel
else:  # pragma: no cover - config validation
    raise SystemExit(
        f"Unsupported SPEECH_BACKEND='{SPEECH_BACKEND}'. Use 'vosk' or 'whisper'."
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("blad-bot")

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

RECOGNIZER_SAMPLE_RATE = 16_000  # Hz, expected input sample rate for the recognizer.

if SPEECH_BACKEND == "vosk":
    VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "./model")
    VOSK_MODEL_PATH = Path(VOSK_MODEL_PATH).expanduser().resolve()
    if not VOSK_MODEL_PATH.exists():
        raise SystemExit(
            f"Vosk model directory '{VOSK_MODEL_PATH}' is missing. "
            "Download a model (e.g. Russian: vosk-model-small-ru-0.22) "
            "and unpack it into this path."
        )

    try:
        VOSK_MODEL = VoskModel(str(VOSK_MODEL_PATH))
    except Exception as exc:  # pragma: no cover - dependency failure
        raise SystemExit(f"Не удалось загрузить модель Vosk: {exc}") from exc

    def _create_vosk_recognizer() -> KaldiRecognizer:
        recognizer = KaldiRecognizer(VOSK_MODEL, RECOGNIZER_SAMPLE_RATE)
        recognizer.SetWords(True)
        return recognizer

elif SPEECH_BACKEND == "whisper":
    WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "small")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
    WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE") or None
    WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
    WHISPER_WINDOW_SECONDS = float(os.getenv("WHISPER_WINDOW_SECONDS", "2.5"))
    WHISPER_MAX_BUFFER_SECONDS = float(os.getenv("WHISPER_MAX_BUFFER_SECONDS", "8.0"))

    try:
        WHISPER_MODEL = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    except Exception as exc:  # pragma: no cover - dependency failure
        raise SystemExit(f"Не удалось загрузить модель Whisper: {exc}") from exc

VOICE_SAMPLING_ENABLED = (
    os.getenv("VOICE_SAMPLING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
)
VOICE_SAMPLE_DIR = Path(os.getenv("VOICE_SAMPLE_DIR", "./voice_samples")).expanduser().resolve()
try:
    VOICE_SAMPLE_SECONDS = float(os.getenv("VOICE_SAMPLE_SECONDS", "5.0"))
except ValueError:
    VOICE_SAMPLE_SECONDS = 5.0
if VOICE_SAMPLE_SECONDS <= 0:
    VOICE_SAMPLE_SECONDS = 5.0
VOICE_SAMPLE_THRESHOLD_BYTES = int(RECOGNIZER_SAMPLE_RATE * 2 * VOICE_SAMPLE_SECONDS)
VOICE_SAMPLE_THRESHOLD_BYTES = max(VOICE_SAMPLE_THRESHOLD_BYTES, RECOGNIZER_SAMPLE_RATE * 2)
if VOICE_SAMPLING_ENABLED:
    VOICE_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
else:
    VOICE_SAMPLE_THRESHOLD_BYTES = 0

# Discord bot setup.
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents)
logger.info("Используется движок распознавания: %s", SPEECH_BACKEND.upper())
if VOICE_SAMPLING_ENABLED:
    logger.info("Сбор голосовых семплов включен. Каталог: %s, длительность файла: %.1f с", VOICE_SAMPLE_DIR, VOICE_SAMPLE_SECONDS)
else:
    logger.info("Сбор голосовых семплов выключен. VOICE_SAMPLING_ENABLED=false")


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
        self._last_trigger: dict[int, float] = {}
        self.backend = SPEECH_BACKEND
        self.sample_buffers: dict[int, bytearray] = {}

        if self.backend == "vosk":
            self._recognizers: dict[int, KaldiRecognizer] = {}
        else:
            self.whisper_model = WHISPER_MODEL
            self.whisper_language = WHISPER_LANGUAGE
            self.whisper_beam_size = WHISPER_BEAM_SIZE
            self._buffers: dict[int, deque[np.ndarray]] = {}
            self._pending_tasks: dict[int, asyncio.Future] = {}
            self._min_window_samples = int(
                RECOGNIZER_SAMPLE_RATE * max(self.context_seconds, WHISPER_WINDOW_SECONDS)
            )
            self._max_buffer_samples = int(
                RECOGNIZER_SAMPLE_RATE * max(
                    WHISPER_MAX_BUFFER_SECONDS, self.context_seconds * 3
                )
            )

    @Filters.container  # type: ignore[attr-defined]
    def write(self, data: bytes, user: Any) -> None:
        try:
            user_id = int(user)
        except (TypeError, ValueError):
            return

        pcm = self._convert_and_resample(data)
        if not pcm:
            return

        self._collect_samples(user_id, pcm)

        if self.backend == "vosk":
            self._write_vosk(user_id, pcm)
        else:
            pcm_array = np.frombuffer(pcm, dtype=np.int16)
            self._write_whisper(user_id, pcm_array)

    def cleanup(self) -> None:
        # Override to avoid keeping audio buffers alive.
        self.finished = True
        if VOICE_SAMPLING_ENABLED:
            for user_id in list(self.sample_buffers.keys()):
                self._flush_sample_buffer(user_id, force=True)

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

    def _write_vosk(self, user_id: int, pcm: bytes) -> None:
        recognizer = self._recognizers.setdefault(user_id, _create_vosk_recognizer())
        try:
            if recognizer.AcceptWaveform(pcm):
                result = json.loads(recognizer.Result())
                self._handle_vosk_result(user_id, recognizer, result)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Ошибка распознавания речи: %s", exc)

    def _write_whisper(self, user_id: int, pcm_array: np.ndarray) -> None:
        buffer = self._buffers.setdefault(user_id, deque())
        buffer.append(pcm_array)
        total_samples = sum(chunk.size for chunk in buffer)
        while total_samples > self._max_buffer_samples and buffer:
            removed = buffer.popleft()
            total_samples -= removed.size

        if (
            total_samples >= self._min_window_samples
            and user_id not in self._pending_tasks
        ):
            audio = np.concatenate(list(buffer))
            future = asyncio.run_coroutine_threadsafe(
                self._analyse_whisper(user_id, audio), self.loop
            )
            self._pending_tasks[user_id] = future

            def _task_done(fut: asyncio.Future, *, uid: int) -> None:
                self._pending_tasks.pop(uid, None)
                if fut.cancelled():
                    return
                exc = fut.exception()
                if exc:
                    logger.error("Ошибка распознавания (whisper): %s", exc)

            future.add_done_callback(lambda fut, uid=user_id: _task_done(fut, uid=uid))

    async def _analyse_whisper(self, user_id: int, audio: np.ndarray) -> None:
        loop = asyncio.get_running_loop()
        audio_f32 = audio.astype(np.float32) / 32768.0

        def _run_transcribe() -> list[Any]:
            segment_iter, _ = self.whisper_model.transcribe(
                audio_f32,
                beam_size=self.whisper_beam_size,
                language=self.whisper_language,
                vad_filter=True,
                word_timestamps=True,
            )
            return list(segment_iter)

        try:
            segments = await loop.run_in_executor(None, _run_transcribe)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Ошибка запуска Whisper: %s", exc)
            self._retain_tail(user_id)
            return
        if not segments:
            self._retain_tail(user_id)
            return

        words: list[dict[str, Any]] = []
        transcript_parts: list[str] = []
        for segment in segments:
            if segment.text:
                transcript_parts.append(segment.text.strip())
            segment_words = getattr(segment, "words", None)
            if segment_words:
                for word in segment.words:
                    token = (word.word or "").strip()
                    if not token:
                        continue
                    words.append(
                        {
                            "word": token,
                            "start": word.start if word.start is not None else segment.start,
                            "end": word.end if word.end is not None else segment.end,
                        }
                    )
            else:
                words.append(
                    {
                        "word": segment.text or "",
                        "start": segment.start,
                        "end": segment.end,
                    }
                )

        transcript = " ".join(part for part in transcript_parts if part)
        if transcript:
            logger.info("Whisper transcript (%s): %s", user_id, transcript)
        match = self._find_match(words, transcript)
        if match is None:
            self._retain_tail(user_id)
            return

        matched_word, start, end = match
        now = time.perf_counter()
        if now - self._last_trigger.get(user_id, 0.0) < self.cooldown:
            self._retain_tail(user_id, keep_recent=True)
            return

        snippet = self._build_context_snippet(words, start, end, transcript)
        self._last_trigger[user_id] = now
        try:
            self._flush_sample_buffer(user_id, force=True)
            await self.on_detect(
                user_id, matched_word, snippet or transcript or matched_word
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Ошибка обработки детекта: %s", exc)
        self._buffers[user_id].clear()

    def _retain_tail(self, user_id: int, *, keep_recent: bool = False) -> None:
        buffer = self._buffers.get(user_id)
        if not buffer:
            return
        if not keep_recent:
            buffer.clear()
            return
        tail_samples = self._min_window_samples
        concatenated = np.concatenate(list(buffer))
        if concatenated.size <= tail_samples:
            buffer.clear()
            buffer.append(concatenated)
            return
        tail = concatenated[-tail_samples:]
        buffer.clear()
        buffer.append(tail)

    def _handle_vosk_result(
        self, user_id: int, recognizer: KaldiRecognizer, result: dict[str, Any]
    ) -> None:
        transcript = (result.get("text") or "").strip()
        words = result.get("result") or []
        if transcript:
            logger.info("Vosk transcript (%s): %s", user_id, transcript)

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
        self._flush_sample_buffer(user_id, force=True)
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

    def _collect_samples(self, user_id: int, pcm: bytes) -> None:
        if not VOICE_SAMPLING_ENABLED or not pcm:
            return
        buffer = self.sample_buffers.setdefault(user_id, bytearray())
        buffer.extend(pcm)
        self._flush_sample_buffer(user_id)

    def _flush_sample_buffer(self, user_id: int, force: bool = False) -> None:
        if not VOICE_SAMPLING_ENABLED:
            return
        buffer = self.sample_buffers.get(user_id)
        if not buffer:
            return
        if VOICE_SAMPLE_THRESHOLD_BYTES <= 0:
            chunk = bytes(buffer)
            if chunk:
                self._write_sample_file(user_id, chunk)
            buffer.clear()
            return
        while len(buffer) >= VOICE_SAMPLE_THRESHOLD_BYTES:
            chunk = bytes(buffer[:VOICE_SAMPLE_THRESHOLD_BYTES])
            del buffer[:VOICE_SAMPLE_THRESHOLD_BYTES]
            self._write_sample_file(user_id, chunk)
        if force and buffer:
            chunk = bytes(buffer)
            buffer.clear()
            self._write_sample_file(user_id, chunk)

    def _write_sample_file(self, user_id: int, chunk: bytes) -> None:
        if not chunk:
            return
        user_dir = VOICE_SAMPLE_DIR / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        filename = f"{timestamp}_{uuid.uuid4().hex[:8]}.wav"
        file_path = user_dir / filename
        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RECOGNIZER_SAMPLE_RATE)
            wf.writeframes(chunk)
        duration = len(chunk) / (RECOGNIZER_SAMPLE_RATE * 2)
        logger.info(
            "Сохранил голосовой семпл пользователя %s: %s (%.2f с)",
            user_id,
            file_path,
            duration,
        )

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
