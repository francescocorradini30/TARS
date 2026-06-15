from elevenlabs import ElevenLabs
from config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

_client: ElevenLabs | None = None


def _get_client() -> ElevenLabs:
    global _client
    if _client is None:
        _client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    return _client


def synthesize(text: str) -> bytes | None:
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        return None
    try:
        chunks = _get_client().text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        return b"".join(chunks)
    except Exception as e:
        msg = str(e)
        if 'quota_exceeded' in msg:
            print("[TTS] ElevenLabs quota exceeded — top up at elevenlabs.io")
        else:
            print(f"[TTS] {type(e).__name__}: {msg[:200]}")
        return None
