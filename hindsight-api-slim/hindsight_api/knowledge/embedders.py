"""Multimodal embedders for knowledge banks: one embedding space for text, images, audio and video.

One embedder per server (``HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_PROVIDER``):

* ``local`` — EmbeddingGemma 2 in-process (Apache-2.0, 768 dims, text/image/audio/video).
* ``gemini`` — Gemini Embedding 2 over the API (text/image/audio/video).
* ``memory`` — the memory banks' own text embedder. Text-only, so non-text writes are refused.

Every passage reaches the embedder as bytes plus a MIME type (a PNG page, an m4a window, an
mp4 window) so an API provider can send them as they are and a local one decodes them.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from . import media

if TYPE_CHECKING:
    from ..config import ConfigLike
    from ..engine.embeddings import Embeddings

logger = logging.getLogger(__name__)

Modality = Literal["text", "image", "audio", "video"]
ALL_MODALITIES: tuple[Modality, ...] = ("text", "image", "audio", "video")

# EmbeddingGemma 2's prompts (model card). Gemini Embedding 2 reads the same instruction style.
_QUERY_PREFIX = "task: search result | query: "


def _document_text(text: str, title: str | None) -> str:
    return f"title: {title or 'none'} | text: {text}"


@dataclass(frozen=True)
class EmbedInput:
    """One passage to embed: text, or media bytes with their MIME type."""

    modality: Modality
    text: str | None = None
    data: bytes | None = None
    mime: str | None = None
    title: str | None = None
    tags: tuple[str, ...] = ()
    # Vision tokens for this image (70..1120); None: the server default. A PDF page wants more than a band.
    image_tokens: int | None = None


def _media_prompt(item: EmbedInput, placeholders: str) -> str | None:
    """`title: <name> | tags: <tags> | text: <media>` — the asset's name and tags in the same input as
    the media. On 5.7k game assets this took R@1 from 12 to 58, and beat the name or the media alone
    on human-captioned sounds (GAME_ASSETS.md). None when there is nothing to add."""
    if not item.title and not item.tags:
        return None
    tags = f" | tags: {', '.join(item.tags)}" if item.tags else ""
    return f"title: {item.title or 'none'}{tags} | text: {placeholders}"


class KnowledgeEmbedder(ABC):
    provider: str
    model: str
    dimension: int
    modalities: frozenset[Modality]

    @abstractmethod
    async def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
        """One vector per passage, in order."""

    @abstractmethod
    async def embed_query(self, text: str | None, image: bytes | None = None) -> list[float]:
        """The vector a search compares passages against."""


class LocalGemmaEmbedder(KnowledgeEmbedder):
    """EmbeddingGemma 2 through sentence-transformers, loaded on first use."""

    provider = "local"
    modalities = frozenset(ALL_MODALITIES)

    def __init__(
        self,
        model: str,
        *,
        dimensions: int | None,
        device: str | None,
        batch_size: int,
        image_tokens: int,
    ) -> None:
        self.model = model
        self.dimension = dimensions or 768
        self._truncate = dimensions
        self._device = device
        self._batch_size = batch_size
        self._image_tokens = image_tokens
        self._st: Any = None
        # The model is not safe to call from two threads at once (MPS/CUDA state), and it is
        # big enough that one copy per process is the budget.
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._st is None:
            import torch
            from sentence_transformers import SentenceTransformer

            # float16 returns NaN embeddings on this model (model card); bfloat16 only on CUDA.
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            st = SentenceTransformer(self.model, device=self._device, model_kwargs={"torch_dtype": dtype})
            processor = st[0].processor
            processor.image_processor.max_soft_tokens = self._image_tokens
            processor.image_processor.image_seq_length = self._image_tokens
            native = st.get_sentence_embedding_dimension()
            if native and not self._truncate:
                self.dimension = native
            self._st = st
            logger.info(f"Knowledge embedder loaded: {self.model} on {st.device} ({self.dimension} dims)")
        return self._st

    def _model_input(self, item: EmbedInput) -> Any:
        if item.modality == "text":
            return _document_text(item.text or "", item.title)
        assert item.data is not None
        model_input: dict[str, Any]
        if item.modality == "image":
            model_input, placeholders = {"image": media.open_image(item.data)}, "<|image|>"
        elif item.modality == "audio":
            pcm = media.decode_audio_pcm(item.data)
            model_input, placeholders = {"audio": {"array": pcm, "sampling_rate": media.PCM_RATE}}, "<|audio|>"
        else:
            # Picture and soundtrack in one vector, so speech in a clip is findable too.
            model_input, placeholders = {"video": media.decode_video_frames(item.data)}, "<|video|>"
            pcm = media.decode_audio_pcm(item.data, required=False)
            if pcm is not None and pcm.size:
                model_input["audio"] = {"array": pcm, "sampling_rate": media.PCM_RATE}
                placeholders += "<|audio|>"
        prompt = _media_prompt(item, placeholders)
        if prompt:
            model_input["text"] = prompt
        return model_input

    def _encode(self, inputs: list[Any], image_tokens: int | None = None) -> list[list[float]]:
        out: list[list[float]] = []
        # The lock is taken per model batch, not per call, so a search arriving during a long
        # ingest waits for one batch rather than the whole document.
        for start in range(0, len(inputs), self._batch_size):
            with self._lock:
                st = self._load()
                processor = st[0].processor.image_processor
                processor.max_soft_tokens = processor.image_seq_length = image_tokens or self._image_tokens
                vectors = st.encode(
                    inputs[start : start + self._batch_size],
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    truncate_dim=self._truncate,
                    show_progress_bar=False,
                )
                if any(
                    isinstance(i, dict) and ("video" in i or "audio" in i)
                    for i in inputs[start : start + self._batch_size]
                ):
                    # Every clip length allocates differently sized blocks and the caching allocator
                    # keeps them all: video ingest grew past 26GB on an M3 Max without this.
                    _release_device_cache()
            out.extend(v.tolist() for v in vectors)
        return out

    def ensure_loaded(self) -> None:
        with self._lock:
            self._load()

    async def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
        def run() -> list[list[float]]:
            # One encode per vision-token budget: the processor setting applies to a whole batch.
            out: list[list[float] | None] = [None] * len(items)
            for budget in {item.image_tokens for item in items}:
                group = [i for i, item in enumerate(items) if item.image_tokens == budget]
                for i, vector in zip(
                    group, self._encode([self._model_input(items[i]) for i in group], budget), strict=True
                ):
                    out[i] = vector
            return [v for v in out if v is not None]

        return await asyncio.to_thread(run)

    async def embed_query(self, text: str | None, image: bytes | None = None) -> list[float]:
        if image is not None:
            inp: Any = {"image": media.open_image(image)}
            if text:
                inp["text"] = f"{_QUERY_PREFIX}{text} <|image|>"
        else:
            inp = f"{_QUERY_PREFIX}{text or ''}"
        vectors = await asyncio.to_thread(self._encode, [inp])
        return vectors[0]


def _release_device_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class GeminiEmbedder(KnowledgeEmbedder):
    """Gemini Embedding 2: every passage is sent as its own bytes, one content per vector."""

    provider = "gemini"
    modalities = frozenset(ALL_MODALITIES)

    def __init__(self, model: str, *, api_key: str | None, dimensions: int | None, batch_size: int) -> None:
        from google import genai

        self.model = model
        self.dimension = dimensions or 3072
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def _content(self, item: EmbedInput) -> Any:
        from google.genai import types

        if item.modality == "text":
            return types.Content(parts=[types.Part.from_text(text=_document_text(item.text or "", item.title))])
        assert item.data is not None and item.mime is not None
        return types.Content(parts=[types.Part.from_bytes(data=item.data, mime_type=item.mime)])

    async def _embed(self, contents: list[Any]) -> list[list[float]]:
        from google.genai import types

        config = types.EmbedContentConfig(output_dimensionality=self._dimensions) if self._dimensions else None
        out: list[list[float]] = []
        for start in range(0, len(contents), self._batch_size):
            response = await self._client.aio.models.embed_content(
                model=self.model, contents=contents[start : start + self._batch_size], config=config
            )
            for embedding in response.embeddings or []:
                out.append(media.l2_normalize(list(embedding.values or [])))
        if len(out) != len(contents):
            raise RuntimeError(f"Gemini returned {len(out)} embeddings for {len(contents)} inputs")
        return out

    async def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
        return await self._embed([self._content(item) for item in items])

    async def embed_query(self, text: str | None, image: bytes | None = None) -> list[float]:
        from google.genai import types

        parts = []
        if text:
            parts.append(types.Part.from_text(text=f"{_QUERY_PREFIX}{text}"))
        if image is not None:
            parts.append(types.Part.from_bytes(data=image, mime_type=media.sniff_image_mime(image)))
        return (await self._embed([types.Content(parts=parts)]))[0]


_MEDIA_CONCURRENCY = 4  # media requests in flight per embed call (see OpenAIEmbedder.embed_documents)


class OpenAIEmbedder(KnowledgeEmbedder):
    """Any OpenAI-compatible ``/v1/embeddings`` server, e.g. vLLM serving a multimodal model.

    Text goes as ``input``; an image or a video goes as chat-style ``messages`` with a data URL,
    which is how vLLM takes multimodal input on the embeddings route. One request per media
    passage (vLLM embeds one conversation per request), texts batched.
    """

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None,
        dimensions: int | None,
        modalities: frozenset[Modality],
        batch_size: int,
    ) -> None:
        from ..engine.aiohttp_session import LoopLocalSession, per_phase_timeout

        self.model = model
        self.dimension = dimensions or 0  # resolved by the first call when unset
        self._dimensions = dimensions
        self.modalities = modalities
        self._url = base_url.rstrip("/") + "/embeddings"
        self._batch_size = max(1, batch_size)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._session = LoopLocalSession(timeout=per_phase_timeout(300.0, connect=10.0), headers=headers)

    async def _post(self, body: dict[str, Any]) -> list[list[float]]:
        from ..engine.aiohttp_session import raise_for_status
        from ..engine.remote_retry import RetryPolicy, acall_with_retry

        if self._dimensions:
            body["dimensions"] = self._dimensions

        async def call() -> dict[str, Any]:
            async with self._session.get().post(self._url, json={"model": self.model, **body}) as response:
                await raise_for_status(response)
                return await response.json()

        # A dropped connection or a 5xx must not fail a whole document's ingest: retry transient errors
        # with the memory banks' embedding policy, with more budget since ingest is not latency-bound.
        policy = RetryPolicy(max_retries=5, budget_seconds=60.0)
        payload = await acall_with_retry(call, policy=policy, budget=policy.new_budget(), provider="knowledge-openai")
        rows = sorted(payload["data"], key=lambda row: row["index"])
        vectors = [media.l2_normalize(row["embedding"]) for row in rows]
        if vectors and not self.dimension:
            self.dimension = len(vectors[0])
        return vectors

    @staticmethod
    def _data_url(data: bytes, mime: str) -> str:
        import base64

        return f"data:{mime};base64,{base64.b64encode(data).decode()}"

    async def _embed_media(self, item: EmbedInput) -> list[float]:
        if item.modality not in self.modalities:
            raise UnsupportedModalityError(f"{self.model} does not embed {item.modality}")
        assert item.data is not None and item.mime is not None
        part_type = (
            "video_url" if item.modality == "video" else "audio_url" if item.modality == "audio" else "image_url"
        )
        content: list[dict[str, Any]] = [{"type": part_type, part_type: {"url": self._data_url(item.data, item.mime)}}]
        prompt = _media_prompt(item, "")
        if prompt:
            content.insert(0, {"type": "text", "text": prompt})
        body: dict[str, Any] = {"messages": [{"role": "user", "content": content}]}
        if item.image_tokens:
            body["image_tokens"] = item.image_tokens  # honoured by our EmbeddingGemma 2 server; others ignore it
        return (await self._post(body))[0]

    async def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(items)
        texts = [
            (i, _document_text(item.text or "", item.title)) for i, item in enumerate(items) if item.modality == "text"
        ]
        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start : start + self._batch_size]
            for (i, _), vector in zip(chunk, await self._post({"input": [t for _, t in chunk]}), strict=True):
                out[i] = vector
        media_items = [(i, item) for i, item in enumerate(items) if item.modality != "text"]
        # One request per media passage, a few at a time: firing a document's every page at once
        # (times every document ingesting) flooded the server and its proxy, and each reset sent a
        # whole document back to the retry queue. A server batches on its side; more in flight buys nothing.
        for start in range(0, len(media_items), _MEDIA_CONCURRENCY):
            chunk = media_items[start : start + _MEDIA_CONCURRENCY]
            vectors = await asyncio.gather(*(self._embed_media(item) for _, item in chunk))
            for (i, _), vector in zip(chunk, vectors, strict=True):
                out[i] = vector
        return [v for v in out if v is not None]

    async def embed_query(self, text: str | None, image: bytes | None = None) -> list[float]:
        if image is None:
            return (await self._post({"input": [f"{_QUERY_PREFIX}{text or ''}"]}))[0]
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": self._data_url(image, media.sniff_image_mime(image))}}
        ]
        if text:
            content.append({"type": "text", "text": f"{_QUERY_PREFIX}{text}"})
        return (await self._post({"messages": [{"role": "user", "content": content}]}))[0]


class MemoryTextEmbedder(KnowledgeEmbedder):
    """The memory banks' text embedder, for servers that run without a multimodal model."""

    provider = "memory"
    modalities = frozenset({"text"})

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self.model = embeddings.provider_name
        self.dimension = embeddings.dimension

    async def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
        return await self._embeddings.encode_documents([_document_text(i.text or "", i.title) for i in items])

    async def embed_query(self, text: str | None, image: bytes | None = None) -> list[float]:
        if image is not None:
            raise UnsupportedModalityError("image queries need a multimodal knowledge embedder")
        return (await self._embeddings.encode_query([text or ""]))[0]


class UnsupportedModalityError(ValueError):
    """The configured embedder cannot embed this kind of content."""


def create_knowledge_embedder(config: ConfigLike, memory_embeddings: Embeddings) -> KnowledgeEmbedder:
    provider = config.knowledge_embeddings_provider
    if provider == "local":
        from ..config import DEFAULT_KNOWLEDGE_EMBEDDINGS_LOCAL_MODEL

        return LocalGemmaEmbedder(
            config.knowledge_embeddings_model or DEFAULT_KNOWLEDGE_EMBEDDINGS_LOCAL_MODEL,
            dimensions=config.knowledge_embeddings_dimensions,
            device=config.knowledge_embeddings_device,
            batch_size=config.knowledge_embeddings_batch_size,
            image_tokens=config.knowledge_image_tokens,
        )
    if provider == "gemini":
        from ..config import DEFAULT_KNOWLEDGE_EMBEDDINGS_GEMINI_MODEL

        return GeminiEmbedder(
            config.knowledge_embeddings_model or DEFAULT_KNOWLEDGE_EMBEDDINGS_GEMINI_MODEL,
            api_key=config.knowledge_embeddings_api_key or config.embeddings_gemini_api_key,
            dimensions=config.knowledge_embeddings_dimensions,
            batch_size=config.knowledge_embeddings_batch_size,
        )
    if provider == "openai":
        if not config.knowledge_embeddings_base_url or not config.knowledge_embeddings_model:
            raise ValueError(
                "HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_PROVIDER=openai needs "
                "HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_BASE_URL and HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_MODEL"
            )
        modalities = frozenset(
            m for m in ALL_MODALITIES if m in config.knowledge_embeddings_modalities.split(",") or m == "text"
        )
        return OpenAIEmbedder(
            config.knowledge_embeddings_model,
            base_url=config.knowledge_embeddings_base_url,
            api_key=config.knowledge_embeddings_api_key,
            dimensions=config.knowledge_embeddings_dimensions,
            modalities=modalities,
            batch_size=config.knowledge_embeddings_batch_size,
        )
    if provider == "memory":
        return MemoryTextEmbedder(memory_embeddings)
    raise ValueError(
        f"Unknown HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_PROVIDER '{provider}'. Expected local, gemini, openai or memory."
    )
