"""Knowledge banks: write files and texts, search their passages. No LLM anywhere.

A write stores the original file, records the document as ``pending`` and queues one
``knowledge_ingest`` operation per document. The worker splits the file into passages
(see ``media``), embeds each one with the server's multimodal embedder, stores each
passage's media next to the original, and swaps the passages in at once. Search embeds
the query into the same space and returns the nearest passages.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import asyncpg

from ..config import get_config
from ..engine.db_utils import acquire_with_retry
from ..engine.response_models import KnowledgeDocumentHit
from ..engine.retain import bank_utils
from ..engine.schema import fq_table
from ..engine.storage import bank_storage_prefix, key_segment
from . import media
from .embedders import (
    EmbedInput,
    KnowledgeEmbedder,
    UnsupportedModalityError,
    create_knowledge_embedder,
)
from .language import detect as detect_language
from .store import (
    BankStats,
    DocumentPage,
    KnowledgeDocument,
    KnowledgePassage,
    KnowledgeStore,
    RoleVector,
    SearchFilters,
    SearchHit,
    TimeBucket,
    load_knowledge_store,
)

if TYPE_CHECKING:
    from ..engine.memory_engine import MemoryEngine
    from ..models import RequestContext

logger = logging.getLogger(__name__)

TASK_TYPE = "knowledge_ingest"
KIND = "knowledge"
# Bumped whenever splitting changes what a file becomes, so rewriting an unchanged file re-splits it.
_PIPELINE_VERSION = 9  # 9: PDF page units pack whole lines (as benchmarked)
# How many passages are embedded and stored per round, so a 2h video never sits in memory whole.
_EMBED_ROUND = 16


#: Passages a memory-bank recall returns from its linked knowledge banks. Fixed: a per-bank
#: setting was dropped before launch (nothing tuned it).
LINKED_RECALL_TOP_K = 5


class KnowledgeBankError(Exception):
    """A request that cannot be served, with the HTTP status that says why."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class _Embedding:
    role: str | None  # None: the passage's primary vector
    n: int
    input: EmbedInput


@dataclass
class KnowledgeBankInfo:
    bank_id: str
    created_at: Any
    stats: BankStats


@dataclass
class KnowledgeBankPage:
    banks: list[KnowledgeBankInfo]
    #: Knowledge banks matching the search, before paging.
    total: int


@dataclass
class WriteItem:
    """One document to write: a file's bytes, or a text."""

    data: bytes
    filename: str | None
    content_type: str
    document_id: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class WriteResult:
    document_ids: list[str]
    operation_ids: list[str | None]


@dataclass
class SearchRequest:
    query: str | None
    query_image: bytes | None
    top_k: int
    filters: SearchFilters
    # Text hits come back with this many passages on each side as `context`; hits whose windows
    # overlap in one document are merged into the best one (CHUNKING.md: ±1 was the step that paid).
    neighbors: int = 1
    # Exact text (case-insensitive; a POSIX regular expression when `regex`): passages containing it
    # rank first. Alone, without a query, it ranks documents by how many of their passages match.
    pattern: str | None = None
    regex: bool = False


@dataclass
class SearchResultItem:
    hit: SearchHit
    context: str | None = None
    context_first: int | None = None
    context_last: int | None = None


@dataclass
class KnowledgeConfig:
    """The server's knowledge embedder and the limits a write is held to."""

    provider: str
    model: str
    dimension: int
    media_types: list[str]
    max_file_bytes: int
    max_media_seconds: int
    max_pdf_pages: int
    audio_window_seconds: int
    video_window_seconds: int
    media_overlap_seconds: int
    pdf_page_pixels: int
    text_passage_tokens: int


@dataclass
class DocumentDetail:
    document: KnowledgeDocument
    passages: list[KnowledgePassage]


@dataclass
class OpenedFile:
    stream: AsyncIterator[bytes]
    content_type: str
    filename: str | None = None


@dataclass
class LinkedHit:
    bank_id: str
    hit: SearchHit


# The splitter a media type's typical file goes through (a PDF for "document").
_SPLIT_OF: dict[str, media.Kind] = {
    "document": "pdf",
    "image": "image",
    "audio": "audio",
    "video": "video",
    "text": "text",
}
_REQUIRED_MODALITY: dict[media.Kind, str] = {
    "pdf": "image",
    "image": "image",
    "audio": "audio",
    "video": "video",
    "text": "text",
}


class KnowledgeService:
    def __init__(self, engine: MemoryEngine) -> None:
        self._engine = engine
        self._embedder: KnowledgeEmbedder | None = None
        self._store: KnowledgeStore | None = None

    # --- wiring -----------------------------------------------------------------------------

    @property
    def embedder(self) -> KnowledgeEmbedder:
        if self._embedder is None:
            self._embedder = create_knowledge_embedder(get_config(), self._engine.embeddings)
        return self._embedder

    @property
    def store(self) -> KnowledgeStore:
        if self._store is None:
            self._store = load_knowledge_store(self._engine._get_backend, get_config().knowledge_store_extension)
        return self._store

    async def _ready(self, request_context: RequestContext | None) -> None:
        if request_context is not None:
            await self._engine._authenticate_tenant(request_context)
        if not self.embedder.dimension:
            # A remote model's size is known only from its first answer.
            await self.embedder.embed_query("dimension probe")
        await self.store.ensure_ready(self.embedder.dimension)

    async def describe(self) -> KnowledgeConfig:
        if not self.embedder.dimension:
            await self.embedder.embed_query("dimension probe")
        config = get_config()
        e = self.embedder
        return KnowledgeConfig(
            provider=e.provider,
            model=e.model,
            dimension=e.dimension,
            media_types=[t for t in media.MEDIA_TYPES if _REQUIRED_MODALITY[_SPLIT_OF[t]] in e.modalities],
            max_file_bytes=config.knowledge_max_file_mb * 1024 * 1024,
            max_media_seconds=config.knowledge_max_media_seconds,
            max_pdf_pages=config.knowledge_max_pdf_pages,
            audio_window_seconds=config.knowledge_audio_window_seconds,
            video_window_seconds=config.knowledge_video_window_seconds,
            media_overlap_seconds=config.knowledge_media_overlap_seconds,
            pdf_page_pixels=config.knowledge_pdf_page_pixels,
            text_passage_tokens=config.knowledge_text_passage_tokens,
        )

    # --- banks ------------------------------------------------------------------------------

    async def _kind_of(self, bank_id: str) -> str | None:
        async with acquire_with_retry(await self._engine._get_backend()) as conn:
            return await conn.fetchval(f"SELECT kind FROM {fq_table('banks')} WHERE bank_id = $1", bank_id)

    async def _require_bank(self, bank_id: str) -> None:
        kind = await self._kind_of(bank_id)
        if kind != KIND:
            raise KnowledgeBankError(f"knowledge bank '{bank_id}' not found", 404)

    async def create_bank(self, bank_id: str, request_context: RequestContext) -> KnowledgeBankInfo:
        await self._ready(request_context)
        kind = await self._kind_of(bank_id)
        if kind == "memory":
            raise KnowledgeBankError(f"'{bank_id}' is already a memory bank", 409)
        if kind is None:
            bank_utils.validate_new_bank_id(bank_id)
            validator = self._engine._operation_validator
            if validator:
                from ..extensions import CreateBankContext

                await self._engine._validate_operation(
                    validator.validate_create_bank(CreateBankContext(bank_id=bank_id, request_context=request_context))
                )
            async with acquire_with_retry(await self._engine._get_backend()) as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {fq_table("banks")} (bank_id, name, disposition, mission, internal_id, kind)
                    VALUES ($1, $2, $3::jsonb, '', $4, $5)
                    ON CONFLICT (bank_id) DO NOTHING
                    """,
                    bank_id,
                    bank_id,
                    json.dumps(bank_utils.DEFAULT_DISPOSITION),
                    uuid.uuid4(),
                    KIND,
                )
        return await self.get_bank(bank_id, request_context)

    async def list_banks(
        self, request_context: RequestContext, *, limit: int = 100, offset: int = 0, q: str | None = None
    ) -> KnowledgeBankPage:
        """Newest first; ``q`` matches part of the bank id, case-insensitively. Counts are read for
        the page only."""
        await self._ready(request_context)
        async with acquire_with_retry(await self._engine._get_backend()) as conn:
            rows = await conn.fetch(
                f"SELECT bank_id, created_at, count(*) OVER () AS total FROM {fq_table('banks')} "
                "WHERE kind = $1 AND ($2::text IS NULL OR strpos(lower(bank_id), lower($2)) > 0) "
                "ORDER BY created_at DESC, bank_id LIMIT $3 OFFSET $4",
                KIND,
                q or None,
                max(limit, 0),
                max(offset, 0),
            )
            # An offset past the end returns no rows, and with them no window total.
            total = (
                int(rows[0]["total"])
                if rows
                else await conn.fetchval(
                    f"SELECT count(*) FROM {fq_table('banks')} "
                    "WHERE kind = $1 AND ($2::text IS NULL OR strpos(lower(bank_id), lower($2)) > 0)",
                    KIND,
                    q or None,
                )
            )
        stats = await self.store.bank_stats([r["bank_id"] for r in rows])
        return KnowledgeBankPage(
            banks=[KnowledgeBankInfo(r["bank_id"], r["created_at"], stats[r["bank_id"]]) for r in rows], total=total
        )

    async def get_bank(self, bank_id: str, request_context: RequestContext) -> KnowledgeBankInfo:
        await self._ready(request_context)
        async with acquire_with_retry(await self._engine._get_backend()) as conn:
            row = await conn.fetchrow(
                f"SELECT bank_id, created_at FROM {fq_table('banks')} WHERE bank_id = $1 AND kind = $2",
                bank_id,
                KIND,
            )
        if not row:
            raise KnowledgeBankError(f"knowledge bank '{bank_id}' not found", 404)
        stats = await self.store.bank_stats([bank_id])
        return KnowledgeBankInfo(row["bank_id"], row["created_at"], stats[bank_id])

    async def delete_bank(self, bank_id: str, request_context: RequestContext) -> None:
        await self._ready(request_context)
        await self._require_bank(bank_id)
        await self._engine._file_storage.delete_prefix(f"{bank_storage_prefix(bank_id)}kb/")
        # The banks row cascades to kb_documents -> kb_passages, and to its async operations.
        await self._engine.delete_bank(bank_id, request_context=request_context)

    # --- write ------------------------------------------------------------------------------

    def _check_item(self, item: WriteItem) -> media.Kind:
        """How the file will be split; refuses what cannot be written."""
        kind = media.classify(item.content_type, item.filename)
        if kind is None:
            raise KnowledgeBankError(
                f"unsupported file type for '{item.filename}' ({item.content_type}). Knowledge banks take PDF, "
                "images, audio, video, text, Markdown, JSON, CSV, HTML and Office documents.",
                415,
            )
        needed = _REQUIRED_MODALITY[kind]
        if needed not in self.embedder.modalities:
            raise KnowledgeBankError(
                f"'{item.filename}' is {media.media_type_of(kind, item.filename)}, but the knowledge embedder ({self.embedder.provider}) only embeds "
                f"{', '.join(sorted(self.embedder.modalities))}. Configure HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_PROVIDER "
                "with a multimodal model to write it.",
                415,
            )
        limit = get_config().knowledge_max_file_mb * 1024 * 1024
        if len(item.data) > limit:
            raise KnowledgeBankError(
                f"'{item.filename}' is {len(item.data) / 1048576:.1f}MB; the limit is "
                f"{get_config().knowledge_max_file_mb}MB (HINDSIGHT_API_KNOWLEDGE_MAX_FILE_MB)",
                413,
            )
        if not item.data:
            raise KnowledgeBankError(f"'{item.filename}' is empty")
        return kind

    def _content_hash(self, item: WriteItem, kind: str) -> str:
        # The splitting settings are part of the hash: rewriting the same file after they change
        # must re-split it, not be skipped as unchanged.
        c = get_config()
        settings = (
            f"v{_PIPELINE_VERSION}|{kind}|{self.embedder.provider}:{self.embedder.model}:{self.embedder.dimension}|{c.knowledge_pdf_page_pixels}"
            f"|{c.knowledge_image_tokens}|{c.knowledge_audio_window_seconds}|{c.knowledge_video_window_seconds}"
            f"|{c.knowledge_media_overlap_seconds}|{c.knowledge_pdf_text_weight}|{c.knowledge_pdf_bands}|{c.knowledge_text_passage_tokens}"
        )
        return hashlib.sha256(item.data + settings.encode()).hexdigest()

    async def write(self, bank_id: str, items: list[WriteItem], request_context: RequestContext) -> WriteResult:
        await self._ready(request_context)
        await self._require_bank(bank_id)
        kinds = [self._check_item(item) for item in items]  # all-or-nothing: validate before storing any
        result = WriteResult(document_ids=[], operation_ids=[])
        for item, kind in zip(items, kinds, strict=True):
            document_id = item.document_id or str(uuid.uuid4())
            item.filename = item.filename or f"{document_id}.md"
            content_hash = self._content_hash(item, kind)
            existing = await self.store.get_document(bank_id, document_id)
            if (
                existing
                and existing.content_hash == content_hash
                and existing.status == "ready"
                and existing.tags == item.tags
            ):
                result.document_ids.append(document_id)
                result.operation_ids.append(None)  # unchanged: nothing to do
                continue
            storage_key = (
                f"{bank_storage_prefix(bank_id)}kb/files/{uuid.uuid4()}/{key_segment(item.filename or document_id)}"
            )
            await self._engine._file_storage.store(
                file_data=item.data,
                key=storage_key,
                metadata={"content_type": item.content_type, "bank_id": bank_id, "document_id": document_id},
            )
            previous_key = await self.store.put_document(
                bank_id,
                KnowledgeDocument(
                    document_id=document_id,
                    filename=item.filename,
                    content_type=item.content_type,
                    media_type=media.media_type_of(kind, item.filename),
                    size_bytes=len(item.data),
                    storage_key=storage_key,
                    content_hash=content_hash,
                    status="pending",
                    error=None,
                    passage_count=existing.passage_count if existing else 0,
                    tags=item.tags,
                ),
            )
            if previous_key and previous_key != storage_key:
                await self._delete_keys([previous_key])
            payload: dict[str, Any] = {"document_id": document_id, "storage_key": storage_key}
            if request_context.tenant_id:
                payload["_tenant_id"] = request_context.tenant_id
            if request_context.api_key_id:
                payload["_api_key_id"] = request_context.api_key_id
            op = await self._engine._submit_async_operation(
                bank_id=bank_id,
                operation_type=TASK_TYPE,
                task_type=TASK_TYPE,
                task_payload=payload,
                result_metadata={"document_id": document_id, "filename": item.filename},
            )
            result.document_ids.append(document_id)
            result.operation_ids.append(str(op["operation_id"]))
        return result

    async def _delete_keys(self, keys: list[str]) -> None:
        for key in keys:
            try:
                await self._engine._file_storage.delete(key)
            except Exception as exc:  # an orphaned blob is harmless; a failed write over it is not
                logger.warning(f"Could not delete knowledge file {key}: {exc}")

    # --- ingest (worker) --------------------------------------------------------------------

    def _drafts(self, document: KnowledgeDocument, data: bytes) -> Iterator[media.PassageDraft]:
        c = get_config()
        kind = media.classify(document.content_type, document.filename)
        if kind == "pdf":
            return media.split_pdf(
                data,
                page_pixels=c.knowledge_pdf_page_pixels,
                max_pages=c.knowledge_max_pdf_pages,
                bands=c.knowledge_pdf_bands,
            )
        if kind == "image":
            return iter(media.split_image(data))
        if kind in ("audio", "video"):
            window = c.knowledge_audio_window_seconds if kind == "audio" else c.knowledge_video_window_seconds
            return media.split_media(
                data,
                kind,
                filename=document.filename,
                window_seconds=window,
                overlap_seconds=c.knowledge_media_overlap_seconds,
                max_seconds=c.knowledge_max_media_seconds,
            )
        text = media.read_text(data, document.filename)
        return iter(media.split_text(text, tokens=c.knowledge_text_passage_tokens))

    def _vectors_to_embed(
        self, draft: media.PassageDraft, document: KnowledgeDocument, title: str | None
    ) -> list[_Embedding]:
        """The vectors one passage is stored as. A PDF page (PDF_GEMMA.md): the whole page image at 560
        vision tokens with the file name, three overlapping horizontal bands at 280, the page text and its
        128-token units, all titled with the file name. Anything else: one vector, with the asset's name and
        tags in the same input (GAME_ASSETS.md)."""
        if draft.page is not None and document.media_type == "document" and draft.modality == "image":
            out = [
                _Embedding(
                    None,
                    0,
                    EmbedInput(
                        modality="image",
                        data=draft.data,
                        mime=draft.mime,
                        title=title,
                        image_tokens=media.PDF_PAGE_IMAGE_TOKENS,
                    ),
                )
            ]
            out += [
                _Embedding(
                    "band",
                    n,
                    EmbedInput(modality="image", data=band, mime="image/png", image_tokens=media.PDF_BAND_IMAGE_TOKENS),
                )
                for n, band in enumerate(draft.bands)
            ]
            if draft.embed_text and "text" in self.embedder.modalities:
                out.append(_Embedding("page_text", 0, EmbedInput(modality="text", text=draft.embed_text, title=title)))
                units = media.pack_lines(draft.embed_text, tokens=media.PDF_TEXT_UNIT_TOKENS)
                out += [
                    _Embedding("unit", n, EmbedInput(modality="text", text=u, title=title)) for n, u in enumerate(units)
                ]
            return out
        return [
            _Embedding(
                None,
                0,
                EmbedInput(
                    modality=draft.modality,
                    text=draft.text,
                    data=draft.data,
                    mime=draft.mime,
                    title=title,
                    tags=tuple(document.tags) if draft.modality != "text" else (),
                ),
            )
        ]

    async def ingest(self, bank_id: str, document_id: str, storage_key: str, *, retries_left: bool = False) -> None:
        await self._ready(None)
        document = await self.store.get_document(bank_id, document_id)
        if document is None or document.storage_key != storage_key:
            logger.info(f"Knowledge ingest skipped: {bank_id}/{document_id} was replaced or deleted")
            return
        await self.store.set_status(bank_id, document_id, "processing")
        started = time.perf_counter()
        media_prefix = f"{bank_storage_prefix(bank_id)}kb/media/{key_segment(document_id)}/{uuid.uuid4()}/"
        stored_keys: list[str] = []
        try:
            data = await self._engine._file_storage.retrieve(storage_key)
            passages: list[KnowledgePassage] = []
            batch: list[media.PassageDraft] = []
            title = _title_hint(document)
            doc: KnowledgeDocument = document  # narrowed for the closure below

            async def flush() -> None:
                plans = [self._vectors_to_embed(d, doc, title) for d in batch]
                flat = [e for plan in plans for e in plan]
                embedded = await self.embedder.embed_documents([e.input for e in flat])
                cursor = 0
                vectors: list[list[float]] = []
                extras: list[list[RoleVector]] = []
                for plan in plans:
                    mine = embedded[cursor : cursor + len(plan)]
                    cursor += len(plan)
                    vectors.append(mine[0])
                    extras.append(
                        [RoleVector(e.role, e.n, v) for e, v in zip(plan[1:], mine[1:], strict=True) if e.role]
                    )
                for draft, vector, extra in zip(batch, vectors, extras, strict=True):
                    index = len(passages)
                    key = None
                    if draft.data is not None:
                        key = f"{media_prefix}{index}.{draft.extension}"
                        await self._engine._file_storage.store(
                            file_data=draft.data, key=key, metadata={"content_type": draft.mime or ""}
                        )
                        stored_keys.append(key)
                    passages.append(
                        KnowledgePassage(
                            index=index,
                            modality=draft.modality,
                            text=draft.text,
                            page=draft.page,
                            start_ms=draft.start_ms,
                            end_ms=draft.end_ms,
                            media_key=key,
                            embedding=vector,
                            extra_vectors=extra,
                        )
                    )
                batch.clear()

            # Splitting is blocking (pdfium, ffmpeg); it runs in a thread, one draft at a time.
            drafts = await asyncio.to_thread(self._drafts, document, data)
            try:
                while (draft := await asyncio.to_thread(_next_draft, drafts)) is not None:
                    batch.append(draft)
                    if len(batch) >= _EMBED_ROUND:
                        await flush()
            finally:
                # Close the splitter now, off the event loop, rather than whenever it is collected.
                await asyncio.to_thread(_close_drafts, drafts)
            if batch:
                await flush()
            if not passages:
                raise media.MediaError("the file holds nothing to search")
            current = await self.store.get_document(bank_id, document_id)
            if current is None or current.storage_key != storage_key:
                # Replaced or deleted while we worked: our passages belong to nobody.
                await self._delete_keys(stored_keys)
                return
            language = detect_language("\n".join(p.text or "" for p in passages))
            stale = await self.store.replace_passages(bank_id, document_id, passages, language)
            await self._delete_keys(stale)
            logger.info(
                f"Knowledge ingest {bank_id}/{document_id}: {len(passages)} {document.media_type} passages "
                f"in {time.perf_counter() - started:.1f}s"
            )
        except Exception as exc:
            await self._delete_keys(stored_keys)
            permanent = isinstance(exc, ValueError)  # a MediaError or UnsupportedModalityError: same next time
            message = str(exc) if permanent else repr(exc)
            # A transient error (the embedder unreachable) is retried by the worker: until the retries
            # run out the document is still on its way, so it stays pending, with the error noted.
            # It used to be marked failed at once, and clients waiting for ready/failed gave up on it.
            if retries_left and not permanent:
                await self.store.set_status(bank_id, document_id, "pending", f"retrying after: {message}"[:2000])
            else:
                await self.store.set_status(bank_id, document_id, "failed", message[:2000])
            raise

    # --- read -------------------------------------------------------------------------------

    async def list_documents(
        self,
        bank_id: str,
        request_context: RequestContext,
        *,
        limit: int,
        offset: int,
        query: str | None,
        status: str | None,
        media_type: str | None,
        tags: list[str] | None,
    ) -> DocumentPage:
        await self._ready(request_context)
        await self._require_bank(bank_id)
        return await self.store.list_documents(
            bank_id, limit=limit, offset=offset, query=query, status=status, media_type=media_type, tags=tags
        )

    async def get_document(self, bank_id: str, document_id: str, request_context: RequestContext) -> DocumentDetail:
        await self._ready(request_context)
        await self._require_bank(bank_id)
        document = await self.store.get_document(bank_id, document_id)
        if document is None:
            raise KnowledgeBankError(f"document '{document_id}' not found", 404)
        return DocumentDetail(document=document, passages=await self.store.list_passages(bank_id, document_id))

    async def delete_document(self, bank_id: str, document_id: str, request_context: RequestContext) -> None:
        await self._ready(request_context)
        await self._require_bank(bank_id)
        keys = await self.store.media_keys(bank_id, document_id)
        deleted = await self.store.delete_document(bank_id, document_id)
        if deleted is None:
            raise KnowledgeBankError(f"document '{document_id}' not found", 404)
        await self._delete_keys([*keys, *([deleted.storage_key] if deleted.storage_key else [])])

    async def open_file(self, bank_id: str, document_id: str, request_context: RequestContext) -> OpenedFile:
        """The original upload: (byte stream, content type, filename)."""
        await self._ready(request_context)
        await self._require_bank(bank_id)
        document = await self.store.get_document(bank_id, document_id)
        if document is None or not document.storage_key:
            raise KnowledgeBankError(f"document '{document_id}' not found", 404)
        stream = self._engine._file_storage.retrieve_stream(document.storage_key)
        return OpenedFile(stream=stream, content_type=document.content_type, filename=document.filename)

    async def open_passage_media(
        self, bank_id: str, document_id: str, index: int, request_context: RequestContext
    ) -> OpenedFile:
        await self._ready(request_context)
        await self._require_bank(bank_id)
        passage = await self.store.get_passage(bank_id, document_id, index)
        if passage is None or not passage.media_key:
            raise KnowledgeBankError("this passage has no media", 404)
        ext = passage.media_key.rsplit(".", 1)[-1]
        content_type = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp", "m4a": "audio/mp4"}.get(
            ext, "video/mp4" if ext == "mp4" else "application/octet-stream"
        )
        return OpenedFile(
            stream=self._engine._file_storage.retrieve_stream(passage.media_key), content_type=content_type
        )

    async def timeseries(
        self, bank_id: str, period: str, trunc: str, request_context: RequestContext
    ) -> list[TimeBucket]:
        await self._ready(request_context)
        await self._require_bank(bank_id)
        days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}[period]
        since = datetime.now(UTC) - timedelta(days=days)
        return await self.store.timeseries(bank_id, since=since, trunc=trunc)

    # --- search -----------------------------------------------------------------------------

    async def search(
        self, bank_id: str, request: SearchRequest, request_context: RequestContext
    ) -> list[SearchResultItem]:
        await self._ready(request_context)
        await self._require_bank(bank_id)
        return await self._with_context(bank_id, await self._search(bank_id, request), request.neighbors)

    async def _search(self, bank_id: str, request: SearchRequest) -> list[SearchHit]:
        pattern = request.pattern if (request.pattern or "").strip() else None
        has_query = bool((request.query or "").strip()) or request.query_image is not None
        if not has_query and pattern is None:
            raise KnowledgeBankError("search needs a query, a query_image or a pattern")
        try:
            if not has_query and pattern is not None:
                return await self.store.grep(
                    bank_id, pattern, top_k=request.top_k, filters=request.filters, regex=request.regex
                )
            try:
                vector = await self.embedder.embed_query(request.query, request.query_image)
            except UnsupportedModalityError as exc:
                raise KnowledgeBankError(str(exc), 415) from exc
            return await self.store.search(
                bank_id,
                vector,
                top_k=request.top_k,
                filters=request.filters,
                text_weight=get_config().knowledge_pdf_text_weight,
                query_text=request.query,
                keyword_weight=get_config().knowledge_keyword_weight,
                pattern=pattern,
                regex=request.regex,
            )
        except asyncpg.InvalidRegularExpressionError as exc:
            raise KnowledgeBankError(f"invalid regular expression: {exc}") from exc

    async def _with_context(self, bank_id: str, hits: list[SearchHit], neighbors: int) -> list[SearchResultItem]:
        """Expand text hits to their neighbouring passages, merging overlapping windows per document."""
        items: list[SearchResultItem] = []
        windows: dict[str, list[SearchResultItem]] = {}
        for hit in hits:
            if neighbors <= 0 or hit.passage.modality != "text":
                items.append(SearchResultItem(hit))
                continue
            first, last = max(0, hit.passage.index - neighbors), hit.passage.index + neighbors
            same_doc = windows.setdefault(hit.document.document_id, [])
            merged = next(
                (w for w in same_doc if first <= (w.context_last or 0) + 1 and last >= (w.context_first or 0) - 1),
                None,
            )
            if merged is not None:  # a better hit already covers (or touches) this one
                merged.context_first = min(merged.context_first or 0, first)
                merged.context_last = max(merged.context_last or 0, last)
                continue
            item = SearchResultItem(hit, context_first=first, context_last=last)
            same_doc.append(item)
            items.append(item)
        for item in items:
            if item.context_first is None or item.context_last is None:
                continue
            passages = await self.store.passage_range(
                bank_id, item.hit.document.document_id, item.context_first, item.context_last
            )
            item.context = "\n\n".join(p.text or "" for p in passages)
            item.context_first = passages[0].index if passages else item.context_first
            item.context_last = passages[-1].index if passages else item.context_last
        return items

    async def search_linked(
        self, bank_ids: list[str], query: str, top_k: int, request_context: RequestContext
    ) -> list[LinkedHit]:
        """Search the knowledge banks a memory bank links, best passages first across all of them."""
        await self._ready(request_context)
        hits: list[LinkedHit] = []
        vector: list[float] | None = None
        for bank_id in bank_ids:
            if await self._kind_of(bank_id) != KIND:
                logger.warning(f"Linked knowledge bank '{bank_id}' does not exist; skipped")
                continue
            if vector is None:
                vector = await self.embedder.embed_query(query)
            for hit in await self.store.search(
                bank_id,
                vector,
                top_k=top_k,
                filters=SearchFilters(),
                # no keyword arm: fused scores are rescaled per bank and could not be ranked across banks
                text_weight=get_config().knowledge_pdf_text_weight,
            ):
                hits.append(LinkedHit(bank_id=bank_id, hit=hit))
        hits.sort(key=lambda linked: linked.hit.score, reverse=True)
        return hits[:top_k]


async def documents_for_memory_bank(
    engine: MemoryEngine, bank_id: str, query: str, request_context: RequestContext
) -> list[KnowledgeDocumentHit] | None:
    """The linked knowledge banks' best passages for a memory-bank recall, or None when none are linked.

    A failure here never fails the recall: the memories are still the answer, without the docs.
    """
    if not get_config().enable_knowledge_banks:
        return None
    config = await engine._config_resolver.get_bank_config(bank_id, request_context)
    linked = [b for b in (config.get("knowledge_banks") or []) if b != bank_id]
    if not linked or not (query or "").strip():
        return None
    try:
        hits = await get_knowledge_service(engine).search_linked(linked, query, LINKED_RECALL_TOP_K, request_context)
    except Exception as exc:
        logger.warning(f"Knowledge search for memory bank '{bank_id}' failed: {exc!r}")
        return []
    from .routes import search_result

    out = []
    for linked in hits:
        r = search_result(linked.bank_id, linked.hit)
        out.append(
            KnowledgeDocumentHit(
                bank_id=linked.bank_id,
                document_id=r.document_id,
                passage_index=r.index,
                filename=r.filename,
                media_type=r.media_type,
                text=r.text,
                page=r.page,
                start_ms=r.start_ms,
                end_ms=r.end_ms,
                score=r.score,
                passage_url=r.passage_url,
                document_url=r.document_url,
                tags=r.tags,
            )
        )
    return out


def _title_hint(document: KnowledgeDocument) -> str | None:
    """The file name without its extension: the embedder's "title: ..." prompt slot. None for the name
    a text write gets when it has none (``<document_id>.md``): an id, often a UUID, is not a title. A
    file named like its id (``report.pdf`` as ``report``) keeps it: the title is worth 2-10 nDCG@10."""
    if not document.filename or document.filename == f"{document.document_id}.md":
        return None
    stem = Path(document.filename).stem
    return re.sub(r"[_\-]+", " ", stem).strip()  # "footstep_gravel_03" reads as words


def _close_drafts(drafts: Iterator[media.PassageDraft]) -> None:
    close = getattr(drafts, "close", None)  # generators release the PDF / media they hold
    if close is not None:
        close()


def _next_draft(drafts: Iterator[media.PassageDraft]) -> media.PassageDraft | None:
    return next(drafts, None)


_services: weakref.WeakKeyDictionary[MemoryEngine, KnowledgeService] = weakref.WeakKeyDictionary()


def get_knowledge_service(engine: MemoryEngine) -> KnowledgeService:
    """One service per engine (one embedder, loaded once), created on first use."""
    service = _services.get(engine)
    if service is None:
        service = _services[engine] = KnowledgeService(engine)
    return service


async def run_ingest_task(engine: MemoryEngine, task: dict[str, Any]) -> None:
    retries_left = task.get("_retry_count", 0) < get_config().worker_max_retries  # see execute_task's retry
    await get_knowledge_service(engine).ingest(
        task["bank_id"], task["document_id"], task["storage_key"], retries_left=retries_left
    )
