"""Semantic compression — abstractive/extractive summarization.

Uses HuggingFace transformers for summarization with graceful
fallback to extractive methods when models are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from twotrim.types import StrategyName, StrategyResult

logger = logging.getLogger(__name__)


class SemanticCompressor:
    """Compress text via summarization models."""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        max_summary_ratio: float = 0.5,
        min_summary_ratio: float = 0.3,
        min_input_length: int = 200,
        batch_size: int = 4,
        prefer_extractive: bool = False,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.max_summary_ratio = max_summary_ratio
        self.min_summary_ratio = min_summary_ratio
        self.min_input_length = min_input_length
        self.batch_size = batch_size
        self.prefer_extractive = prefer_extractive
        self.device = device
        self.name = StrategyName.SEMANTIC
        self._model: Any = None
        self._tokenizer: Any = None
        self._available: bool | None = None
        self._resolved_device: str | None = None

    async def _ensure_model(self) -> bool:
        """Lazy-load the summarization model."""
        if self._available is not None:
            return self._available
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device_map = self.device
            if device_map == "auto":
                device_map = "cuda" if torch.cuda.is_available() else "cpu"

            self._resolved_device = device_map
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device_map == "cuda" else torch.float32,
            ).to(device_map)

            self._available = True
            logger.info("Loaded summarization model: %s on %s", self.model_name, device_map)
        except Exception as e:
            logger.warning("Summarization model unavailable, falling back to extractive: %s", e)
            self._available = False
        return self._available

    async def compress(
        self,
        text: str,
        token_counter: object | None = None,
        query: str | None = None,
        target_reduction: float = 0.0,
    ) -> StrategyResult:
        """Compress text using summarization."""
        original_tokens = _count_tokens(text, token_counter)

        # Skip if too short
        if len(text.split()) < self.min_input_length:
            return StrategyResult(
                strategy=self.name,
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=0.0,
                metadata={"skipped": True, "reason": "below_min_length"},
            )

        model_available = await self._ensure_model()

        # Rule 1: We NEVER use abstractive compression if a specific query is given (prevent hallucination for QA/RAG)
        force_extractive = self.prefer_extractive or (query is not None and len(query.strip()) > 0)

        if model_available and self._model is not None and not force_extractive:
            method = "abstractive"
            compressed = await self._abstractive_compress(text, query=query, target_reduction=target_reduction)
        else:
            method = "extractive"
            compressed = await self._extractive_compress(text, query=query, target_reduction=target_reduction)

        compressed_tokens = _count_tokens(compressed, token_counter)

        return StrategyResult(
            strategy=self.name,
            original_text=text,
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=1 - (compressed_tokens / max(original_tokens, 1)),
            metadata={"method": "abstractive" if model_available else "extractive"},
        )

    async def estimate_reduction(self, text: str) -> float:
        """Estimate compression potential."""
        word_count = len(text.split())
        if word_count < self.min_input_length:
            return 0.0
        # Summarization typically achieves 40-60% reduction on longer texts
        return min(0.6, self.max_summary_ratio)

    async def _abstractive_compress(self, text: str, query: str | None = None, target_reduction: float = 0.0) -> str:
        """Use the summarization model for abstractive compression."""
        import asyncio
        import torch

        # Ensure model is ready
        await self._ensure_model()

        # 1. Calculate query length in tokens to determine remaining capacity
        query_prefix = f"As a detailed answer to '{query}', summarize: " if query else ""
        query_ids = self._tokenizer(query_prefix, add_special_tokens=False)["input_ids"]
        query_token_count = len(query_ids)
        
        # 2. BART context limit is 1024 tokens. 
        # We reserve 24 tokens for BOS/EOS and general padding safety.
        max_chunk_tokens = 1000 - query_token_count
        # Ensure we always have at least some room for context
        max_chunk_tokens = max(100, max_chunk_tokens)

        # 3. Precise Token-Aware Split (Pointer #5: Stability)
        chunks = self._token_split_for_model(text, max_tokens=max_chunk_tokens)

        # 4. Inject query into each chunk
        if query:
            chunks = [f"{query_prefix}{c}" for c in chunks]

        def _run_generation() -> list[str]:
            results = []
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]

                # Tokenize
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(self._resolved_device)

                # Ensure truncation didn't slice the actual question part if it's too long
                # We prioritize the first 1024 tokens but BART's max_pos is 1024.

                # Calculate generation bounds for this batch
                # If policy target_reduction is provided, use it if it's more aggressive
                effective_min_ratio = max(self.min_summary_ratio, 1.0 - target_reduction) if target_reduction > 0 else self.min_summary_ratio
                effective_max_ratio = max(self.max_summary_ratio, 1.1 - target_reduction) if target_reduction > 0 else self.max_summary_ratio
                
                batch_max = int(max(len(c.split()) * effective_max_ratio for c in batch))
                batch_max = max(batch_max, 30)
                batch_min = int(max(len(c.split()) * effective_min_ratio for c in batch))
                batch_min = max(batch_min, 10)

                # Generate
                with torch.no_grad():
                    summary_ids = self._model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=batch_max,
                        min_length=min(batch_min, batch_max),
                        do_sample=False,
                    )

                # Decode
                batch_summaries = self._tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
                results.extend(batch_summaries)
            return results

        loop = asyncio.get_event_loop()
        summaries = await loop.run_in_executor(None, _run_generation)
        return "\n\n".join(summaries)

    async def _extractive_compress(self, text: str, query: str | None = None, target_reduction: float = 0.0) -> str:
        """Fallback or fast-path: extractive summarization via sentence scoring.
        
        If a query is provided, uses semantic similarity (Pointer #2).
        Otherwise falls back to frequency-based scoring.
        """
        import re
        import numpy as np
        from collections import Counter
        
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 3:
            return text

        # Step 1: Industry-Standard Semantic Scoring (if query is present)
        semantic_scores = {}
        if query:
            try:
                from sentence_transformers import SentenceTransformer, util
                # Lazy-load a fast embedding model
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
                query_emb = embedder.encode(query, convert_to_tensor=True)
                sent_embs = embedder.encode(sentences, convert_to_tensor=True)
                
                # Compute cosine similarities
                hits = util.cos_sim(query_emb, sent_embs)[0]
                for idx, score in enumerate(hits):
                    semantic_scores[idx] = float(score)
            except Exception as e:
                logger.debug("Smart extractive scoring failed, falling back to frequency: %s", e)

        # Step 2: Frequency-based scoring (fallback or augmentation)
        words: list[str] = []
        for s in sentences:
            words.extend(re.findall(r"\w+", s.lower()))
        freq = Counter(words)
        max_freq = max(freq.values()) if freq else 1

        # Step 3: Combine scores
        scored: list[tuple[float, int, str]] = []
        for i, sentence in enumerate(sentences):
            s_words = re.findall(r"\w+", sentence.lower())
            if not s_words:
                continue
            
            # Base frequency score
            freq_score = sum(freq.get(w, 0) / max_freq for w in s_words) / len(s_words)
            
            # Combine with semantic score if available
            if i in semantic_scores:
                # 90% weight on query-relevance to ensure exact factual retention, 10% on frequency
                score = (0.9 * semantic_scores[i]) + (0.1 * freq_score)
            else:
                score = freq_score

            # Soft boost for positional importance is no longer needed since we dynamically reorder, 
            # but we'll apply a tiny 1.05 multiplier just to retain typical contextual framing.
            if i == 0:
                score *= 1.05
            elif i == len(sentences) - 1:
                score *= 1.05

            scored.append((score, i, sentence))

        # Keep top sentences up to target ratio
        # Honor the policy's target reduction if it's set
        effective_ratio = 1.0 - target_reduction if target_reduction > 0 else self.max_summary_ratio
        
        target_count = max(3, int(len(sentences) * effective_ratio))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = scored[:target_count]
        
        if not top_sentences:
            return ""

        # LongLLMLingua Reordering Mitigation ("Lost in the Middle")
        # LLMs pay structural attention to the absolute beginning and absolute end of the prompt window.
        if len(top_sentences) > 2:
            best = top_sentences[0]
            second_best = top_sentences[1]
            
            # The rest are sorted chronologically to maintain basic readability and logical flow
            remainder_chronological = sorted(top_sentences[2:], key=lambda x: x[1])
            
            # Reconstruct: Best at the front, second best at the end.
            selected = [best] + remainder_chronological + [second_best]
        else:
            selected = sorted(top_sentences, key=lambda x: x[1]) # chronological for tiny chunks

        return " ".join(s[2] for s in selected)

    def _token_split_for_model(self, text: str, max_tokens: int = 512) -> list[str]:
        """Strictly split text by token count to ensure BART stability."""
        if not self._tokenizer:
            return text.split("\n\n") # Fallback to paragraphs if tokenizer fails
            
        all_ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        
        chunks = []
        for i in range(0, len(all_ids), max_tokens):
            chunk_ids = all_ids[i : i + max_tokens]
            chunk_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text)
                
        return chunks if chunks else [text]

    def _split_for_model(self, text: str, max_words: int = 800) -> list[str]:
        """Split text into chunks appropriate for the summarization model."""
        words = text.split()
        if len(words) <= max_words:
            return [text]

        chunks: list[str] = []
        # Try to split on paragraph boundaries
        paragraphs = text.split("\n\n")
        current_chunk: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para.split())
            if current_len + para_len > max_words and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = para_len
            else:
                current_chunk.append(para)
                current_len += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [text]


def _count_tokens(text: str, counter: object | None = None) -> int:
    if counter is not None and hasattr(counter, "count"):
        return counter.count(text)  # type: ignore[union-attr]
    return max(1, int(len(text.split()) / 0.75))
