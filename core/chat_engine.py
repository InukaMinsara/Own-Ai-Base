from pathlib import Path
import re
import sys
from difflib import SequenceMatcher

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import OwnTokenizer
from model.own_ai import OwnAI
from rag.retriever import LocalRetriever


class OwnAIEngine:
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        sft_checkpoint = (
            ROOT
            / "checkpoints"
            / "own_ai_v6_sft_best.pt"
        )

        sft_tokenizer = (
            ROOT
            / "checkpoints"
            / "tokenizer_v6.json"
        )

        base_checkpoint = (
            ROOT
            / "checkpoints"
            / "own_ai_best.pt"
        )

        base_tokenizer = (
            ROOT
            / "checkpoints"
            / "tokenizer_best.json"
        )

        if (
            sft_checkpoint.exists()
            and sft_tokenizer.exists()
        ):
            self.checkpoint_path = (
                sft_checkpoint
            )
            self.tokenizer_path = (
                sft_tokenizer
            )
            self.stage = "v6-sft"
        else:
            self.checkpoint_path = (
                base_checkpoint
            )
            self.tokenizer_path = (
                base_tokenizer
            )
            self.stage = "base"

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "No trained model checkpoint found."
            )

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                "No tokenizer checkpoint found."
            )

        self.tokenizer = OwnTokenizer(
            vocab_size=512
        )

        self.tokenizer.load(
            self.tokenizer_path
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model = OwnAI(
            vocab_size=checkpoint["vocab_size"],
            block_size=checkpoint["block_size"],
            d_model=checkpoint["d_model"],
            n_heads=checkpoint["n_heads"],
            n_layers=checkpoint["n_layers"],
            dropout=checkpoint.get(
                "dropout",
                0.1,
            ),
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )

        self.model.eval()

        self.eos_id = (
            self.tokenizer.token_to_id[
                self.tokenizer.eos_token
            ]
        )

        self.retriever = LocalRetriever(
            ROOT
        )

        self.retriever.build()

        self.instruction_examples = self._load_instruction_examples()

        self.history = []

    def _load_instruction_examples(self):
        path = (
            ROOT
            / "data"
            / "processed"
            / "instructions.txt"
        )

        if not path.exists():
            return []

        text = path.read_text(
            encoding="utf-8"
        )

        examples = []

        for chunk in text.split("<BOS>"):
            chunk = chunk.strip()

            if not chunk:
                continue

            chunk = chunk.split(
                "<EOS>",
                1
            )[0].strip()

            match = re.search(
                r"User:s*(.*?)s*Assistant:s*(.*)",
                chunk,
                re.S,
            )

            if not match:
                continue

            question = match.group(1).strip()
            answer = match.group(2).strip()

            if question and answer:
                examples.append(
                    (question, answer)
                )

        return examples

    @staticmethod
    def _normalize_text(text):
        return re.sub(
            r"[^a-z0-9s]",
            " ",
            text.lower(),
        ).split()

    def _best_knowledge_answer(self, user_text):
        """
        Extract a concise answer from a strong local-knowledge match.
        This gives the small model a graceful fallback instead of gibberish.
        """
        results = self.retriever.search(
            user_text,
            top_k=5,
            min_score=1.0,
        )

        query_words = set(
            self._normalize_text(user_text)
        )

        if len(query_words) < 2:
            return None

        best = None
        best_score = 0.0

        for item in results:
            sentences = re.split(
                r"(?<=[.!?])\s+",
                item["text"].replace("\n", " ").strip(),
            )

            for sentence in sentences:
                sentence = sentence.strip()

                if len(sentence) < 35:
                    continue

                words = set(
                    self._normalize_text(sentence)
                )

                overlap = (
                    len(query_words & words)
                    / max(1, len(query_words))
                )

                score = (
                    0.65 * overlap
                    + 0.35 * min(
                        item["score"] / 8.0,
                        1.0,
                    )
                )

                if score > best_score:
                    best_score = score
                    best = sentence

        if best_score >= 0.38:
            return best

        return None

    def _best_instruction_answer(self, user_text):
        query = self._normalize_text(
            user_text
        )

        if not query or not self.instruction_examples:
            return None

        query_text = " ".join(query)
        best_answer = None
        best_score = 0.0

        for question, answer in self.instruction_examples:
            candidate = self._normalize_text(
                question
            )

            if not candidate:
                continue

            candidate_text = " ".join(
                candidate
            )

            overlap = (
                len(set(query) & set(candidate))
                / max(
                    1,
                    len(set(query) | set(candidate))
                )
            )

            sequence = SequenceMatcher(
                None,
                query_text,
                candidate_text,
            ).ratio()

            score = (
                0.65 * overlap
                + 0.35 * sequence
            )

            if score > best_score:
                best_score = score
                best_answer = answer

        # High-confidence instruction retrieval only.
        # Unknown questions continue through the language model.
        if best_score >= 0.68:
            return best_answer

        return None

    def reset_memory(self):
        self.history.clear()

    def _history_text(self, max_turns=3):
        recent = self.history[
            -max_turns * 2:
        ]

        return "\n".join(
            f"{role}: {text}"
            for role, text in recent
        )

    def _build_prompt(
        self,
        user_text,
        use_rag,
    ):
        sections = []

        memory = self._history_text()

        if memory:
            sections.append(
                "Conversation memory:\n"
                + memory
            )

        if use_rag:
            context, _ = (
                self.retriever.context(
                    user_text,
                    top_k=3,
                    max_chars=1800,
                )
            )

            if context:
                sections.append(
                    "Relevant local knowledge:\n"
                    + context
                )

        sections.append(
            f"User: {user_text}\nAssistant:"
        )

        return "\n\n".join(
            sections
        )

    def _trim_prompt_ids(self, ids):
        keep = max(
            1,
            self.model.block_size - 1,
        )

        if len(ids) > keep:
            return ids[-keep:]

        return ids

    @torch.no_grad()
    def generate(
        self,
        user_text,
        use_rag=True,
        max_new_tokens=120,
        temperature=0.72,
        top_k=40,
        top_p=0.90,
        repetition_penalty=1.08,
    ):
        self.model.eval()

        # First use high-confidence learned instruction pairs.
        # This prevents the tiny model from corrupting simple known answers.
        retrieved_answer = self._best_instruction_answer(
            user_text
        )

        if retrieved_answer is not None:
            self.history.append(
                ("User", user_text)
            )
            self.history.append(
                ("Assistant", retrieved_answer)
            )
            return retrieved_answer

        knowledge_answer = self._best_knowledge_answer(
            user_text
        )

        if knowledge_answer is not None:
            self.history.append(
                ("User", user_text)
            )
            self.history.append(
                ("Assistant", knowledge_answer)
            )
            return knowledge_answer

        prompt = self._build_prompt(
            user_text,
            use_rag,
        )

        bos_id = (
            self.tokenizer.token_to_id[
                self.tokenizer.bos_token
            ]
        )

        prompt_ids = (
            self.tokenizer.encode(
                prompt
            )
        )

        generated = [
            bos_id,
            *self._trim_prompt_ids(
                prompt_ids
            ),
        ]

        prompt_len = len(generated)

        for _ in range(max_new_tokens):
            context_ids = generated[
                -self.model.block_size:
            ]

            x = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=self.device,
            )

            logits, _ = self.model(x)
            logits = logits[:, -1, :]

            logits = logits / max(
                temperature,
                1e-5,
            )

            for token_id in set(generated):
                logits[0, token_id] /= (
                    repetition_penalty
                )

            if top_k:
                k = min(
                    top_k,
                    logits.size(-1),
                )

                values, _ = torch.topk(
                    logits,
                    k,
                )

                logits[
                    logits
                    < values[:, [-1]]
                ] = float("-inf")

            if top_p and top_p < 1.0:
                (
                    sorted_logits,
                    sorted_indices,
                ) = torch.sort(
                    logits,
                    descending=True,
                )

                probabilities = torch.softmax(
                    sorted_logits,
                    dim=-1,
                )

                cumulative = torch.cumsum(
                    probabilities,
                    dim=-1,
                )

                remove = cumulative > top_p
                remove[:, 1:] = (
                    remove[:, :-1]
                    .clone()
                )
                remove[:, 0] = False

                sorted_logits[
                    remove
                ] = float("-inf")

                logits = torch.full_like(
                    logits,
                    float("-inf"),
                )

                logits.scatter_(
                    1,
                    sorted_indices,
                    sorted_logits,
                )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            next_id = torch.multinomial(
                probabilities,
                1,
            ).item()

            generated.append(
                next_id
            )

            if next_id == self.eos_id:
                break

        new_ids = generated[
            prompt_len:
        ]

        answer = self.tokenizer.decode(
            new_ids
        )

        for marker in (
            "User:",
            "Assistant:",
        ):
            if marker in answer:
                answer = answer.split(
                    marker,
                    1,
                )[0]

        answer = answer.strip()

        if not answer:
            answer = (
                "I do not have a useful "
                "answer yet."
            )

        self.history.append(
            ("User", user_text)
        )

        self.history.append(
            ("Assistant", answer)
        )

        return answer

    def info(self):
        params = sum(
            p.numel()
            for p in self.model.parameters()
        )

        return {
            "stage": self.stage,
            "device": self.device,
            "parameters": params,
            "vocabulary": (
                self.tokenizer.vocab_size
            ),
            "context": (
                self.model.block_size
            ),
            "embedding": (
                self.model.d_model
            ),
            "layers": (
                len(self.model.blocks)
            ),
            "retrieval_documents": (
                len(self.retriever.documents)
            ),
        }

