import numpy as np
import math
from typing import List, Optional, Iterable, Union


class GenRNN:
    def __init__(self, hidden_size=128, seq_length=50, learning_rate=1e-1, clip_value=5.0, seed=42):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.clip_value = clip_value

        self.text: str = ""
        self.chars: List[str] = []
        self.char_to_ix = {}
        self.ix_to_char = {}
        self.vocab_size = 0

        self.Wxh = None  # (H, V)
        self.Whh = None  # (H, H)
        self.Why = None  # (V, H)
        self.bh = None   # (H, 1)
        self.by = None   # (V, 1)

        self.mWxh = None
        self.mWhh = None
        self.mWhy = None
        self.mbh = None
        self.mby = None

        self.sentence_terminators = set([".", "!", "?", "\n"])

    def prepare_data(self, text, lowercase: bool = True):
        text = self._normalize_corpus_to_text(text)

        if lowercase:
            text = text.lower()

        if all(t not in text for t in ".!?"):
            text = text.strip()

            if not text.endswith("."):
                text = text + "."

        self.text = text
        self._build_vocab()
        self._init_params()

    def train(self, text, epochs: int = 10):
        if text is None:
            raise ValueError("Empty training text.")
        
        text = self._normalize_corpus_to_text(text)
        
        if not text.strip():
            raise ValueError("Empty training text after normalization.")

        self.prepare_data(text)

        data = self.text
        N = len(data)
        hprev = np.zeros((self.hidden_size, 1))
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length

        n_iter = 0
        
        for _ in range(epochs):
            # Randomizing start offset each epoch when possible to reduce boundary artifacts.
            if N > self.seq_length + 1:
                p = np.random.randint(0, N - self.seq_length - 1)
            else:
                p = 0

            while p + self.seq_length + 1 < N:
                inputs = [self.char_to_ix[ch] for ch in data[p : p + self.seq_length]]
                targets = [self.char_to_ix[ch] for ch in data[p + 1 : p + self.seq_length + 1]]

                loss, grads, hprev = self._forward_backward(inputs, targets, hprev)
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                self._adagrad_update(grads)

                p += self.seq_length
                n_iter += 1

    def generate_quote(self, seed_text: str = "", max_length: int = 100, temperature: float = 1.0, top_k: Optional[int] = None) -> str:
        if self.vocab_size == 0:
            raise ValueError("Model not initialized. Call train(text, ...) or prepare_data(text) first.")

        h = np.zeros((self.hidden_size, 1))
        last_ix = None

        for ch in (seed_text or "").lower():
            if ch in self.char_to_ix:
                ix = self.char_to_ix[ch]
                x = self._one_hot(ix)
                h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
                last_ix = ix

        if last_ix is None:
            last_ix = np.random.randint(self.vocab_size)

        raw = self._sample_chars(h, last_ix, n=max_length, temperature=temperature, top_k=top_k)
        out = (seed_text or "") + raw
        sent = self._truncate_to_sentence(out)

        return sent.strip()
    
    def _normalize_corpus_to_text(self, text_or_list: Union[str, Iterable[str]]) -> str:
        if isinstance(text_or_list, str):
            corpus = text_or_list
        else:
            parts = []
            
            for item in text_or_list:
                if isinstance(item, str):
                    s = item.strip()

                    if not s:
                        continue

                    if not any(s.endswith(p) for p in ".!?"):
                        s = s + "."

                    parts.append(s)

            corpus = " ".join(parts)

        return corpus

    def _build_vocab(self):
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def _init_params(self):
        V = self.vocab_size
        H = self.hidden_size
        self.Wxh = np.random.randn(H, V) * math.sqrt(2.0 / (V + H))
        q, _ = np.linalg.qr(np.random.randn(H, H))
        self.Whh = q
        self.Why = np.random.randn(V, H) * math.sqrt(2.0 / (H + V))
        self.bh = np.zeros((H, 1))
        self.by = np.zeros((V, 1))
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    def _one_hot(self, ix: int):
        x = np.zeros((self.vocab_size, 1))
        x[ix] = 1.0

        return x

    def _forward_backward(self, inputs: List[int], targets: List[int], hprev: np.ndarray):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0.0

        # Forward pass.
        for t in range(len(inputs)):
            xs[t] = self._one_hot(inputs[t])
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t - 1] + self.bh)
            ys[t] = self.Why @ hs[t] + self.by
            
            # Stable softmax.
            logits = ys[t] - np.max(ys[t])
            ps[t] = np.exp(logits) / np.sum(np.exp(logits))
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

        # Backward pass
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])

        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1.0
            dWhy += dy @ hs[t].T
            dby += dy
            dh = self.Why.T @ dy + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += dhraw @ xs[t].T
            dWhh += dhraw @ hs[t - 1].T
            dhnext = self.Whh.T @ dhraw

        # Gradient clipping.
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -self.clip_value, self.clip_value, out=dparam)

        grads = (dWxh, dWhh, dWhy, dbh, dby)

        return loss, grads, hs[len(inputs) - 1]

    def _adagrad_update(self, grads):
        dWxh, dWhh, dWhy, dbh, dby = grads
        
        for param, dparam, mem in [
            (self.Wxh, dWxh, self.mWxh),
            (self.Whh, dWhh, self.mWhh),
            (self.Why, dWhy, self.mWhy),
            (self.bh, dbh, self.mbh),
            (self.by, dby, self.mby),
        ]:
            mem += dparam * dparam
            param += -self.learning_rate * dparam / (np.sqrt(mem) + 1e-8)

    def _sample_chars(self, h: np.ndarray, seed_ix: int, n: int, temperature: float = 1.0, top_k: Optional[int] = None) -> str:
        x = self._one_hot(seed_ix)
        out_chars = []
        
        for t in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            logits = (self.Why @ h + self.by).ravel()
            logits -= np.max(logits)
        
            # Temperature scaling.
            if abs(temperature - 1.0) > 1e-8:
                logits = logits / max(1e-8, temperature)

            probs = np.exp(logits)
            probs /= probs.sum()

            # Top-k filtering.
            if top_k is not None and 1 < top_k < self.vocab_size:
                idx = np.argpartition(-probs, top_k)[:top_k]
                mask = np.zeros_like(probs)
                mask[idx] = probs[idx]
                s = mask.sum()
                probs = mask / (s if s > 0 else 1.0)

            ix = np.random.choice(self.vocab_size, p=probs)
            x = self._one_hot(ix)
            ch = self.ix_to_char[ix]
            out_chars.append(ch)

            # Optional early stop if a terminator is reached after a few characters
            if ch in self.sentence_terminators and len(out_chars) > 10:
                break

        return "".join(out_chars)

    def _truncate_to_sentence(self, text: str) -> str:
        for i, ch in enumerate(text):
            if ch in ".!?":
                return text[: i + 1]
            
        nl = text.find("\n")
        
        if nl != -1:
            return text[: nl + 1]
        
        last_space = text.rfind(" ")
        
        if last_space > 0:
            return text[:last_space]
        
        return text