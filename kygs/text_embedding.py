import math
from typing import Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import track
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from kygs.utils.typing import NDArrayFloat


class TextEmbeddingModel:
    def __init__(
        self,
        model: str,
        batch_size: int,
        max_input_seq_length: int,
        device: str,
        verbose: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.max_input_seq_length = max_input_seq_length
        self.device = device
        self.verbose = verbose
        tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer = cast(PreTrainedTokenizer, tokenizer)
        self.model: PreTrainedModel = AutoModel.from_pretrained(
            model, device_map="auto"
        )
        self.model.eval()  # some layers behave differently in traning and eval

    def _get_embedding_dimension(self) -> int:
        hidden_size = self.model.config.hidden_size
        if hidden_size is None:
            raise ValueError("Model config missing hidden_size")
        return int(hidden_size)

    def predict(self, text_sequences: Sequence[str]) -> NDArrayFloat:
        n_samples = len(text_sequences)
        emb_dim = self._get_embedding_dimension()
        embeddings = np.zeros((n_samples, emb_dim), np.float32)

        n_batch = math.ceil(n_samples / self.batch_size)

        batch_iterator = range(n_batch)
        if self.verbose:
            batch_iterator = track(
                batch_iterator,  # type: ignore
                description="Computing embeddings",
                total=n_batch,
            )

        with torch.no_grad():
            for batch_i in batch_iterator:
                l_i = batch_i * self.batch_size
                r_i = (
                    (batch_i + 1) * self.batch_size if batch_i != n_batch - 1 else None
                )
                batch_text_sequences = text_sequences[l_i:r_i]

                # Tokenize the input texts
                # batch_dict.inputs_ids -> tokenized sequences,
                # shape = [# seq, max # tokens]
                # batch_dict.attention_mask -> masks for padded tokens
                # (zeros and ones as usual), shape = [# seq, max # tokens]
                if len(batch_text_sequences) == 1:
                    max_length = min(
                        self.max_input_seq_length, len(batch_text_sequences[0])
                    )
                else:
                    max_length = self.max_input_seq_length

                batch_dict = self.tokenizer(
                    list(batch_text_sequences),
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                batch_dict["input_ids"] = batch_dict["input_ids"].to(self.device)
                batch_dict["attention_mask"] = batch_dict["attention_mask"].to(
                    self.device
                )
                outputs = self.model(**batch_dict)

                # Instead of using outputs.pooler_output,
                # we use average pool (they do differ)
                # embeddings ->  shape = [# seq, emb dim]
                batch_embeddings = average_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )

                # Below is literally cosine similarity
                # normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings[l_i:r_i, :] = batch_embeddings.cpu().detach().numpy()

        return embeddings


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # just fill masked values with zeros
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    # average over non-masked values across the token dimension
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
