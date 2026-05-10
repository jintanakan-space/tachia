from transformers import PreTrainedTokenizerFast

class PratchyaTokenizer(PreTrainedTokenizerFast):

    def __init__(self, **kwargs):
        kwargs.setdefault("bos_token", "<|BOS|>")
        kwargs.setdefault("eos_token", "<|EOS|>")
        kwargs.setdefault("pad_token", "<|PAD|>")

        is_pretrain = kwargs.pop("is_pretrain", False)
        
        super().__init__(**kwargs)

        if not is_pretrain:
            self.add_eos_token = False
            self.add_bos_token = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        kwargs["use_fast"] = kwargs.get("use_fast", True)
        return super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        if isinstance(token_ids, int):
            token_ids = [token_ids]

        tokens = self.convert_ids_to_tokens(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )
        return "".join(tokens)

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        return [
            self.decode(
                sequence,
                skip_special_tokens=skip_special_tokens,
                **kwargs,
            )
            for sequence in sequences
        ]

    @property
    def vocab_size(self) -> int:
        return len(self)
