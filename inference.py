from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLama:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, f"No checkpoints found in {checkpoints_dir}"
            chk_path = checkpoints[0]
            print(f'Loading Checkpoint {chk_path}')
            checkpoint = torch.load(chk_path, map_location='cpu')
            print(f'Checkpoint loaded in {time.time() - prev_time:.2f}s')
            prev_time = time.time()
        # Load the parameters
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.load(f)

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        # Laod the tokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.Load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "mps":
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f'Model loaded in {time.time() - prev_time:.2f}s')

        return LLama(model, tokenizer, model_args)
    
    # Implementing the top p sampling strategy
    def _sample_top_p(self, probs: torch.Tensor, p: float):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)# Cumulative sum of the sorted probabilities
        mask = probs_sum - probs_sort > p# Mask of the tokens that are not in the top p
        probs_sort[mask] = 0
        #Redistribute the probabilities
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)# Get the original token index
        return next_token

    # Implementing th top p inferencing strategy
    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        
        # Convert each token of prompt to token ids
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]

        # Make sure the batch size is not too large as we specify the max batch also for the model when we build it for the KV Cache.
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"Batch size {batch_size} exceeds the maximum batch size {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum seq length
        assert max_prompt_len <= self.args.max_seq_len, f"Prompt length {max_prompt_len} exceeds the maximum sequence length {self.args.max_seq_len}"
        total_len  = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create list that will contain the generated tokens, along with the intial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        # Fill intial tokens with the prompt tokens
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        for cur_pos in tqdm(range(1, total_len),desc="Generating tokens"):
            # Generate one token at a time
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1:cur_pos], cur_pos)
            if temperature > 0:
                # The temperature is applied BEFORE the softmax
                probs = torch.softmax(logits[:,-1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the maximum probability
                next_token = torch.argmax(logits[:,-1], dim=-1)

            next_token = next_token.reshape(-1)# Make sure the shape is (batch_size,)
            # Only replace the token if it is a padding token
            # We give all the prompts to the model to build the KV Cache, so we don't want to replace the prompt tokens
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)# Replace the token if it is not a prompt token
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_imdex, current_prompt_tokens in enumerate(tokens.tolist()):# Cut to the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)

if __name__ == '__main__':
    torch.manual_seed(0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    prompts = [
        "SImply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would ",

        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe polvrée 
        plush girafe => girafe peluche
        cheese =>"""
        #Zero Shot Prompt
        """Tell me if the following person is actually Doraemon disguised as a human:
        Name: Charles Xavier
        Decision:"""
        ]

    model = LLama.build(
        checkpoints_dir='llama-2-7b',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )

    # Inference the Model
    out_tokens, out_text = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(f'{out_text[i]}')
        print('-' * 50)