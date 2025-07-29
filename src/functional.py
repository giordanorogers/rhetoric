from typing import Literal
from nnsight import LanguageModel

def logit_lens(
    model: LanguageModel,
    prompt: str,
    kind: Literal["full", "last"] = "last",
    k: int = 5,
    return_logits: bool = False
):
    logit_lens = []
    num_layers = model.config.num_hidden_layers

    if kind == "last":
        logits_probs = []
        with model.trace(prompt):
            logits = model.output.logits[0].save()
            probs = logits.softmax(dim=-1).save()
            logits_probs.append((logits, probs))
        logits = logits_probs[0][0][-1].topk(k=k).values
        probs = logits_probs[0][1][-1].topk(k=k).values
        token_ids = logits_probs[0][1][-1].topk(k=k).indices
        tokens = [model.tokenizer.decode(id) for id in token_ids]

        for t, l, p, id in zip(tokens, logits, probs, token_ids):
            logit_lens.append(f'"{t}"[{id}] (p={p}, logit={l})')

        return logit_lens