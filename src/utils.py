import json
from typing import List
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

@dataclass
class Claim(DataClassJsonMixin):
    text: str       # "The capital of Australia is Canberra"
    label: int      # 1=true, 0=false
    entity: str     # "Australia"
    predicate: str  # "capital"
    object: str     # "Canberra" or counterfactual
    domain: str     # geo, bio, date, sports, science

def load_claims_jsonl(path: str) -> List[Claim]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            items.append(Claim.from_dict(d))
    return items