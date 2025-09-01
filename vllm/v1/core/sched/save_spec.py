from dataclasses import dataclass

@dataclass
class SaveSpec:
    request_id: str
    block_id: int
    file_id: int