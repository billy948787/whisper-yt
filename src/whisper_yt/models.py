from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Subtitle:
    id: int
    start: float
    end: float
    text: str
    translated_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Subtitle":
        return cls(
            id=int(data["id"]),
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data["text"]),
            translated_text=str(data.get("translated_text", "")),
        )
