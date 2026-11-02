"""Environment configuration; no credential is embedded in source."""
import os
from dataclasses import dataclass, field

@dataclass
class Config:
    API_KEY: str = field(default_factory=lambda: os.environ.get("TERMSHEET_API_KEY", ""), repr=False)
    BASE_URL: str = field(default_factory=lambda: os.environ.get("TERMSHEET_BASE_URL", "https://api.siliconflow.cn/v1"))
    MODEL: str = field(default_factory=lambda: os.environ.get("TERMSHEET_MODEL", "deepseek-ai/DeepSeek-V3.2"))
    MAX_CHARS: int = 70000

    def validate(self):
        if not self.API_KEY:
            raise ValueError("Set TERMSHEET_API_KEY before running the API workflow.")
        if not self.BASE_URL.startswith("https://"):
            raise ValueError("The model endpoint must use HTTPS.")
