from typing import Literal
import pickledb

class PaperStore:
    def __init__(self, filepath: str) -> None:
        self.db = pickledb.load(filepath, auto_dump=False)

    def save_summary(self, paper_id: str, summary_type: str, summary: str):
        self.db.set(f"{paper_id}-{summary_type}", summary)

    def get_summary(self, paper_id: str, summary_type: str) -> str | Literal[False]:
        self.db.get(f"{paper_id}-{summary_type}")

    def save_title(self, paper_id: str, title: str):
        self.db.set(paper_id, title)

    def get_title(self, paper_id: str) -> str | Literal[False]:
        return self.db.get(paper_id)
    
    def save(self):
        self.db.dump()
    
    