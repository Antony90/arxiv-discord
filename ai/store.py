from typing import Literal
import pickledb

class PaperStore:
    """Rudimentary persistent storage for paper titles, abstracts and generated summaries."""

    def __init__(self, filepath: str) -> None:
        self.db = pickledb.load(filepath, auto_dump=False)

    def save_summary(self, paper_id: str, summary_type: str, summary: str):
        self.db.set(f"{paper_id}-{summary_type}", summary)

    def get_summary(self, paper_id: str, summary_type: str) -> str | Literal[False]:
        self.db.get(f"{paper_id}-{summary_type}")

    def save_title_abstract(self, paper_id: str, title: str, abstract: str):
        self.db.set(f"{paper_id}-title", title)
        self.db.set(f"{paper_id}-abstract", abstract)

    def get_title(self, paper_id: str) -> str | Literal[False]:
        return self.db.get(f"{paper_id}-title")
    
    def get_abstract(self, paper_id: str) -> str | Literal[False]:
        return self.db.get(f"{paper_id}-abstract")

    def save(self):
        self.db.dump()
    
    