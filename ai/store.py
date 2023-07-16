from pickledb import PickleDB

class PaperStore:
    """Rudimentary persistent storage for paper titles, abstracts and generated summaries."""

    def __init__(self, filepath: str) -> None:
        self.db = PickleDB(filepath, auto_dump=False, sig=False)

    def save_summary(self, paper_id: str, summary_type: str, summary: str):
        self.db.set(f"{paper_id}-{summary_type}", summary)

    def get_summary(self, paper_id: str, summary_type: str):
        return self.db.get(f"{paper_id}-{summary_type}")

    def save_title_abstract(self, paper_id: str, title: str, abstract: str):
        self.db.set(f"{paper_id}-title", title)
        self.db.set(f"{paper_id}-abstract", abstract)

    def get_title(self, paper_id: str):
        return self.db.get(f"{paper_id}-title")
    
    def get_abstract(self, paper_id: str):
        return self.db.get(f"{paper_id}-abstract")

    def save(self):
        self.db.dump()

    def add_mentioned_paper(self, paper_id: str, chat_id: str):
        if self.db.exists(chat_id):
            papers = self.db.get(chat_id)
        else:
            papers = []
        papers.append(paper_id)
        self.db.set(chat_id, papers)
    
    def get_papers(self, chat_id: str) -> str:
        if not self.db.exists(chat_id):
            return "NONE"
        papers = self.db.get(chat_id)
        return "\n".join([f"[`{paper_id}`] {self.get_title(paper_id)}" for paper_id in papers])
    