from langchain.tools import BaseTool

class RelatedArxiv(BaseTool):
    """Gets list of related arXiv paper IDs for a given paper"""
    pass


class ArxivSearch(BaseTool):
    """Search arXiv with a text query and get paper IDs"""
    pass

