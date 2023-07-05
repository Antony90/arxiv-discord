# ============= #
# AGENT PROMPTS #
# ============= #

from langchain import PromptTemplate

AGENT_PROMPT = \
"""You are arXiv Chat, an expert research assistant with access to a PDF papers.

Use markdown syntax whenever appopriate: markdown headers, bullet point lists etc but never use markdown links. Prefer bullet points over numbered lists.
 when outputting paper IDs, use format [`paper_id`]. Always output a paper's ID before it's title.

When asked about your tools, give a user friendly description, not exposing system terms or exact function names.

IMPORTANT:
You must always respond succinctly, in as little words as possible; do not without decorate your responses.
Focus on objective details and never make stuff up."""
# Only use tools if strictly necessary or are definitely related to a loaded paper.


# ============ #
# TOOL PROMPTS #
# ============ #

SEARCH_TOOL = \
"""
Search arXiv and get a list of relevant papers (title and ID).
You may rephrase a question to be a better search query.
Only use if user specifically wants you to search arXiv.
"""
# Assume the user wants you to search arXiv if given a vague set of terms or if asked to find/search.

# paper_title cannot be easily substituted at runtime, so generate the prompt with it fixed
MULTI_QUERY_PROMPT = lambda title: PromptTemplate(
template=f"""You are an expert reserach assistant with access to arXiv papers.
Your task is to generate 3 different versions of the given user 
question to retrieve relevant documents from a vector database for a paper titled {title}.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations 
of distance-based similarity search. Provide these alternative questions seperated by newlines. 
Original question: {{question}}""",
input_variables=["question"]
)



# SUMMARIZATION

MAP_PROMPT = PromptTemplate(input_variables=["text"], template=\
"""Summarize this text from an academic paper. Extract any key points with reasoning:

"{text}"

Summary:
""")


REDUCE_KEYPOINTS_PROMPT = PromptTemplate(input_variables=["text"], template=\
"""Write a summary collated from this collection of key points extracted from an academic paper.
The summary should highlight the core argument, conclusions and evidence, and answer the user's query.
The summary should be structured in markdown bulleted lists (with optional sub-bulletpoints) following the headings Core Argument, Evidence, and Conclusions but this can be adapted.
Key points:
{text}

Summary:
""")


REDUCE_LAYMANS_PROMPT = PromptTemplate(input_variables=["text"], template=\
"""Write a laymans summary of the key points from a paper, focussing on objective fact:


"{text}"


Summary:""")
REDUCE_COMPREHENSIVE_PROMPT = PromptTemplate(input_variables=["text"], template=\
"""Write a concise summary of the following paper, focussing on objective fact:


"{text}"


Summary:""")