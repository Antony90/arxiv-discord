<p align="center"> 
  <img src="images/icon.png" alt="logo" width="85px" height="85px">
</p>
<h1 align="center">arXiv Chat</h1>
<h3 align="center"> An AI research assistant and Discord bot </h3>  

<p align="center">
  <a href="https://discord.gg/Y38bcWSzSD">
    <img src="https://dcbadge.vercel.app/api/server/Y38bcWSzSD?style=flat-square"/>
  </a>
</p>


An AI chatbot agent, designed to assist researchers and enthusiasts accessing and interacting with the [arXiv paper archive](https://arxiv.org/).

The goal is to make the process of literature exploration more efficient and facilitate discussions across multiple papers, as well as with peers. Built with [Langchain](https://python.langchain.com/docs/get_started/introduction.html), [discord.py](https://discordpy.readthedocs.io/), GPT-3.5 using [OpenAI API](https://platform.openai.com/docs/introduction).

## Demo
Join the discord server [here](https://discord.gg/Y38bcWSzSD), start a chat. Bot invite link coming soon.

## Features

- **Querying Papers**
    - Ask questions about specific papers, arXiv-Chat will respond with appropriately cited responses.
- **Paper Summaries**
    - Summarize papers concisely, highlighting the main argument and conclusions. There are 3 variations: {laymans, keypoints, comprehensive} which are chosen based on preference and the level of detail required.
- **Generate Questions**
    - arXiv Chat can generate a set of research questions that provide alternative perspectives and valuable insights. Throw these questions back at the AI to further your aid your exploration and understanding of the paper's important take aways.
- **Engaging Discussion**
    - You can start discussions involving multiple papers. When answering comparative questions, the contents of each paper will be polled.
- **Citations**
    - If you want to explore further, you can ask for a paper's citations.
- **Paper Recommendation/Search**
    - arXiv-Chat may recommend specific papers based on your discussion. It can also simply search with a query.
---

- As an autonomous agent, it can decompose user prompts into several tasks, removing the need to specifically name tasks to be carried out. See OpenAI's [Function Calling API](https://platform.openai.com/docs/guides/gpt/function-calling).


## Planned Features

- **git Repository Loader**
    - The ability to clone in-paper git repo links, allowing their code to be part of the agent's accessible knowledge base. Could assist in understanding the implementation of a paper with code.

If you have any ideas for additional features or want to participate in developing these features, see [here](#contributing)


## Installation
Note: you must have Python 3.9 or later installed.

1. Fill `.env.example` and rename to `.env`. You'll need {[OpenAI](https://platform.openai.com/account/api-keys), [SerpAPI](https://serpapi.com/)} API keys.

2. 
    #### Without Docker
    Install python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    #### With Docker
    Build the image:
    ```bash
    docker build . -f Dockerfile -t arxiv-chat
    ```


## Usage

Run the discord bot locally:
#### Without Docker
```bash
python3 main.py
```

#### With Docker
```bash
docker run -it --rm --env-file .env arxiv-chat
```

Specify `-t` option to run in REPL/termnial user input mode.

## Contributing

Contributions are appreciated. Submit a PR if you have a new feature idea, or to suggest improvements.
