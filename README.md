# arXiv Chat: AI research assistant & Discord bot
An AI chatbot agent, designed to assist researchers and enthusiasts accessing and interacting with the [arXiv paper archive](https://arxiv.org/).

The goal is to make the process of literature exploration more efficient and facilitate discussions across multiple papers, as well as with peers. Built with [Langchain](https://python.langchain.com/docs/get_started/introduction.html), [discord.py](https://discordpy.readthedocs.io/), GPT-3.5 using [OpenAI API](https://platform.openai.com/docs/introduction).

## Demo
Join the official discord [here](https://discord.gg/Y38bcWSzSD). ~~Invite the bot to your server~~. Coming soon.

## Features

- **Querying Papers**
    - Ask questions about specific papers, arXiv-Chat will respond with appropriately cited responses.
- **Critical Discussions**
    - You can engage in discussions involving multiple papers asking comparative questions. Key arguments will be raised to facilitate an insightful conversation
- **Paper Summaries**
    - Summarize papers concisely, highlighting the main argument and conclusions. There are 3 variations: {laymans, keypoints, comprehensive} which are chosen based on preference and the level of detail required.
- **Paper Recommendation/Search**
    - arXiv-Chat may point you to specific papers based on your discussion. It can also simply search arXiv.
---

- As an autonomous agent, it can decompose user prompts into several tasks, removing the need to specifically name tasks to be carried out. See OpenAI's [Function Calling API](https://platform.openai.com/docs/guides/gpt/function-calling).


## Planned Features

- **git Repository Loader**
    - The ability to clone in-paper git repo links, allowing their code to be part of the agent's accessible knowledge base. Could assist in understanding the implementation of a paper with code.

If you have any ideas for additional features or want to participate in developing these features, see [here](#contributing)


## Installation
Note: you must have Python 3.9 or later installed.

1. Create a `.env` file in the project directory:
    ```env
    OPENAI_API_KEY=
    BOT_TOKEN=
    ```
    Get an OpenAI API key [here](https://platform.openai.com/account/api-keys).

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
docker run -it --rm arxiv-chat
```

Specify `-t` option to run in REPL/termnial user input mode.

## Contributing

Contributions are appreciated. Submit a PR if you have a new feature idea, or to suggest improvements.
