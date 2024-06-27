# @title Finally, run a server:
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server

from sotopia.server import LLM_Name

# model_path = ""
await run_async_server(
        model_dict={
            "env": "",
            "agent1": "together_ai/togethercomputer/llama-2-7b-chat",
            "agent2": "together_ai/togethercomputer/llama-2-7b-chat",
        },
        sampler=UniformSampler(),
        tag="simple",
        omniscient=True
    )