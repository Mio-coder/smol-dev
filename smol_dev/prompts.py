import logging
import re
from textwrap import dedent
from typing import List, Any

import openai
from openai_function_call import openai_function
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from json import loads, JSONDecodeError

logger = logging.getLogger(__name__)

SMOL_DEV_SYSTEM_PROMPT = """
You are a top tier AI developer who is trying to write a program that will generate code for the user based on their intent.
Do not leave any todos, fully implement every feature requested.

When writing code, add comments to explain what you intend to do and why it aligns with the program plan and specific instructions from the original prompt.
"""

code_pattern = re.compile(r"```([\w\s]*)\n([\s\S]*?)```", re.MULTILINE)  # codeblocks at start of the string, less eager
override_no_function = True
if override_no_function:
    print("Are you sure you want to relay on LLM's mood to answer in json?")

@openai_function
def file_paths(files_to_edit: List[str]) -> List[str]:
    """
    Construct a list of strings.
    """
    # print("filesToEdit", files_to_edit)
    return files_to_edit


def specify_file_paths(prompt: str, plan: str, model: str = 'gpt-3.5-turbo-0613'):
    try:
        if override_no_function:
            raise openai.error.InvalidRequestError("that does not exist", None)
        completion = openai.ChatCompletion.create(
            model=model,
            temperature=0.7,
            functions=[file_paths.openai_schema],
            function_call={"name": "file_paths"},
            messages=[
                {
                    "role": "system",
                    "content": f"""{SMOL_DEV_SYSTEM_PROMPT}
          Given the prompt and the plan, return a list of strings corresponding to the new files that will be generated.
                      """,
                },
                {
                    "role": "user",
                    "content": f""" I want a: {prompt}, and the plan we have agreed on is: {plan} """,
                },
            ],
        )
        result = file_paths.from_response(completion)
        return result
    except openai.error.InvalidRequestError as e:
        print(f"Error {e}, trying without functions")
        error = ""
        for i in range(10):
            # functions do not work with gpt4ll
            completion = openai.ChatCompletion.create(
                model=model,
                temperature=0.7,
                messages=[
                    {
                        "role": "system",
                        "content": dedent(f"""
                            {SMOL_DEV_SYSTEM_PROMPT}
                            Given the prompt and the plan, return a list of strings corresponding to the new files that will be generated.
                        """),
                    },
                    {
                        "role": "user",
                        "content": dedent(f"""
                            I want a: {prompt}, and the plan we have agreed on is: {plan},
                            please provide output in json with a single list of strings
                        """) + (f", and please follow guidelines from here: {error}" if error else ""),
                    },
                ],
                max_tokens=100,
            )
            result = completion["choices"][0]["message"]["content"]  # GPT4All returns
            print(completion)
            match = code_pattern.match(result)
            if match is None or match[0] != "json":
                continue
            try:
                # TODO: probably not safe
                files = loads(match[1])
            except JSONDecodeError as e:
                error += e.doc
                print(e)
            else:
                break
        else:
            raise ValueError("LLM refused to answer in json 10 times")
        return files


def plan(prompt: str, model: str = 'gpt-3.5-turbo-0613', extra_messages: List[Any] = None):
    extra_messages = extra_messages or []
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": f"""{SMOL_DEV_SYSTEM_PROMPT}

    In response to the user's prompt, write a plan using GitHub Markdown syntax. Begin with a YAML description of the new files that will be created.
    In this plan, please name and briefly describe the structure of code that will be generated, including, for each file we are generating, what variables they export, data schemas, id names of every DOM elements that javascript functions will use, message names, and function names.
                Respond only with plans following the above schema.
                  """,
            },
            {
                "role": "user",
                "content": f""" the app prompt is: {prompt} """,
            },
            *extra_messages,
        ],
        max_tokens=200,
    )
    """
    {
      "choices": [
        {
          "finish_reason": "length",
          "index": 0,
          "logprobs": null,
          "message": {
            "content": "Here is a simple implementation of the Pong game using JavaScript, HTML, CSS",
            "role": "assistant"
          },
          "references": null
        }
      ],
      "created": 1731443010,
      "id": "placeholder",
      "model": "Llama 3 8B Instruct",
      "object": "chat.completion",
      "usage": {
        "completion_tokens": 16,
        "prompt_tokens": 202,
        "total_tokens": 218
    }
    """
    try:
        return completion["choices"][0]["text"]
    except KeyError:
        return completion["choices"][0]["message"]["content"]  # GPT4All returns that


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_code(prompt: str, plan: str, current_file: str, model: str = 'gpt-3.5-turbo-0613') -> str:
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": f"""{SMOL_DEV_SYSTEM_PROMPT}

  In response to the user's prompt,
  Please name and briefly describe the structure of the app we will generate, including, for each file we are generating, what variables they export, data schemas, id names of every DOM elements that javascript functions will use, message names, and function names.

  We have broken up the program into per-file generation.
  Now your job is to generate only the code for the file: {current_file}

  only write valid code for the given filepath and file type, and return only the code.
  do not add any other explanation, only return valid code for that file type.
                  """,
            },
            {
                "role": "user",
                "content": f""" the plan we have agreed on is: {plan} """,
            },
            {
                "role": "user",
                "content": f""" the app prompt is: {prompt} """,
            },
            {
                "role": "user",
                "content": f"""
    Make sure to have consistent filenames if you reference other files we are also generating.

    Remember that you must obey 3 things:
       - you are generating code for the file {current_file}
       - do not stray from the names of the files and the plan we have decided on
       - MOST IMPORTANT OF ALL - every line of code you generate must be valid code. Do not include code fences in your response, for example

    Bad response (because it contains the code fence):
    ```javascript
    console.log("hello world")
    ```

    Good response (because it only contains the code):
    console.log("hello world")

    Begin generating the code now.

    """,
            },
        ],
    )

    code_file = completion["choices"][0]["text"]
    code_blocks = code_pattern.findall(code_file)
    # [0] is lang
    return code_blocks[1] if code_blocks else code_file