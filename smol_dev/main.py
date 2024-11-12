import argparse
import sys
from argparse import Namespace
from functools import wraps

import openai

from smol_dev.prompts import plan, specify_file_paths, generate_code
from smol_dev.utils import generate_folder, write_file

# monkeypatch for custom api
openai.api_key = "-"
openai.api_base = "http://localhost:4891/v1"

# model = "gpt-3.5-turbo-0613"
# defaultmodel = "gpt-4-0613"
defaultmodel = "Llama 3 8B Instruct"


def get_print_debug(debug: bool):
    if debug:
        @wraps(print)
        def print_debug(*args, **kwargs):
            print(*args, **kwargs)
    else:
        @wraps(print)
        def print_debug(*args, **kwargs):
            pass
    return print_debug


def main(prompt, generate_folder_path="generated", debug=False, model: str = defaultmodel):
    print_debug = get_print_debug(debug)

    # create generateFolder folder if doesnt exist
    generate_folder(generate_folder_path)

    # plan shared_deps
    print_debug("--------shared_deps---------")
    with open(f"{generate_folder_path}/shared_deps.md", "wb") as f:
        shared_deps = plan(prompt, model=model)
    print_debug(shared_deps)

    write_file(f"{generate_folder_path}/shared_deps.md", shared_deps)

    print_debug("--------shared_deps---------")

    # specify file_paths
    print_debug("--------specify_filePaths---------")
    file_paths = specify_file_paths(prompt, shared_deps, model=model)
    print_debug(file_paths)
    print_debug("--------file_paths---------")

    # loop through file_paths array and generate code for each file
    for file_path in file_paths:
        file_path = f"{generate_folder_path}/{file_path}"  # just append prefix
        print_debug(f"--------generate_code: {file_path} ---------")

        code = generate_code(prompt, shared_deps, file_path, model=model)
        print_debug(code)

        print_debug(f"--------generate_code: {file_path} ---------")

        # create file with code content
        write_file(file_path, code)

    print("--------smol dev done!---------")


# for local testing
# python main.py --prompt "a simple JavaScript/HTML/CSS/Canvas app that is a one player game of PONG..." --generate_folder_path "generated" --debug True

if __name__ == "__main__":
    prompt = """
a simple JavaScript/HTML/CSS/Canvas app that is a one player game of PONG. 
The left paddle is controlled by the player, following where the mouse goes.
The right paddle is controlled by a simple AI algorithm, which slowly moves the paddle toward the ball at every frame, with some probability of error.
Make the canvas a 400 x 400 black square and center it in the app.
Make the paddles 100px long, yellow and the ball small and red.
Make sure to render the paddles and name them so they can controlled in javascript.
Implement the collision detection and scoring as well.
Every time the ball bouncess off a paddle, the ball should move faster.
It is meant to run in Chrome browser, so dont use anything that is not supported by Chrome, and don't use the import and export keywords.
  """
    if len(sys.argv) == 2:
        prompt = sys.argv[1]
        args = Namespace(prompt=prompt)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", type=str, required=True, help="Prompt for the app to be created.")
        parser.add_argument("--generate_folder_path", type=str, default="generated",
                            help="Path of the folder for generated code.")
        parser.add_argument("-d", "--debug", type=bool, default=False, help="Enable or disable debug mode.")
        args = parser.parse_args()

    main(prompt=args.prompt, generate_folder_path=args.generate_folder_path, debug=args.debug)
