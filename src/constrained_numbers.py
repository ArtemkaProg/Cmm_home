"""Constrained decoding, step 1: functions whose parameters are numbers.

Works end to end for fn_add_numbers and fn_get_square_root.
String parameters are left for you (see generate_parameters).
"""

import json
import re
import time
from typing import Any, Callable

import numpy as np
from llm_sdk import Small_LLM_Model

# "" , "-", "0", "12", "12.", "12.5" are all valid *beginnings* of a number.
NUMBER_PREFIX = re.compile(r"-?((0|[1-9]\d*)(\.\d*)?)?")
# Only these are valid *finished* numbers ("12." and "-" are not).
NUMBER_FULL = re.compile(r"-?(0|[1-9]\d*)(\.\d+)?")
MAX_NUMBER_LEN = 15


# helpers


def is_number_prefix(text: str) -> bool:
    """Return True if text can still grow into a valid JSON number."""
    return NUMBER_PREFIX.fullmatch(text) is not None


def is_complete_number(text: str) -> bool:
    """Return True if text is already a valid JSON number."""
    return NUMBER_FULL.fullmatch(text) is not None


def is_name_prefix(text: str, names: list[str]) -> bool:
    """Return True if text is the beginning of at least one name."""
    return any(name.startswith(text) for name in names)


def name_is_finished(text: str, names: list[str]) -> bool:
    """Return True if text is a full name that no other name extends."""
    if text not in names:
        return False
    return not any(n != text and n.startswith(text) for n in names)


def number_token_ok(partial: str, token: str, terminator: str) -> bool:
    """Decide if `token` may follow the number written so far.

    - the terminator (',' or '}') is allowed only once the number is complete
    - any other token must keep the text a valid number prefix
    """
    if token == terminator:
        return is_complete_number(partial)
    return is_number_prefix(partial + token)


def string_token_ok(
    partial: str, token: str, terminator: str
) -> bool:
    if token == terminator:
        return True

    candidate = partial + token

    if len(candidate) > 200:
        return False

    if any(ord(char) < 32 for char in token):
        return False

    if '"' in token or "\\" in token:
        return False

    return True


def build_prompt(functions: list[dict[str, Any]], user_prompt: str) -> str:
    """Build the text the model sees before the forced JSON starts."""
    lines = [
        "Pick the function that matches the request and fill in its",
        "arguments. Answer with one JSON object.",
        "",
        "Functions:",
    ]
    for fn in functions:
        params = ", ".join(  # 'a' (number), 'b' (number)
            f"{name} ({spec['type']})"
            for name, spec in fn["parameters"].items()
        )
        lines.append(
            (
                f"- "
                f"{fn['name']}"
                f"("
                f"{params}"
                f"): "
                f"{fn['description']}"
                # "- fn_add_numbers('a' (number), 'b' (number)): Add two
                # numbers
                # together and return their sum."
            )
        )
    lines += ["", f"Request: {user_prompt}", "JSON:", ""]
    return "\n".join(lines)


# The decoder


class FunctionCaller:
    """Turns a prompt into a function call using constrained decoding."""

    def __init__(
        self, model: Small_LLM_Model, functions: list[dict[str, Any]]
    ) -> None:
        """Store the model and the function definitions, load the vocab."""
        self.model = model
        self.functions = {fn["name"]: fn for fn in functions}
        self.names = list(self.functions)
        self.pvocab = self._build_pvocab()

    def _build_pvocab(self) -> dict[int, str]:
        """Map token id -> readable text ('Ġ' means a leading space)."""
        with open(
            self.model.get_path_to_vocab_file(), encoding="utf-8"
        ) as file:
            vocab: dict[str, int] = json.load(file)
        return {
            token_id: self.model.decode([token_id])
            for token_id in vocab.values()
        }

    def _append_text(self, ids: list[int], text: str) -> None:
        """Force text into the sequence (no model call, no choice)."""
        ids.extend(self.model.encode(text).tolist()[0])

    def _pick_token(
        self, ids: list[int], is_valid: Callable[[str], bool]
    ) -> int:
        """Return the best-scoring token that passes is_valid.

        Walking tokens from highest to lowest logit and stopping at the
        first valid one gives the same result as "set invalid logits to
        -inf, then argmax", but is faster. Ids missing from pvocab
        (special tokens, padding) are skipped, i.e. always invalid.
        """
        logits = np.array(self.model.get_logits_from_input_ids(ids))
        for token_id in np.argsort(-logits):
            text = self.pvocab.get(int(token_id))
            if text and is_valid(text):
                actual = self.model.decode([int(token_id)])
                if text != actual:
                    print("TOKEN MISMATCH:", repr(text), repr(actual))
                return int(token_id)
        raise RuntimeError("No valid token found at this step")

    def choose_name(self, ids: list[int]) -> str:
        """Let the LLM choose a function name, token by token."""
        name = ""
        while not name_is_finished(name, self.names):
            token_id = self._pick_token(
                ids, lambda tok: is_name_prefix(name + tok, self.names)
            )
            name += self.pvocab[token_id]
            ids.append(token_id)
        return name

    def generate_number(self, ids: list[int], terminator: str) -> str:
        """Let the LLM write a number; it stops by choosing terminator.

        The terminator is NOT appended to ids, the caller writes it.
        """
        number = ""
        while True:
            if len(number) >= MAX_NUMBER_LEN and is_complete_number(number):
                break
            token_id = self._pick_token(
                ids, lambda tok: number_token_ok(number, tok, terminator)
            )
            token = self.pvocab[token_id]
            if token == terminator:
                break
            number += token
            ids.append(token_id)
        return number

    def generate_string(self, ids: list[int], terminator: str) -> str:
        string = ""
        max_length = 100
        max_tokens = 60

        for _ in range(max_tokens):
            if len(string) >= max_length:
                break

            token_id = self._pick_token(
                ids,
                lambda tok: string_token_ok(string, tok, terminator)
                and (
                    tok == terminator
                    or len(string + tok) <= max_length
                )
            )

            token = self.pvocab[token_id]

            if token == terminator:
                return string

            string += token
            ids.append(token_id)

        raise RuntimeError(
            f"String generation did not terminate: {string!r}"
        )

    def generate_parameters(self, ids: list[int], name: str) -> None:
        """Write all parameters of function name after '"parameters": {'."""
        params = self.functions[name]["parameters"]
        keys = list(params)
        for index, key in enumerate(keys):
            if index > 0:
                self._append_text(ids, ", ")
            self._append_text(ids, f'"{key}": ')
            kind = params[key]["type"]
            last = index == len(keys) - 1
            if kind == "number":
                self.generate_number(ids, "}" if last else ",")
            elif kind == "string":
                self._append_text(ids, '"')
                self.generate_string(ids, '"')
                self._append_text(ids, '"')
            else:
                raise NotImplementedError(
                    f"type {kind!r} of parameter {key!r} not supported yet"
                )

    def call(self, prompt: str) -> dict[str, Any]:
        """Return {"prompt", "name", "parameters"} for one user prompt."""
        ids = self.model.encode(
            build_prompt(list(self.functions.values()), prompt)
        ).tolist()[0]
        start = len(ids)
        self._append_text(ids, '{"name": "')
        name = self.choose_name(ids)
        self._append_text(ids, '", "parameters": {')
        self.generate_parameters(ids, name)
        self._append_text(ids, "}}")
        generated = self.model.decode(ids[start:])

        try:
            call = json.loads(generated)
        except json.JSONDecodeError as error:
            print("\nINVALID JSON:")
            print(repr(generated))
            print("\nERROR POSITION:")
            print(repr(generated[max(0, error.pos - 30):error.pos + 30]))
            raise

        return {
            "prompt": prompt,
            "name": call["name"],
            "parameters": call["parameters"],
        }


def load_json(path: str) -> Any:
    """Read a JSON file, exit with a clear message on failure."""
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot read {path}: {error}")


def main() -> None:
    functions = load_json("data/input/functions_definition.json")
    tests = load_json("data/input/function_calling_tests.json")
    model = Small_LLM_Model(device="cpu")
    caller = FunctionCaller(model, functions)
    for test in tests:
        start = time.time()
        try:
            result = caller.call(test["prompt"])
        except NotImplementedError as error:
            print(f"SKIP  {test['prompt']!r}: {error}")
            continue
        print(f"{time.time() - start:5.1f}s", json.dumps(result))


if __name__ == "__main__":
    main()
