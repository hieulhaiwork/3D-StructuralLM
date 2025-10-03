import json
import random
from copy import deepcopy

def generate_few_shot_prompt(
    full_dataset: list[dict],
    current_input: str,
    num_few_shots: int = 2,
    shuffle_examples: bool = True
) -> str:
    """
    Generates a prompt string with few-shot examples and the current input for a 3D scene generator model.

    Args:
        full_dataset: A list of dictionaries, where each dictionary has 'instruction', 'input', and 'output' keys.
                      This represents your complete fine-tuning dataset.
        current_input: The new room description for which to generate latent indices.
        num_few_shots: The number of examples to include in the few-shot prompt.
        shuffle_examples: Whether to randomly shuffle the selected few-shot examples.

    Returns:
        A string containing the formatted prompt.
    """

    if num_few_shots > len(full_dataset):
        print(f"Warning: Requested {num_few_shots} few-shot examples, but only {len(full_dataset)} available. Using all available examples.")
        num_few_shots = len(full_dataset)

    # The fixed instruction from your data, which is part of every prompt
    fixed_instruction = full_dataset[0]['instruction']

    # Select few-shot examples
    # Exclude the current_input if it happens to be in the full_dataset (unlikely but good practice)
    available_examples = [
        item for item in full_dataset if item['input'] != current_input
    ]

    if len(available_examples) < num_few_shots:
        print(f"Warning: Not enough unique examples for few-shot. Using all {len(available_examples)} unique examples.")
        selected_examples = available_examples
    else:
        selected_examples = random.sample(available_examples, num_few_shots)

    if shuffle_examples:
        random.shuffle(selected_examples)

    # Build the prompt
    prompt_parts = [
        f"{fixed_instruction}\n" # The general instruction goes first
    ]

    for i, example in enumerate(selected_examples):
        prompt_parts.append(f"### Example {i+1} Input:\n{example['input']}")
        prompt_parts.append(f"### Example {i+1} Output JSON:\n{example['output']}")

    prompt_parts.append(f"### Current Input:\n{current_input}")
    prompt_parts.append(f"### Current Output JSON:")

    return "\n".join(prompt_parts)


def generate_few_shot_prompt_dataset(
    full_dataset: list[dict],
    num_few_shots: int=2,
) -> list[dict]:
    """
    Generates a prompt dataset with few-shot examples randomly selected from whole dataset. Instruction will be remained same and input will be modified

    Args:
        full_dataset: A list of dictionaries, where ech dictionary has 'instruct', 'input' and 'output' keys
        num_few_shots: The number of examples to include in the few-shot prompt.
        shuffle_examples: Whether to randomly shuffle the selected few-shot examples.

    Returns:
        A dict with few-shot formatted datasets
    """

    few_shot_dataset = []

    if num_few_shots > len(full_dataset):
        print(f"Warning: Requested {num_few_shots} few-shot examples, but only {len(full_dataset)} available. Using all available examples.")
        num_few_shots = len(full_dataset)

    # Go through data
    for item in range(len(full_dataset)):

        few_shot_item = full_dataset[item]

        selected_examples = random.sample(full_dataset, num_few_shots)
        random.shuffle(selected_examples)

        prompt_parts = []

        # Add few-shot example
        for i, example in enumerate(selected_examples):
            prompt_parts.append(f"### Example {i+1} Input:\n{example['input']}")
            prompt_parts.append(f"### Example {i+1} Ouput JSON:\n{example['output']}")

        prompt_parts.append(f"### Current Input:\n{full_dataset[item]['input']}")
        prompt_parts.append(f"### Current Output JSON:")

        prompt_input = "\n".join(prompt_parts)

        few_shot_item['input'] = prompt_input

        few_shot_dataset.append(few_shot_item)

    return few_shot_dataset


if __name__ == "__main__":
    with open("latent_dataset_for_finetuning_v1_tiny.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    few_shot_data = generate_few_shot_prompt_dataset(data, 2)

    with open("latent_dataset_for_finetuning_v1_tiny_twoshot.json", 'w', encoding='utf-8') as f:
        json.dump(few_shot_data, f, indent=2)