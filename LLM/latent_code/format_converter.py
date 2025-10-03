import json

def convert_scene_to_alpaca_format(raw_data_list, output_filepath="alpaca_finetuning_data.json"):
    """
    Converts a list of raw scene graph data into the Alpaca fine-tuning format.

    Args:
        raw_data_list (list): A list of dictionaries, where each dictionary
                              contains 'text' and 'scene_graph' keys.
        output_filepath (str): The path to save the converted JSON data.
    """

    alpaca_instruction = (
        "You are an expert scene graph generator. Your task is to convert a natural language "
        "description of a room into a structured JSON scene graph. The scene graph should "
        "identify all named entities (objects) in the room with their class, description, "
        "and pose (position, rotation, and scale), and describe all significant relationships "
        "between these entities."
    )

    formatted_data = []

    for item in raw_data_list:
        text_description = item['text']
        
        # Extract only 'entities' and 'relationships' from the scene_graph
        scene_graph_output = {
            "entities": item['scene_graph']['entities'],
            "relationships": item['scene_graph']['relationships']
        }
        
        # Convert the scene_graph_output dictionary to a JSON string
        # and ensure it's on a single line for typical fine-tuning loaders
        json_output_string = json.dumps(scene_graph_output, separators=(',', ':'))

        alpaca_entry = {
            "instruction": alpaca_instruction,
            "input": text_description,
            "output": json_output_string
        }
        formatted_data.append(alpaca_entry)

    # Save the formatted data to a JSON file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully converted {len(formatted_data)} entries to Alpaca format.")
    print(f"Saved to: {output_filepath}")


def convert_latent_to_alpaca_format(raw_data_list, output_filepath="alpaca_latent_finetuning_data.json"):
    """
    Converts a list of raw VQ-VAE latent index data into the Alpaca fine-tuning format.

    Args:
        raw_data_list (list): A list of dictionaries, where each dictionary
                              contains 'text' and 'predicted_indices' keys.
                              'predicted_indices' is expected to be a dictionary mapping
                              object names to a list of 64 indices.
        output_filepath (str): The path to save the converted JSON data.
    """

    alpaca_instruction = (
        "You are a 3D scene generator. Given a room description, generate the quantized latent "
        "indices for each object mentioned. Each object should be represented by exactly 64 "
        "indices from the VQ-VAE codebook (range 0-2047). Output in JSON format with object "
        "names as keys and arrays of 64 integers as values. "
        "Ensure the output is a *single, unformatted JSON string*."
    )

    formatted_data = []

    for item in raw_data_list:
        text_description = item['text']
        
        # The 'predicted_indices' dictionary is already in the desired output structure
        # for your model's response.
        vq_vae_output = item['predicted_indices']
        
        # Convert the VQ-VAE output dictionary to a JSON string
        # using separators to ensure it's unformatted (no extra whitespace/newlines)
        json_output_string = json.dumps(vq_vae_output, separators=(',', ':'))

        alpaca_entry = {
            "instruction": alpaca_instruction,
            "input": text_description,
            "output": json_output_string
        }
        formatted_data.append(alpaca_entry)

    # Save the formatted data to a JSON file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully converted {len(formatted_data)} entries to Alpaca VQ-VAE format.")
    print(f"Saved to: {output_filepath}")

def convert_latent_to_alpaca_format_v2(raw_data_list, output_filepath="alpaca_latent_finetuning_data.json"):
    """
    Converts a list of raw VQ-VAE latent index data into the Alpaca fine-tuning format.

    Args:
        raw_data_list (list): A list of dictionaries, where each dictionary
                              contains 'text' and 'predicted_indices' keys.
                              'predicted_indices' is expected to be a dictionary mapping
                              object names to a list of 64 indices.
        output_filepath (str): The path to save the converted JSON data.
    """

    alpaca_instruction = (
        "You are an expert 3D scene generator. Your task is to process room descriptions and "
        "generate quantized latent indices for objects.Given a room description, generate the "
        "quantized latent indices for each object mentioned. Each object should be represented "
        "by exactly 64 indices from the VQ-VAE codebook (range 0-2047). Output in JSON format "
        "with object names as keys and arrays of 64 integers as values. If there are multiple "
        "objects of the same type, append a numerical suffix (e.g., 'table', 'table_2', 'table_3'). "
        "Ensure the output is a *single, unformatted JSON string* and contains *only* the JSON string."
    )

    formatted_data = []

    for item in raw_data_list:
        text_description = item['text']
        
        # The 'predicted_indices' dictionary is already in the desired output structure
        # for your model's response.
        vq_vae_output = item['predicted_indices']
        
        # Convert the VQ-VAE output dictionary to a JSON string
        # using separators to ensure it's unformatted (no extra whitespace/newlines)
        json_output_string = json.dumps(vq_vae_output, separators=(',', ':'))

        alpaca_entry = {
            "instruction": alpaca_instruction,
            "input": text_description,
            "output": json_output_string
        }
        formatted_data.append(alpaca_entry)

    # Save the formatted data to a JSON file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully converted {len(formatted_data)} entries to Alpaca VQ-VAE format.")
    print(f"Saved to: {output_filepath}")