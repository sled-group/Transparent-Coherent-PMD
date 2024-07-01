import os
import json
import sys

def find_and_replace_in_json(file_path, find_str, replace_str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def replace_strings(obj):
        if isinstance(obj, dict):
            return {k: replace_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_strings(item) for item in obj]
        elif isinstance(obj, str):
            return obj.replace(find_str, replace_str)
        else:
            return obj

    updated_data = replace_strings(data)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=4)

def traverse_directory(directory, find_str, replace_str):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                find_and_replace_in_json(file_path, find_str, replace_str)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <directory> <find_string> <replace_string>")
        sys.exit(1)

    directory = sys.argv[1]
    find_string = sys.argv[2]
    replace_string = sys.argv[3]

    traverse_directory(directory, find_string, replace_string)
    print(f"Find and replace for {directory} complete.")
