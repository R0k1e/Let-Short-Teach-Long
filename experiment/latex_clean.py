import re
import json



# Function to clean LaTeX content by removing commands and environments
def clean_latex(latex_str):
    # Remove comments
    cleaned = re.sub(r'%.*?\n', '', latex_str)
    # Remove commands
    cleaned = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', cleaned)
    # Remove remaining single braces and commands
    cleaned = re.sub(r'[{}\\]', '', cleaned)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    return cleaned



input_path = './acl_latex.tex'
output_path = './acl_latex_cleaned.json'

with open(input_path, 'r', encoding='utf-8') as file:
    latex_content = file.read()
# Clean the LaTeX content
cleaned_content = clean_latex(latex_content)
cleaned_content = {"text": cleaned_content}
# Write the cleaned content to a JSON file
with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(cleaned_content, file, ensure_ascii=False)
