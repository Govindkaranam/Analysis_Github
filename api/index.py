!pip install flask nbformat requests gitpython torch transformers
from flask import Flask, render_template, request
import os
import shutil
import nbformat
import requests
import git
import torch
from nbformat import read, write, validate
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Fetch user repositories from GitHub
def fetch_user_repositories(github_url):
    # Extract the username from the user URL
    username = github_url.split('/')[-1]

    # GitHub API endpoint to retrieve user repositories
    api_url = f'https://api.github.com/users/{username}/repos'

    # Make an API request to retrieve repository data
    response = requests.get(api_url)
    repos = response.json()

    # Iterate through the repositories
    for repo in repos:
        clone_url = repo['clone_url']
        repo_name = repo['name']
        repo_directory = os.path.join("./temp", repo_name)

        # Check if the repository directory already exists
        if os.path.exists(repo_directory):
            print(f"Updating repository: {repo_name}")
            repo = git.Repo(repo_directory)

            # Pull the latest changes from the remote repository
            origin = repo.remote(name='origin')
            origin.pull()

        else:
            print(f"Cloning repository: {repo_name}")
            # Clone the repository if it doesn't exist locally
            git.Repo.clone_from(clone_url, repo_directory)

    # Return the list of repository names
    return [repo['name'] for repo in repos]


# Preprocess code in repositories
def preprocess_code(repository):
    # Define a temporary directory to store preprocessed files
    temp_directory = "./temp"
    os.makedirs(temp_directory, exist_ok=True)

    # Clone the repository locally if it doesn't already exist
    repo_directory = os.path.join(temp_directory, repository)
    if not os.path.exists(repo_directory):
        clone_command = f"git clone https://github.com/{repository}.git {repo_directory}"
        os.system(clone_command)

    # Iterate through the repository files
    for root, dirs, files in os.walk(repo_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1]

            # Preprocess specific file types
            if file_extension == ".ipynb":
                preprocess_jupyter_notebook(file_path)
            elif file_extension == ".py":
                preprocess_python_file(file_path)

    # Remove temporary directory
    shutil.rmtree(temp_directory)


# Preprocess Jupyter notebook file
def preprocess_jupyter_notebook(file_path):
    # Read the Jupyter notebook
    with open(file_path, 'r') as file:
        notebook = read(file, as_version=4)

    # Normalize the notebook to add the missing ID field
    validate(notebook)

    # Remove outputs from all cells
    for cell in notebook.cells:
        if 'outputs' in cell:
            cell['outputs'] = []

    # Save the preprocessed Jupyter notebook
    with open(file_path, 'w') as file:
        write(notebook, file, version=4)


# Preprocess Python file
def preprocess_python_file(file_path):
    # Read the Python file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Remove comments and whitespace
    lines = [line for line in lines if not line.strip().startswith("#")]

    # Save the preprocessed Python file
    with open(file_path, "w") as file:
        file.writelines(lines)


# Assess technical complexity using GPT
def assess_technical_complexity(code):
    try:
        # Define your own GPT2 model and tokenizer or load a pre-trained model
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Apply prompt engineering techniques to evaluate the code's technical complexity
        prompt = "The code provided is:"
        input_text = f"{prompt} {code}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Set attention mask and pad token ID
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = tokenizer.eos_token_id

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            max_length=100,
            num_return_sequences=1
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Compute the complexity score based on generated text (e.g., using LangChain)
        complexity_score = compute_complexity_score(generated_text)

        return complexity_score

    except Exception as e:
        print(f"Error assessing technical complexity: {e}")
        return None



def compute_complexity_score(text):

    return complexity_score


# Identify most technically complex repository
def identify_most_complex_repository(repositories):
    most_complex_repository = None
    max_complexity_score = 0

    for repository in repositories:
        preprocess_code(repository)

        # Assess the technical complexity of the code in the repository
        code = get_repository_code(repository)
        complexity_score = assess_technical_complexity(code)

        if complexity_score is not None and complexity_score > max_complexity_score:
            max_complexity_score = complexity_score
            most_complex_repository = repository

    return most_complex_repository


# Get the code from the repository
def get_repository_code(repository):
    # Define the directory where the repository is cloned
    repo_directory = f"./temp/{repository}"

    # Iterate through the repository files
    code = ""
    for root, dirs, files in os.walk(repo_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1]

            # Read and append code from specific file types
            if file_extension == ".ipynb" or file_extension == ".py":
                with open(file_path, "r") as file:
                    code += file.read()

    return code


# Generate justification using GPT
def generate_justification(repository):
    try:
        # Define your own GPT2 model and tokenizer or load a pre-trained model
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Craft prompts based on the assessment and the selected repository
        prompt = f"The repository {repository} was selected as the most technically complex because:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Set attention mask and pad token ID
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = tokenizer.eos_token_id

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            max_length=200,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_text

    except Exception as e:
        print(f"Error generating justification: {e}")
        return None


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    github_url = request.form['github_url']

    # Fetch user repositories from GitHub
    repositories = fetch_user_repositories(github_url)

    # Identify the most technically complex repository
    most_complex_repository = identify_most_complex_repository(repositories)

    if most_complex_repository is not None:
        # Generate justification for the selected repository
        justification = generate_justification(most_complex_repository)

        return render_template('result.html', repository=most_complex_repository, justification=justification)
    else:
        return render_template('result.html', repository=None, justification=None)


if __name__ == '__main__':
    app.run(debug=True)
