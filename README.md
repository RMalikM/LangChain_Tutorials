# LangChain_Tutorials
Simple Code Examples for components in LangChain

## Usage
1. Create a Python virtual environment:

    **Using Conda:**

        conda create --name <env_name> python=3.10
        conda activate env_name

    **Using venv:**

        python3 -m venv <env_name>
        source env_name/bin/activate

        # On Windows use `env_name\Scripts\activate`

3. Install required dependencies:

        pip install -r requirements.txt

## Environment Variables
Before running any script create a `.env` file and add the API Token keys for the models whichever you want to use. 

    OPENAI_API_KEY = ""
    ANTHROPIC_API_KEY = ""
    GOOGLE_API_KEY = ""
    HUGGINGFACEHUB_API_TOKEN = ""

## Runing Scripts
Each script can be run individually. For example, to run OpenAI ChatModel use: 

    python Chat_Models/chatmodel_openai.py


**Note:** Edit the scripts and try with different examples.