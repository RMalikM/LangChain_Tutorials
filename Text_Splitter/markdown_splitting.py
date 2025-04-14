from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
# Project Name: Simple Task Manager

A simple Python-based project to manage and track daily tasks, including their priority, deadlines, and completion status.


## Features

- Add new tasks with relevant info
- View task list and details
- Mark tasks as completed
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/simple-task-manager.git

"""


splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)