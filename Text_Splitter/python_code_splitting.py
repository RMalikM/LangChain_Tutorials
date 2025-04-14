from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages  # Pages is an integer (like 250 or 432)

    def get_title(self):
        return self.title

    def is_lengthy(self):
        return self.pages > 300


# Example usage
book1 = Book("The Silent Patient", "Alex Michaelides", 336)
print(book1.get_title())

if book1.is_lengthy():
    print("The book is lengthy.")
else:
    print("The book is not lengthy.")


"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
for i in range(len(chunks)):
    print("\n", chunks[i])
