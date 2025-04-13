from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader(
    path='data_files/books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# # In-memory load (only for few pdfs)
# docs1 = loader.load()
# print(len(docs1))
# print(docs1[0].page_content)
# print(docs1[0].metadata)


# Lazy Loader
docs2 = loader.lazy_load()

for document in docs2:
    print(document.metadata)


