from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class retriever():
  def __init__(self, docs,persist_directory):
    self.docs=docs
    self.embeddings = HuggingFaceEmbeddings()
    #client_settings = {"chroma_tenant": "default_tenant", "database": "in-memory"}
    self.vectorstore = Chroma.from_documents(
      documents=docs,
      embedding=self.embeddings,
       persist_directory=persist_directory
    )
    #self.vectorstore = Chroma.from_documents(documents=docs, embedding=self.embeddings)

  def get_retriever(self, search_type="similarity",kwargs=3):
    return self.vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": kwargs})
