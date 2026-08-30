from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1")
response = llm.invoke("What is agentic RAG in one sentence?")
print(response.content)
