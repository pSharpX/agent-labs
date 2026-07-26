from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", '''Answer the question based on the context bellow. If the question cannot be answered using the information provided, answer with "I don\' know"."'''),
    ("human", 'Context: {context}'),
    ("human", 'Question: {question}'),
])