# type: ignore
from agents.triage_model import triage, triage_llm
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agents.embedding_model import embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Dict


def loadDocuments():
    docs = []
    for i in Path('./docs/pdfs').glob('*.pdf'):
        try:
            loader = PyMuPDFLoader(str(i))
            docs.extend(loader.load())
            print(f'loaded: {i.name}')
        except e:
            print(e)

    print(f'Total docs loaded: {len(docs)}')
    return docs


def main():
    # tests = ['Qual o nome da empresa?',
    #          'Posso reembolsar a internet?',
    #          'Como funciona a política de alimentação em viagens?',
    #          'Posso trabalhar de casa na sexta-feira de acordo com a política de home office?',
    #          'quero ter mais 5 dias de trabalho remoto, como faço? ',
    #          'posso reembolsar cursos ou treinamentos da alura?',
    #          'quantas capivaras tem no rio de tietê?']

    # for i in tests:
    #     res = triage(i)
    #     print(f'Prompt: {i}\n -> {res}')

    docs = loadDocuments()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_type='similarity_score_threshold',
                                         search_kwargs={
                                             'score_threshold': 0.3,
                                             'k': 4
                                         })

    context = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
         "Responda SOMENTE com base no contexto fornecido. "
         "Se não houver base suficiente, responda apenas 'Não sei'."),

        ("human", "Pergunta: {input}\n\nContexto:\n{context}")
    ])

    doc_chain = create_stuff_documents_chain(triage_llm, context)

    def ask_rag_police(question: str) -> Dict:
        related_docs = retriever.invoke(question)

        if not related_docs:
            return {
                'answer': 'Não sei',
                'mentions': [],
                'found_context': False
            }

        answer = doc_chain.invoke({'input': question, 'context': related_docs})

        txt = (answer or '').strip()

        if txt.rstrip('.!?') == 'Não sei':
            return {
                'answer': 'Não sei',
                'mentions': [],
                'found_context': False
            }

        return {
            'answer': txt,
            'mentions': related_docs,
            'found_context': True
        }

    tests = ['Qual o nome da empresa?',
             'Posso reembolsar a internet?',
             'Como funciona a política de alimentação em viagens?',
             'Posso trabalhar de casa na sexta-feira de acordo com a política de home office?',
             'quero ter mais 5 dias de trabalho remoto, como faço? ',
             'posso reembolsar cursos ou treinamentos da alura?',
             'quantas capivaras tem no rio de tietê?']

    for i in tests:
        res = ask_rag_police(i)
        print(
            f'Prompt: {i}\n -> {res["answer"]}\n\n')


if __name__ == "__main__":

    main()
