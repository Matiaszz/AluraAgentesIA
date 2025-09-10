# type: ignore
from .config.settings import load_google_api_key
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict
from models.models import TriageOutput

GOOGLE_API_KEY = load_google_api_key()

triage_llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0,
    api_key=GOOGLE_API_KEY
)

triage_prompt = "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
"Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
"{\n"
'  "decision": "AUTO_RESOLVE" | "ASK_INFO" | "OPEN_TICKET",\n'
'  "urgency": "LOW" | "MEDIA" | "ALTA",\n'
'  "missing_fields": ["..."]\n'
"}\n"
"Regras:\n"
'- **AUTO_RESOLVE**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
'- **ASK_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
'- **OPEN_TICKET**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
"Analise a mensagem e decida a ação mais apropriada."

triage_chain = triage_llm.with_structured_output(TriageOutput)


def triage(message: str) -> Dict:
    output: TriageOutput = triage_chain.invoke([
        SystemMessage(content=triage_prompt),
        HumanMessage(content=message)
    ])

    return output.model_dump()
