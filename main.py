# type: ignore
from agents.triage_model import triage


def main():
    tests = ['Qual o nome da empresa?',
             'Posso reembolsar a internet?',
             'Como funciona a política de alimentação em viagens?',
             'Posso trabalhar de casa na sexta-feira de acordo com a política de home office?',
             'quero ter mais 5 dias de trabalho remoto, como faço? ',
             'posso reembolsar cursos ou treinamentos da alura?',
             'quantas capivaras tem no rio de tietê?']

    for i in tests:
        res = triage(i)
        print(f'Prompt: {i}\n -> {res}')


if __name__ == "__main__":
    main()
