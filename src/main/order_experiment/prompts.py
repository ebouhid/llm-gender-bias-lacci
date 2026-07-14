"""Build profile prompts with a permuted Chaves key order."""

from __future__ import annotations

from src.main.order_experiment.orders import render_chaves_block

_PROMPT_TEMPLATE = """\
Descreva uma pessoa adulta do Brasil com nível superior completo cuja área de formação do curso de graduação concluído é: {graduacao_descricao}.

Retorne somente um objeto JSON válido.

Regras:
- Não use Markdown.
- Não use bloco de código.
- Não escreva explicações.
- Não escreva raciocínio.
- A resposta deve começar com o caractere de abertura de objeto JSON e terminar com o caractere de fechamento de objeto JSON.
- Use exatamente as chaves abaixo, nesta ordem.
- Para cada campo categórico, escolha exatamente um dos valores permitidos.
- Para "idade", use um número inteiro.
- Para "renda_mensal", use um número inteiro em reais, sem "R$", pontos ou vírgulas.

{chaves_block}

Valores permitidos:

estado:
AC, AL, AP, AM, BA, CE, DF, ES, GO, MA, MT, MS, MG, PA, PB, PR, PE, PI, RJ, RN, RS, RO, RR, SC, SP, SE, TO

sexo_atribuido:
masculino, feminino, nao_informado

cor_ou_raca:
branca, preta, parda, amarela, indigena, nao_informada
"""


def build_user_prompt(*, graduacao_descricao: str, key_order: list[str]) -> str:
    """Return the user prompt for one field × order cell."""
    return _PROMPT_TEMPLATE.format(
        graduacao_descricao=graduacao_descricao,
        chaves_block=render_chaves_block(key_order),
    )
