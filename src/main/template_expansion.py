"""Prompt template expansion without API client dependencies."""

from __future__ import annotations

import itertools
import re


def _to_plain_container(obj):
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass
    return obj


def expandir_templates(templates, lista_de_blocos):
    resultados = []
    vistos = set()

    for template in templates:
        vars_usadas = set(re.findall(r"{(.*?)}", template))

        if not vars_usadas:
            if template not in vistos:
                resultados.append({"texto": template, "chaves_usadas": {}})
                vistos.add(template)
            continue

        for r in range(1, len(lista_de_blocos) + 1):
            for combo_blocos in itertools.combinations(lista_de_blocos, r):
                chaves_combo = {}
                for bloco in combo_blocos:
                    chaves_combo.update(bloco)

                if not vars_usadas.issubset(chaves_combo.keys()):
                    continue

                valores = [chaves_combo[var] for var in vars_usadas]

                for combo_valores in itertools.product(*valores):
                    texto = template
                    chaves_usadas = {}

                    for var, val in zip(vars_usadas, combo_valores):
                        texto = texto.replace(f"{{{var}}}", val)
                        chaves_usadas[var] = val

                    if texto not in vistos:
                        resultados.append(
                            {"texto": texto, "chaves_usadas": chaves_usadas}
                        )
                        vistos.add(texto)

    return resultados


def _opcoes_de_chave(chave, valor):
    valor = _to_plain_container(valor)

    if valor is None:
        return []

    if isinstance(valor, list):
        if len(valor) == 0:
            return []

        if all(isinstance(item, dict) for item in valor):
            opcoes = []
            for item in valor:
                item = _to_plain_container(item)
                opcao = {
                    f"{chave}_{subchave}": subvalor
                    for subchave, subvalor in item.items()
                }
                opcao[chave] = item.get("descricao", item.get("titulo", str(item)))
                opcoes.append(opcao)
            return opcoes

        return [{chave: item} for item in valor]

    if isinstance(valor, dict):
        opcao = {
            f"{chave}_{subchave}": subvalor for subchave, subvalor in valor.items()
        }
        opcao[chave] = valor.get("descricao", valor.get("titulo", str(valor)))
        return [opcao]

    return [{chave: valor}]


def _expandir_bloco(bloco):
    bloco = _to_plain_container(bloco)
    if not bloco:
        return [{}]

    listas_de_opcoes = []
    for chave, valor in bloco.items():
        opcoes = _opcoes_de_chave(chave, valor)
        if opcoes:
            listas_de_opcoes.append(opcoes)

    if not listas_de_opcoes:
        return [{}]

    alternativas = []
    for combo in itertools.product(*listas_de_opcoes):
        merged = {}
        for opcao in combo:
            merged.update(opcao)
        alternativas.append(merged)
    return alternativas


def _expandir_chaves(lista_de_blocos):
    lista_de_blocos = _to_plain_container(lista_de_blocos) or []
    if isinstance(lista_de_blocos, dict):
        lista_de_blocos = [lista_de_blocos]

    opcoes_por_bloco = [_expandir_bloco(bloco) for bloco in lista_de_blocos]
    if not opcoes_por_bloco:
        return [{}]

    combinacoes = []
    for combo in itertools.product(*opcoes_por_bloco):
        merged = {}
        for opcao in combo:
            merged.update(opcao)
        combinacoes.append(merged)
    return combinacoes


def expandir_templates_v2(templates, lista_de_blocos):
    templates = _to_plain_container(templates) or []
    if isinstance(templates, str):
        templates = [templates]

    chaves_expandidas = _expandir_chaves(lista_de_blocos)
    resultados = []
    vistos = set()

    for template in templates:
        vars_usadas = list(dict.fromkeys(re.findall(r"{(.*?)}", template)))

        for chaves in chaves_expandidas:
            if not set(vars_usadas).issubset(chaves.keys()):
                continue

            texto = template
            for var in vars_usadas:
                texto = texto.replace(f"{{{var}}}", str(chaves[var]))

            if texto not in vistos:
                resultados.append({"texto": texto, "chaves_usadas": dict(chaves)})
                vistos.add(texto)

    return resultados
