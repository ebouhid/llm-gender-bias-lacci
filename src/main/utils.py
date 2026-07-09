import os
import re
import pickle
import ollama
import asyncio
import hashlib
import itertools
from google import genai
from xai_sdk import Client
from dotenv import load_dotenv
from xai_sdk.chat import user, system
from openai import OpenAI, AsyncOpenAI
from openai import AsyncOpenAI
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
MARITACA_API_KEY = os.getenv("MARITACA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
LOCAL_TIPO_OPENAI_BASE_URL = os.getenv("LOCAL_TIPO_OPENAI_BASE_URL", "http://127.0.0.1:30000/v1")
LOCAL_TIPO_OPENAI_API_KEY = os.getenv("LOCAL_TIPO_OPENAI_API_KEY", "EMPTY")

# Inicializar clientes de API
client_grok = Client(api_key=GROK_API_KEY)
client_ollama = ollama.Client(host=OLLAMA_HOST)
client_openai = OpenAI(api_key=OPEN_AI_API_KEY)
client_genai = genai.Client(api_key=GEMINI_API_KEY)
cliente_maritaca = OpenAI(api_key=MARITACA_API_KEY, base_url="https://chat.maritaca.ai/api")
client_deepinfra = AsyncOpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
client_local_tipo_openai = AsyncOpenAI(
    api_key=LOCAL_TIPO_OPENAI_API_KEY,
    base_url=LOCAL_TIPO_OPENAI_BASE_URL,
)
    

gemini_semaphore = asyncio.Semaphore(10)
deepinfra_semaphore = asyncio.Semaphore(10)

def gerar_chave_cache(modelo, system_prompt, prompt, temperatura, repeticao):
    chave = f"{modelo}|{system_prompt}|{prompt}|{temperatura}|{repeticao}"
    return hashlib.md5(chave.encode()).hexdigest()

def carregar_cache(ARQUIVO_CACHE, logger):
    """Carrega o cache de respostas anteriores."""
    if os.path.exists(ARQUIVO_CACHE):
        try:
            with open(ARQUIVO_CACHE, 'rb') as f:
                cache = pickle.load(f)
                logger.info(f"Cache carregado com {len(cache)} entradas")
                return cache
        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {e}")
            return {}
    return {}

def salvar_cache(cache, ARQUIVO_CACHE, logger):
    """Salva o cache de respostas."""
    try:
        with open(ARQUIVO_CACHE, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"Cache salvo com {len(cache)} entradas")
    except Exception as e:
        logger.error(f"Erro ao salvar cache: {e}")

def atualizar_cache_e_salvar_se_necessario(CONTADOR_NOVAS_RESPOSTAS, chave, valor, cache_respostas, ARQUIVO_CACHE, INTERVALO_SALVAMENTO, logger):
    cache_respostas[chave] = valor
    CONTADOR_NOVAS_RESPOSTAS += 1
    if CONTADOR_NOVAS_RESPOSTAS % INTERVALO_SALVAMENTO == 0:
        logger.info(f"Salvamento incremental ({CONTADOR_NOVAS_RESPOSTAS} novas respostas)...")
        try:
            salvar_cache(cache_respostas, ARQUIVO_CACHE, logger) 
        except Exception as e:
            logger.error(f"Erro no salvamento incremental: {e}")
    return CONTADOR_NOVAS_RESPOSTAS

async def chamar_api_provider(abordagem, modelo, temperatura, system_prompt, user_prompt):
    response_content = ""
    if abordagem == 'ollama':
        response = client_ollama.chat(
            model=modelo,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
            options={'temperature': temperatura}
        )
        response_content = response['message']['content']

    elif abordagem == 'gemini':
        async with gemini_semaphore:
            gemini_model = client_genai.GenerativeModel(model_name=modelo)
            gemini_resposta = await gemini_model.generate_content_async(
                [{"role": "model", "parts": system_prompt}, {"role": "user", "parts": user_prompt}],
                generation_config=client_genai.types.GenerationConfig(temperature=temperatura)
            )
            response_content = gemini_resposta.text

    elif abordagem in ['gpt', 'gpt-sem-temperature']:
        kwargs = {
            "model": modelo,
            "input": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        }
        if abordagem == 'gpt':
             kwargs["temperature"] = temperatura
        
        gpt_resposta = client_openai.responses.create(**kwargs)
        response_content = gpt_resposta.output_text
    elif abordagem == 'deepinfra':
        async with deepinfra_semaphore:
            deepinfra_resposta = await client_deepinfra.chat.completions.create(
                model=modelo,
                temperature=temperatura,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_content = deepinfra_resposta.choices[0].message.content
    elif abordagem == 'maritaca':
        maritaca_resposta = cliente_maritaca.chat.completions.create(
            model=modelo,
            temperature=temperatura,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        response_content = maritaca_resposta.choices[0].message.content
    elif abordagem == 'grok':
        chat = client_grok.chat.create(model=modelo, temperature=temperatura)
        chat.append(system(system_prompt))
        chat.append(user(user_prompt))
        grok_resposta = chat.sample()
        response_content = grok_resposta.content
    elif abordagem in ["local_openai", "sglang"]:
        local_tipo_openai_resposta = await client_local_tipo_openai.chat.completions.create(
            model=modelo,
            temperature=temperatura,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        response_content = local_tipo_openai_resposta.choices[0].message.content

    return response_content

def expandir_templates(templates, lista_de_blocos):
    resultados = []
    vistos = set()

    for template in templates:
        vars_usadas = set(re.findall(r"{(.*?)}", template))

        if not vars_usadas:
            if template not in vistos:
                resultados.append({
                    "texto": template,
                    "chaves_usadas": {}
                })
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
                        resultados.append({
                            "texto": texto,
                            "chaves_usadas": chaves_usadas
                        })
                        vistos.add(texto)

    return resultados

def _to_plain_container(obj):
    """
    Converte objetos do OmegaConf/Hydra para dict/list Python comuns.
    Se o objeto já for Python nativo, retorna como está.
    """
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass

    return obj

def _opcoes_de_chave(chave, valor):
    """
    Normaliza uma chave do YAML em uma lista de alternativas.

    Casos suportados:

    1. Lista simples:
       regiao: ["Norte", "Sul"]

       vira:
       [{"regiao": "Norte"}, {"regiao": "Sul"}]

    2. Lista de dicionários:
       atividade:
         - codigo: "01"
           titulo: "MEMBROS DAS FORÇAS ARMADAS"
           descricao: "serviço nas Forças Armadas"

       vira:
       [{
         "atividade_codigo": "01",
         "atividade_titulo": "...",
         "atividade_descricao": "...",
         "atividade": "serviço nas Forças Armadas"
       }]

    3. Valor escalar:
       pais: "Brasil"

       vira:
       [{"pais": "Brasil"}]
    """
    valor = _to_plain_container(valor)

    if valor is None:
        return []

    if isinstance(valor, list):
        if len(valor) == 0:
            return []

        # Caso: atividade: [{codigo, titulo, descricao}, ...]
        if all(isinstance(item, dict) for item in valor):
            opcoes = []

            for item in valor:
                item = _to_plain_container(item)

                opcao = {
                    f"{chave}_{subchave}": subvalor
                    for subchave, subvalor in item.items()
                }

                # Atalho opcional:
                # permite usar {atividade} no prompt, caso você queira.
                opcao[chave] = item.get(
                    "descricao",
                    item.get("titulo", str(item))
                )

                opcoes.append(opcao)

            return opcoes

        # Caso: regiao: ["Norte", "Nordeste", ...]
        return [{chave: item} for item in valor]

    if isinstance(valor, dict):
        opcao = {
            f"{chave}_{subchave}": subvalor
            for subchave, subvalor in valor.items()
        }

        opcao[chave] = valor.get(
            "descricao",
            valor.get("titulo", str(valor))
        )

        return [opcao]

    return [{chave: valor}]

def _expandir_bloco(bloco):
    """
    Expande um bloco de chaves.

    Exemplo:
    {
      "regiao": ["Norte", "Sul"],
      "sexo": ["masculino", "feminino"]
    }

    vira produto cartesiano:
    Norte x masculino
    Norte x feminino
    Sul x masculino
    Sul x feminino
    """
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
    """
    Expande todos os blocos de chaves.

    Exemplo:
    CHAVES_PROMPT:
      - atividade: [...]
      - regiao: [...]

    Gera todas as combinações entre os blocos.
    """
    lista_de_blocos = _to_plain_container(lista_de_blocos) or []

    if isinstance(lista_de_blocos, dict):
        lista_de_blocos = [lista_de_blocos]

    opcoes_por_bloco = [
        _expandir_bloco(bloco)
        for bloco in lista_de_blocos
    ]

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
    """
    Expande templates com variáveis entre chaves.

    Exemplo de template:
        "Descreva alguém com atividade {atividade_descricao}"

    Exemplo de saída:
        {
          "texto": "Descreva alguém com atividade serviço nas Forças Armadas",
          "chaves_usadas": {
              "atividade_codigo": "01",
              "atividade_titulo": "MEMBROS DAS FORÇAS ARMADAS",
              "atividade_descricao": "serviço nas Forças Armadas",
              "atividade": "serviço nas Forças Armadas"
          }
        }
    """
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
                resultados.append({
                    "texto": texto,
                    # Salva todas as chaves do registro selecionado,
                    # não apenas a variável textual usada no prompt.
                    "chaves_usadas": dict(chaves)
                })
                vistos.add(texto)

    return resultados