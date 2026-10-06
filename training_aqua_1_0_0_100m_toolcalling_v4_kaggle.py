# ============================================================
# AQUA 1.0.0 — ~100M LLaMA desde cero
# Español · chat-first · contexto 10.024 · tool calling
# Optimizado para Kaggle/T4 · sin subida automática a HF
# ============================================================

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import glob
import json
import math
import queue
import random
import re
import shutil
import signal
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoModelForCausalLM, LlamaConfig

# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 42
LIMITE_H = float(os.environ.get("AQUA_HOURS", "11.5"))
HORAS_SFT = float(os.environ.get("AQUA_SFT_HOURS", "1.7"))

# El modelo admite 10.024 tokens en inferencia.
# El entrenamiento mantiene bloques más cortos para conservar velocidad.
CONTEXT = 10024
# Ventana larga para inferencia; entrenamiento eficiente por bloques.
LONG_CONTEXT_RESERVE = 1024
TRAIN_BLOCK = int(os.environ.get("AQUA_TRAIN_BLOCK", "2048"))
# No se fuerza la ventana completa durante el pretraining: preserva throughput en T4.

VOCAB = 32000
TOK_PASO = 16384

LR_PRE_MAX = 4e-4
LR_PRE_MIN = 4e-5
WARMUP = 500

LR_SFT_MAX = 3e-5
LR_SFT_MIN = 3e-6
SFT_WARM = 60

CHAT_MIX0 = 0.22
CHAT_MIX1 = 0.55

SINTETICOS = 70000
TOOL_SYNTH = 30000

SHAREGPT = True
SHAREGPT_REP = 3

SFT_MAXB = 32
SFT_EPOCAS_MAX = 3

EVAL_CADA = 350
CKPT_MIN = 15

USAR_COMPILE = os.environ.get("AQUA_COMPILE", "1") == "1"

ARCH = "aqua-1.0.0-100m-toolcalling-v1"

ROL_USER = "USUARIO"
ROL_BOT = "ASISTENTE"

# Tokens especiales para tool calling.
SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<tool>",
    "</tool>",
    "<tool_call>",
    "</tool_call>",
    "<tool_result>",
    "</tool_result>",
    "<tools>",
    "</tools>",
]

GEN = dict(
    temperature=0.72,
    top_p=0.92,
    top_k=50,
    repetition_penalty=1.08,
    no_repeat_ngram_size=3,
)

SALIDA = "/kaggle/working"
DATASET_JSONL = os.path.join(SALIDA, "aqua_dataset.jsonl")
TOK_DIR = os.path.join(SALIDA, "tokenizer")
CKPT = os.path.join(SALIDA, "ckpt.pt")
FINAL = os.path.join(SALIDA, "modelo_final")
ZIP = os.path.join(SALIDA, "aqua_1.0.0_100m_toolcalling.zip")

T0 = time.time()

# ============================================================
# HARDWARE
# ============================================================

dev = "cuda" if torch.cuda.is_available() else "cpu"
usa_amp = dev == "cuda"

if usa_amp:
    major, _ = torch.cuda.get_device_capability()
    AMP_DTYPE = torch.bfloat16 if major >= 8 else torch.float16
else:
    AMP_DTYPE = torch.float32

random.seed(SEED)
torch.manual_seed(SEED)

if usa_amp:
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision("high")

print(
    f"dev={dev} | amp={AMP_DTYPE} | límite={LIMITE_H}h | "
    f"context={CONTEXT} | train_block={TRAIN_BLOCK}"
)

# ============================================================
# UTILIDADES DE TEXTO
# ============================================================

RE_W = re.compile(r"\w+", re.UNICODE)

def normalize_text(x):
    if not isinstance(x, str):
        return ""
    x = x.replace("\x00", " ")
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[ \t\u00a0]+", " ", x)
    x = re.sub(r" ?\n ?", "\n", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()

def words(x):
    return RE_W.findall(x.lower())

def repeated_ngram_ratio(ws, n=3):
    if len(ws) < n * 2:
        return 0.0
    grams = [tuple(ws[i:i+n]) for i in range(len(ws) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))

def quality_pair(user, assistant):
    user = normalize_text(user)
    assistant = normalize_text(assistant)

    if len(user) < 3 or len(assistant) < 5:
        return None

    uw = words(user)
    aw = words(assistant)

    if len(uw) < 3 and len(aw) < 10:
        return None

    if len(uw) + len(aw) < 8:
        return None

    ratio = len(aw) / max(1, len(uw))
    if ratio > 15.0 or ratio < 0.10:
        return None

    if repeated_ngram_ratio(uw) > 0.40:
        return None

    if repeated_ngram_ratio(aw) > 0.40:
        return None

    if assistant.count("http://") + assistant.count("https://") > 10:
        return None

    if assistant.count("\ufffd") > 0:
        return None

    if assistant.count("Ã") > 3:
        return None

    if len(set(assistant)) < 8 and len(assistant) > 40:
        return None

    return user, assistant

def quality_conv(turns):
    if not isinstance(turns, list):
        return None

    clean = []
    for x in turns:
        x = normalize_text(x)
        if x:
            clean.append(x)

    if len(clean) < 2:
        return None

    if len(clean) % 2:
        clean = clean[:-1]

    out = []
    for i in range(0, len(clean), 2):
        pair = quality_pair(clean[i], clean[i + 1])
        if pair is None:
            return None
        out.extend(pair)

    return out if out else None

def stable_key(conv):
    return "\x1f".join(conv).lower().strip()

def dedup(convs, seen):
    out = []
    for c in convs:
        k = stable_key(c)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out

# ============================================================
# PARSER CHAT / SHAREGPT
# ============================================================

USER_ROLES = {"human", "user", "prompter"}
BOT_ROLES = {"gpt", "assistant", "bot", "model"}

def parse_sharegpt_item(item):
    if not isinstance(item, dict):
        return None

    conv = item.get("conversations") or item.get("messages")
    if not isinstance(conv, list):
        return None

    turns = []
    current_role = None

    for t in conv:
        if not isinstance(t, dict):
            continue

        role = str(
            t.get("from")
            or t.get("role")
            or t.get("speaker")
            or ""
        ).lower().strip()

        value = normalize_text(
            t.get("value")
            or t.get("content")
            or t.get("text")
            or ""
        )

        if not value:
            continue

        if role in USER_ROLES:
            new_role = "user"
        elif role in BOT_ROLES:
            new_role = "assistant"
        else:
            continue

        if new_role == current_role and turns:
            turns[-1] += "\n\n" + value
        else:
            turns.append(value)
            current_role = new_role

    if len(turns) < 2:
        return None

    return quality_conv(turns)

def parse_sharegpt_text(txt):
    result = []

    try:
        data = json.loads(txt)
        objs = data if isinstance(data, list) else [data]
    except Exception:
        objs = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except Exception:
                continue

    for obj in objs:
        c = parse_sharegpt_item(obj)
        if c:
            result.append(c)

    return result

# ============================================================
# SHAREGPT ROBUSTO
# ============================================================

def load_sharegpt():
    repo = "FreedomIntelligence/sharegpt-spanish"
    result = []

    api_url = f"https://huggingface.co/api/datasets/{repo}"

    print("\n=== SHAREGPT ===")
    print(f"Consultando API: {api_url}")

    try:
        import requests

        r = requests.get(api_url, timeout=30)
        r.raise_for_status()
        meta = r.json()

        siblings = meta.get("siblings", [])
        files = [
            x.get("rfilename", "")
            for x in siblings
            if isinstance(x, dict)
        ]
        files = [
            f for f in files
            if f.lower().endswith((".json", ".jsonl"))
        ]

        if not files:
            raise RuntimeError("La API no devolvió ningún .json/.jsonl")

        print("Archivos detectados:", files[:10])

        for filename in files[:3]:
            url = (
                "https://huggingface.co/datasets/"
                f"{repo}/resolve/main/{filename}?download=true"
            )

            print(f"Descargando: {url}")

            rr = requests.get(url, timeout=120)
            rr.raise_for_status()

            size_mb = len(rr.content) / 1e6
            print(f"Tamaño: {size_mb:.1f} MB")

            result.extend(parse_sharegpt_text(rr.text))

            if result:
                break

    except Exception as e:
        print(f"API/descarga ShareGPT falló: {type(e).__name__}: {e}")

    if not result:
        fallback = [
            "https://huggingface.co/datasets/"
            "FreedomIntelligence/sharegpt-spanish/resolve/main/"
            "data/train.json",
            "https://huggingface.co/datasets/"
            "FreedomIntelligence/sharegpt-spanish/resolve/main/"
            "sharegpt_spanish.json",
            "https://huggingface.co/datasets/"
            "FreedomIntelligence/sharegpt-spanish/resolve/main/"
            "data.json",
        ]

        try:
            import requests

            for url in fallback:
                try:
                    print(f"Fallback ShareGPT: {url}")
                    r = requests.get(url, timeout=120)
                    r.raise_for_status()
                    print(f"Tamaño: {len(r.content)/1e6:.1f} MB")
                    result = parse_sharegpt_text(r.text)
                    if result:
                        break
                except Exception as e:
                    print(f"Fallback falló: {type(e).__name__}")

        except Exception as e:
            print(f"requests no disponible: {e}")

    if not result:
        raise RuntimeError(
            "ShareGPT no pudo cargarse. El entrenamiento se ABORTA "
            "para evitar entrenar accidentalmente solo con datos sintéticos."
        )

    print(f"ShareGPT válido: {len(result):,} conversaciones")
    return result

# ============================================================
# DATASET CHAT GENERAL
# ============================================================

def load_chat_source(nombre, config=None, split="train", cap=50000):
    result = []

    try:
        from datasets import load_dataset

        attempts = []
        if config:
            attempts.append((nombre, config, split))
            attempts.append((nombre, config, "train_sft"))
        else:
            attempts.append((nombre, None, split))
            attempts.append((nombre, None, "train_sft"))

        ds = None

        for name, conf, sp in attempts:
            try:
                if conf:
                    ds = load_dataset(
                        name,
                        conf,
                        split=sp,
                        streaming=True,
                    )
                else:
                    ds = load_dataset(
                        name,
                        split=sp,
                        streaming=True,
                    )
                break
            except Exception:
                continue

        if ds is None:
            raise RuntimeError("No se pudo abrir ningún split")

        try:
            ds = ds.shuffle(seed=SEED, buffer_size=10000)
        except Exception:
            pass

        for ex in ds:
            c = parse_sharegpt_item(ex)
            if c:
                result.append(c)

            if len(result) >= cap:
                break

        print(f"{nombre}: {len(result):,}")

    except Exception as e:
        print(f"{nombre} omitido: {type(e).__name__}: {e}")

    return result

# ============================================================
# DATOS SINTÉTICOS DE ALTA UTILIDAD
# ============================================================

def sinteticos_aritmetica(n, seed=5):
    rng = random.Random(seed)
    out = []

    for _ in range(n):
        tipo = rng.choice(["+", "-", "*", "/", "%"])

        if tipo == "+":
            a, b = rng.randint(0, 9999), rng.randint(0, 9999)
            out.append([
                f"¿Cuánto es {a} + {b}?",
                f"{a} + {b} = {a+b}."
            ])

        elif tipo == "-":
            a, b = sorted(
                (rng.randint(0, 9999), rng.randint(0, 9999)),
                reverse=True,
            )
            out.append([
                f"¿Cuánto es {a} - {b}?",
                f"{a} - {b} = {a-b}."
            ])

        elif tipo == "*":
            a, b = rng.randint(0, 999), rng.randint(0, 999)
            out.append([
                f"¿Cuánto es {a} × {b}?",
                f"{a} × {b} = {a*b}."
            ])

        elif tipo == "/":
            b = rng.randint(1, 1000)
            r = rng.randint(0, 1000)
            a = b * r
            out.append([
                f"¿Cuánto es {a} ÷ {b}?",
                f"{a} ÷ {b} = {r}."
            ])

        else:
            b = rng.randint(2, 100)
            q = rng.randint(0, 100)
            a = b * q + rng.randint(0, b - 1)
            out.append([
                f"¿Cuál es el resto de {a} ÷ {b}?",
                f"El resto es {a % b}."
            ])

    return [c for c in out if quality_conv(c)]

def sinteticos_conocimiento(seed=17):
    rng = random.Random(seed)

    defs = {
        "algoritmo": "Un conjunto ordenado de pasos para resolver un problema.",
        "variable": "Un nombre asociado a un valor que puede cambiar durante la ejecución.",
        "función": "Un bloque reutilizable de código que recibe datos y produce un resultado.",
        "internet": "Una red mundial de redes y dispositivos que intercambian información.",
        "programación": "El proceso de crear instrucciones que puede ejecutar un ordenador.",
        "energía": "La capacidad de un sistema para producir cambios o realizar trabajo.",
        "gravedad": "La interacción por la que los cuerpos con masa se atraen.",
        "molécula": "Una agrupación de átomos unidos mediante enlaces químicos.",
        "ecosistema": "Un conjunto de seres vivos y factores físicos que interactúan.",
        "metadato": "Información que describe o proporciona contexto sobre otros datos.",
        "recursividad": "Una técnica en la que una función se llama a sí misma con un caso base.",
        "api": "Una interfaz que permite que distintos programas se comuniquen mediante reglas definidas.",
    }

    out = []
    keys = list(defs)

    for _ in range(12000):
        k = rng.choice(keys)
        out.append([
            f"¿Qué es {k}?",
            defs[k],
        ])

    return [c for c in out if quality_conv(c)]

# ============================================================
# TOOL CALLING
# ============================================================

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Calcula una expresión matemática y devuelve el resultado.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "web_search",
        "description": "Busca información actual o específica en Internet.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_weather",
        "description": "Consulta el tiempo actual o previsto de una ubicación.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_time",
        "description": "Obtiene la hora de una zona horaria.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string"}
            },
            "required": ["timezone"]
        }
    },
    {
        "name": "python",
        "description": "Ejecuta código Python para cálculos o procesamiento cuando sea necesario.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout": {"type": "integer"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "read_file",
        "description": "Lee el contenido de un archivo permitido.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Escribe contenido en un archivo permitido.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_files",
        "description": "Lista archivos de una carpeta permitida.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "create_task",
        "description": "Crea una tarea o recordatorio cuando el runtime lo permite.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "due": {"type": "string"}
            },
            "required": ["title"]
        }
    },
    {
        "name": "get_url",
        "description": "Obtiene el contenido de una URL pública permitida.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "github",
        "description": "Consulta o modifica GitHub cuando el usuario lo solicita explícitamente.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "read", "create", "update"]
                },
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "query": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["action"]
        }
    },
]

def tools_text():
    return json.dumps(TOOL_SCHEMAS, ensure_ascii=False, separators=(",", ":"))

def tool_call_json(name, arguments):
    return json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )

def _tool_prompt(user, include_tools=True):
    if include_tools:
        return (
            "<tools>\n"
            + tools_text()
            + "\n</tools>\n"
            + user
        )
    return user

def _tool_call(name, arguments):
    return (
        "<tool_call>\n"
        + tool_call_json(name, arguments)
        + "\n</tool_call>"
    )

def _tool_result(value):
    return (
        "<tool_result>\n"
        + str(value)
        + "\n</tool_result>"
    )

def sinteticos_tools(n=TOOL_SYNTH, seed=123):
    rng = random.Random(seed)
    out = []

    single = [
        ("Calcula 17 por 23.", "calculator",
         {"expression": "17*23"},
         "391", "El resultado es 391."),
        ("Calcula la raíz cuadrada de 144.", "calculator",
         {"expression": "144**0.5"},
         "12", "La raíz cuadrada de 144 es 12."),
        ("Busca información sobre Python 3.13.", "web_search",
         {"query": "Python 3.13", "max_results": 5},
         '{"results":[{"title":"Python 3.13"}]}',
         "Python 3.13 es una versión del lenguaje Python."),
        ("¿Qué tiempo hace en Madrid?", "get_weather",
         {"location": "Madrid", "units": "celsius"},
         '{"temperature":18,"condition":"soleado"}',
         "En Madrid hacen 18 °C y está soleado."),
        ("Dime la hora de Europe/Madrid.", "get_time",
         {"timezone": "Europe/Madrid"},
         '{"time":"14:30","timezone":"Europe/Madrid"}',
         "La hora consultada es 14:30 en Europe/Madrid."),
        ("Ejecuta este Python: print(2 + 2)", "python",
         {"code": "print(2 + 2)", "timeout": 10},
         "4",
         "El resultado de ejecutar el código es 4."),
        ("Lee /tmp/datos.txt.", "read_file",
         {"path": "/tmp/datos.txt", "max_chars": 4000},
         "hola mundo",
         "El archivo contiene: hola mundo."),
        ("Lista los archivos de /tmp.", "list_files",
         {"path": "/tmp"},
         '["a.txt","b.py"]',
         "En /tmp hay a.txt y b.py."),
        ("Crea un recordatorio para estudiar.", "create_task",
         {"title": "Estudiar"},
         '{"created":true}',
         "He creado la tarea."),
        ("Consulta https://example.com.", "get_url",
         {"url": "https://example.com"},
         '{"status":200,"title":"Example Domain"}',
         "La página respondió correctamente."),
        ("Busca el repositorio de Aqua en GitHub.", "github",
         {"action":"search","query":"Aqua AI"},
         '{"items":[{"full_name":"aquas-modela/Aqua-1.0.0"}]}',
         "He encontrado resultados relacionados con Aqua en GitHub."),
    ]

    for _ in range(n):
        user, name, args, result, answer = rng.choice(single)
        prompt = _tool_prompt(user)
        call = _tool_call(name, args)
        out.append([prompt, call, _tool_result(result), answer])

    return out

def sinteticos_multi_tools(n=12000, seed=987):
    rng = random.Random(seed)
    out = []

    cases = [
        (
            "Busca la hora de Madrid y dime también qué tiempo hace allí.",
            [
                ("get_time", {"timezone":"Europe/Madrid"},
                 '{"time":"14:30","timezone":"Europe/Madrid"}'),
                ("get_weather", {"location":"Madrid","units":"celsius"},
                 '{"temperature":18,"condition":"soleado"}')
            ],
            "En Madrid son las 14:30 y hacen 18 °C con tiempo soleado."
        ),
        (
            "Busca Python y después calcula 3 por 14.",
            [
                ("web_search", {"query":"Python lenguaje","max_results":3},
                 '{"results":[{"title":"Python"}]}'),
                ("calculator", {"expression":"3*14"}, "42")
            ],
            "Python es un lenguaje de programación y 3 × 14 = 42."
        ),
        (
            "Lee /tmp/datos.txt y calcula cuántas letras tiene su contenido.",
            [
                ("read_file", {"path":"/tmp/datos.txt","max_chars":4000},
                 "Hola Aqua"),
                ("python", {"code":"text='Hola Aqua'; print(len(text))","timeout":10},
                 "9")
            ],
            "El contenido tiene 9 caracteres."
        ),
        (
            "Busca el repositorio de Aqua en GitHub y dime su nombre.",
            [
                ("github", {"action":"search","query":"Aqua 1.0.0"},
                 '{"items":[{"full_name":"aquas-modela/Aqua-1.0.0"}]}')
            ],
            "El resultado incluye el repositorio aquas-modela/Aqua-1.0.0."
        ),
    ]

    for _ in range(n):
        user, calls, answer = rng.choice(cases)
        conv = [_tool_prompt(user)]

        for name, args, result in calls:
            conv.append(_tool_call(name, args))
            conv.append(_tool_result(result))

        conv.append(answer)
        out.append(conv)

    return out

def sinteticos_tools_result(n=12000, seed=321):
    """Ejemplos de tool-call seguido por resultado y respuesta final."""
    rng = random.Random(seed)
    cases = [
        (
            "¿Cuánto es 875 * 24?",
            "calculator",
            {"expression": "875*24"},
            "21000",
            "875 × 24 = 21 000."
        ),
        (
            "Busca información sobre Llama 3.",
            "web_search",
            {"query": "Llama 3", "max_results": 5},
            '{"results":[{"title":"Llama 3"}]}',
            "He encontrado resultados sobre Llama 3."
        ),
        (
            "¿Qué hora es en Madrid?",
            "get_time",
            {"timezone": "Europe/Madrid"},
            '{"time":"19:20","timezone":"Europe/Madrid"}',
            "En Europe/Madrid son las 19:20."
        ),
        (
            "¿Qué tiempo hace en Pamplona?",
            "get_weather",
            {"location": "Pamplona", "units": "celsius"},
            '{"temperature":17,"condition":"nublado"}',
            "En Pamplona hay 17 °C y está nublado."
        ),
        (
            "Ejecuta print(10**2).",
            "python",
            {"code": "print(10**2)", "timeout": 10},
            "100",
            "La ejecución produjo 100."
        ),
        (
            "Lista los archivos de /tmp.",
            "list_files",
            {"path": "/tmp"},
            '["a.txt","config.json"]',
            "En /tmp están a.txt y config.json."
        ),
        (
            "Lee /tmp/config.json.",
            "read_file",
            {"path": "/tmp/config.json", "max_chars": 4000},
            '{"name":"Aqua","version":"1.0.0"}',
            "El archivo contiene la configuración de Aqua 1.0.0."
        ),
        (
            "Consulta example.com.",
            "get_url",
            {"url": "https://example.com"},
            '{"status":200,"title":"Example Domain"}',
            "La página respondió correctamente con estado 200."
        ),
        (
            "Busca Aqua 1.0.0 en GitHub.",
            "github",
            {"action": "search", "query": "Aqua 1.0.0"},
            '{"items":[{"full_name":"aquas-modela/Aqua-1.0.0"}]}',
            "He encontrado el repositorio aquas-modela/Aqua-1.0.0."
        ),
        (
            "Extrae name de este JSON.",
            "json_transform",
            {"operation": "get", "data": {"name": "Aqua", "version": "1.0.0"}, "path": "name"},
            '"Aqua"',
            "El campo name es Aqua."
        ),
        (
            "Resume este texto en 20 palabras: Aqua es un modelo pequeño orientado a conversaciones y herramientas.",
            "summarize_text",
            {"text": "Aqua es un modelo pequeño orientado a conversaciones y herramientas.", "max_words": 20},
            "Modelo pequeño orientado a conversaciones y uso de herramientas.",
            "Resumen: Aqua es un modelo pequeño orientado a conversaciones y herramientas."
        ),
        (
            "Escribe hola en /tmp/a.txt.",
            "write_file",
            {"path": "/tmp/a.txt", "content": "hola"},
            '{"written":true,"bytes":4}',
            "He escrito el contenido en /tmp/a.txt."
        ),
    ]
    out=[]
    for _ in range(n):
        user, name, args, result, answer = rng.choice(cases)
        out.append([
            _tool_prompt(user),
            _tool_call(name, args),
            _tool_result(result),
            answer,
        ])
    return out

def sinteticos_no_tool(n=12000, seed=654):
    rng = random.Random(seed)
    pairs = [
        ("Hola, ¿quién eres?", "Soy Aqua, un asistente de IA."),
        ("¿Qué es una variable?", "Una variable es un nombre asociado a un valor que puede cambiar."),
        ("Explícame brevemente qué es Python.",
         "Python es un lenguaje de programación de propósito general, conocido por su sintaxis clara y su amplio ecosistema."),
        ("¿Puedes calcular 2 + 2 mentalmente?",
         "Sí. 2 + 2 = 4."),
        ("Dame tres ideas para estudiar mejor.",
         "Puedes dividir el estudio en sesiones cortas, practicar con ejercicios y revisar los errores."),
        ("¿Necesito una herramienta para que me saludes?",
         "No. Para saludarte no necesito ninguna herramienta."),
        ("¿Qué herramienta usarías para una suma compleja?",
         "Usaría la herramienta calculator si necesito delegar el cálculo."),
        ("¿Qué haces si una herramienta no está disponible?",
         "No fingiría haberla ejecutado. Explicaría que no está disponible y ofrecería una alternativa."),
    ]

    return [
        list(rng.choice(pairs))
        for _ in range(n)
    ]

# ============================================================
# CONSTRUIR CHAT
# ============================================================

print("\n=== CARGANDO CHAT ===")

share = load_sharegpt() if SHAREGPT else []

chat_specs = [
    ("HuggingFaceH4/ultrachat_200k", None, 80000),
    ("bertin-project/alpaca-spanish", None, 30000),
    ("jondurbin/airoboros-3.2", None, 25000),
    ("Open-Orca/OpenOrca", None, 15000),
    ("teknium/OpenHermes-2.5", None, 15000),
    ("LDJnr/Capybara", None, 10000),
    ("microsoft/orca-math-word-problems-200k", None, 20000),
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, 20000),
]

chat_extra = []

for name, config, cap in chat_specs:
    chat_extra.extend(
        load_chat_source(name, config, cap=cap)
    )

sint_arit = sinteticos_aritmetica(SINTETICOS)
sint_knowledge = sinteticos_conocimiento()
sint_tools = sinteticos_tools()
sint_multi = sinteticos_multi_tools()
sint_no_tool = sinteticos_no_tool()
sint_tool_results = sinteticos_tools_result()

seen = set()

share = dedup(share, seen)
chat_extra = dedup(chat_extra, seen)
sint_arit = dedup(sint_arit, seen)
sint_knowledge = dedup(sint_knowledge, seen)
sint_tools = dedup(sint_tools, seen)
sint_multi = dedup(sint_multi, seen)
sint_no_tool = dedup(sint_no_tool, seen)
sint_tool_results = dedup(sint_tool_results, seen)

# Validación principalmente con conversaciones reales.
nv = min(1000, max(100, len(share) // 10))

val_p = share[:nv]

# Tool calling recibe una fracción importante del SFT:
# single-call + multi-call + tool-result + ejemplos donde NO debe llamar.
train_chat = (
    share[nv:] * SHAREGPT_REP
    + chat_extra
    + sint_arit
    + sint_knowledge
    + sint_no_tool
    + sint_tools * 3
    + sint_multi * 3
    + sint_tool_results * 2
)

random.Random(SEED + 1).shuffle(train_chat)

with open(DATASET_JSONL, "w", encoding="utf-8") as f:
    for c in train_chat:
        f.write(
            json.dumps(
                {"conversation": c},
                ensure_ascii=False,
            )
            + "\n"
        )

print(
    f"CHAT: {len(train_chat):,} train | "
    f"{len(val_p):,} val | "
    f"tool examples={len(sint_tools) + len(sint_tool_results):,}"
)

# ============================================================
# TEXTO WEB
# ============================================================

FUENTES = [
    ("HuggingFaceFW/fineweb-2", "spa_Latn", 0.28),
    ("HuggingFaceFW/fineweb-edu", None, 0.14),
    ("HuggingFaceTB/cosmopedia", None, 0.08),
    ("wikimedia/wikipedia", "20231101.es", 0.12),
    ("uonlp/CulturaX", "es", 0.12),
    ("oscar-corpus/OSCAR-2301", "es", 0.05),
    ("allenai/c4", "es", 0.04),
    ("bigscience-data/roots_es", None, 0.07),
]

FUENTES_OK = []

try:
    from datasets import load_dataset

    for nombre, config, peso in FUENTES:
        try:
            if config:
                ds = load_dataset(
                    nombre,
                    config,
                    split="train",
                    streaming=True,
                )
            else:
                ds = load_dataset(
                    nombre,
                    split="train",
                    streaming=True,
                )

            it = iter(ds)
            next(it)

            FUENTES_OK.append((nombre, config, ds, peso))
            print(f"PRE OK: {nombre}")

        except Exception as e:
            print(
                f"PRE aviso {nombre}: "
                f"{type(e).__name__}"
            )

except Exception as e:
    print(f"datasets no disponible: {e}")

def extraer_texto(ex):
    if not isinstance(ex, dict):
        return ""

    for key in (
        "text",
        "content",
        "raw_content",
        "document",
    ):
        value = ex.get(key)

        if not isinstance(value, str):
            continue

        value = normalize_text(value)

        if len(value) < 200:
            continue

        value = value[:30000]
        sample = value[:1800]

        alpha = sum(
            c.isalpha() or c in " \n\t"
            for c in sample
        )

        if alpha / max(1, len(sample)) < 0.70:
            continue

        if repeated_ngram_ratio(words(value[:5000])) > 0.45:
            continue

        return value

    return ""

def textos_web(seed=0):
    if not FUENTES_OK:
        return

    rng = random.Random(seed)

    streams = []
    for nombre, config, ds, peso in FUENTES_OK:
        try:
            streams.append(
                (
                    nombre,
                    ds.shuffle(
                        seed=seed,
                        buffer_size=2000,
                    ),
                    peso,
                )
            )
        except Exception:
            streams.append((nombre, ds, peso))

    iteradores = [iter(x[1]) for x in streams]
    pesos = [x[2] for x in streams]
    fallos = [0] * len(streams)

    while any(p > 0 for p in pesos):
        indices = [
            i for i, p in enumerate(pesos)
            if p > 0
        ]

        if not indices:
            break

        pesos_activos = [pesos[i] for i in indices]
        i = rng.choices(
            indices,
            weights=pesos_activos,
            k=1,
        )[0]

        try:
            ex = next(iteradores[i])
            fallos[i] = 0

        except StopIteration:
            iteradores[i] = iter(streams[i][1])
            continue

        except Exception:
            fallos[i] += 1

            if fallos[i] >= 12:
                pesos[i] = 0
                print(
                    f"[datos] fuente descartada: "
                    f"{streams[i][0]}"
                )

            time.sleep(0.5)
            continue

        text = extraer_texto(ex)

        if text:
            yield text

# ============================================================
# TOKENIZER
# ============================================================

os.makedirs(TOK_DIR, exist_ok=True)

if not os.path.exists(os.path.join(TOK_DIR, "vocab.json")):
    print("\n=== ENTRENANDO TOKENIZER BPE ===")

    def tokenizer_texts():
        # Chat primero: fuerza buena representación conversacional.
        for c in train_chat[:250000]:
            yield conversation_to_text(c)

        count = 0
        for text in textos_web(SEED):
            yield text[:6000]
            count += 1
            if count >= 100000:
                break

    bpe = ByteLevelBPETokenizer()

    bpe.train_from_iterator(
        tokenizer_texts(),
        vocab_size=VOCAB,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        length=None,
    )

    bpe.save_model(TOK_DIR)

tok = ByteLevelBPETokenizer(
    os.path.join(TOK_DIR, "vocab.json"),
    os.path.join(TOK_DIR, "merges.txt"),
)

token_ids = {
    token: tok.token_to_id(token)
    for token in SPECIAL_TOKENS
}

PAD = token_ids["<pad>"]
BOS = token_ids["<bos>"]
EOS = token_ids["<eos>"]

print(f"Vocab: {tok.get_vocab_size():,}")

def conversation_to_text(c):
    pieces = []

    for i, value in enumerate(c):
        if i % 2 == 0:
            pieces.append(
                f"{ROL_USER}: {value}\n{ROL_BOT}:"
            )
        else:
            pieces.append(f" {value}\n")

    return "\n".join(pieces)

# ============================================================
# CODIFICACIÓN
# ============================================================

def encode_conversation(c):
    ids = []
    labels = []

    for i, value in enumerate(c):
        value = normalize_text(value)

        # Índices pares son normalmente usuario/tool_result.
        # Un tool_result nunca debe aprenderse como "USUARIO".
        is_tool_result = value.startswith("<tool_result>")

        if i % 2 == 0:
            if is_tool_result:
                text = (
                    ("" if i == 0 else "\n")
                    + value
                    + "\n"
                )
                part = tok.encode(text).ids
                ids.extend(part)
                labels.extend([-100] * len(part))
            else:
                text = (
                    ("" if i == 0 else "\n")
                    + f"{ROL_USER}: {value}\n{ROL_BOT}:"
                )
                part = tok.encode(text).ids
                ids.extend(part)
                labels.extend([-100] * len(part))

        else:
            # Las llamadas de herramienta y respuestas finales
            # son salidas del modelo y SÍ reciben loss.
            part = tok.encode(
                " " + value
            ).ids + [EOS]

            ids.extend(part)
            labels.extend(part)

    return ids, labels

def encode_plain(text):
    return tok.encode(text).ids + [EOS]

def prepare(convs, max_len=TRAIN_BLOCK):
    out = []

    for c in convs:
        ids, labels = encode_conversation(c)

        if len(ids) > max_len:
            ids = ids[:max_len]
            labels = labels[:max_len]

        if any(x != -100 for x in labels):
            out.append((ids, labels))

    return out

def prompt_ids(turns):
    if isinstance(turns, str):
        turns = [turns]

    ids = []

    for i, value in enumerate(turns):
        if i % 2 == 0:
            text = (
                ("" if i == 0 else "\n")
                + f"{ROL_USER}: {value}\n{ROL_BOT}:"
            )
            ids.extend(tok.encode(text).ids)
        else:
            ids.extend(tok.encode(" " + value).ids)
            ids.append(EOS)

    return ids

# ============================================================
# MODELO ~100M
# ============================================================

cfg = LlamaConfig(
    vocab_size=tok.get_vocab_size(),

    hidden_size=768,
    intermediate_size=2048,

    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,

    max_position_embeddings=CONTEXT,

    rms_norm_eps=1e-5,
    hidden_act="silu",

    tie_word_embeddings=True,

    attention_dropout=0.0,

    bos_token_id=BOS,
    eos_token_id=EOS,
    pad_token_id=PAD,

    use_cache=False,
)

try:
    model = AutoModelForCausalLM.from_config(
        cfg,
        attn_implementation="sdpa",
    )
except Exception:
    model = AutoModelForCausalLM.from_config(cfg)

with torch.no_grad():
    std = 0.02 / math.sqrt(
        2 * cfg.num_hidden_layers
    )

    for name, param in model.named_parameters():
        if (
            name.endswith("o_proj.weight")
            or name.endswith("down_proj.weight")
        ):
            torch.nn.init.normal_(
                param,
                mean=0.0,
                std=std,
            )

model = model.to(dev)
model.train()

n_par = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"PARAMETROS: {n_par/1e6:.1f}M | "
    f"CONTEXTO: {CONTEXT}"
)

# ============================================================
# OPTIMIZADOR
# ============================================================

decay = []
no_decay = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue

    if p.dim() >= 2 and "embed_tokens" not in name:
        decay.append(p)
    else:
        no_decay.append(p)

groups = [
    {
        "params": decay,
        "weight_decay": 0.10,
    },
    {
        "params": no_decay,
        "weight_decay": 0.0,
    },
]

try:
    opt = torch.optim.AdamW(
        groups,
        lr=LR_PRE_MAX,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=usa_amp,
    )
except Exception:
    opt = torch.optim.AdamW(
        groups,
        lr=LR_PRE_MAX,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(
        usa_amp
        and AMP_DTYPE == torch.float16
    ),
)

estado = {
    "paso": 0,
    "fase": 1,
    "epoca_sft": 0,
    "mejor": float("inf"),
    "lr": LR_PRE_MAX,
}

# ============================================================
# AUTO MICRO-BATCH
# ============================================================

MB = 1

if usa_amp:
    for candidate in (8, 4, 2, 1):
        x = None

        try:
            x = torch.randint(
                0,
                tok.get_vocab_size(),
                (candidate, TRAIN_BLOCK),
                device=dev,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=AMP_DTYPE,
            ):
                loss = model(
                    input_ids=x,
                    labels=x,
                ).loss

            scaler.scale(loss).backward()

            MB = candidate

            model.zero_grad(
                set_to_none=True
            )

            del x
            gc.collect()
            torch.cuda.empty_cache()

            print(
                f"micro-batch detectado: {MB}"
            )

            break

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

            print(
                f"OOM con micro-batch {candidate}"
            )

            model.zero_grad(
                set_to_none=True
            )

            del x
            gc.collect()
            torch.cuda.empty_cache()

    else:
        raise RuntimeError(
            "OOM incluso con micro-batch 1"
        )

ACUM = max(
    1,
    TOK_PASO // (MB * TRAIN_BLOCK),
)

print(
    f"tokens/paso ≈ {MB * ACUM * TRAIN_BLOCK:,}"
)

# ============================================================
# COMPILE
# ============================================================

fwd = model

if (
    USAR_COMPILE
    and usa_amp
    and hasattr(torch, "compile")
):
    try:
        import torch._dynamo as dynamo

        dynamo.config.suppress_errors = True

        compiled = torch.compile(
            model,
            mode="max-autotune",
            dynamic=False,
        )

        test_x = torch.randint(
            0,
            tok.get_vocab_size(),
            (MB, TRAIN_BLOCK),
            device=dev,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
        ):
            test_loss = compiled(
                input_ids=test_x,
                labels=test_x,
            ).loss

        test_loss.backward()

        model.zero_grad(
            set_to_none=True
        )

        del test_x, test_loss
        gc.collect()
        torch.cuda.empty_cache()

        fwd = compiled
        print("torch.compile=ON")

    except Exception as e:
        print(
            f"torch.compile=OFF: "
            f"{type(e).__name__}: {e}"
        )
        model.zero_grad(
            set_to_none=True
        )
        fwd = model

# ============================================================
# CHECKPOINT
# ============================================================

if os.path.exists(CKPT):
    try:
        c = torch.load(
            CKPT,
            map_location="cpu",
            weights_only=False,
        )

        if c.get("arch") == ARCH:
            model.load_state_dict(
                c["model"]
            )
            opt.load_state_dict(
                c["opt"]
            )
            scaler.load_state_dict(
                c.get("scaler", {})
            )
            estado.update(
                c.get("estado", {})
            )

            print(
                f"Checkpoint restaurado: "
                f"paso={estado['paso']} "
                f"fase={estado['fase']}"
            )

        del c

    except Exception as e:
        print(
            f"Checkpoint no cargado: {e}"
        )

def guardar_ckpt():
    tmp = CKPT + ".tmp"

    torch.save(
        {
            "arch": ARCH,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "estado": estado,
        },
        tmp,
    )

    os.replace(tmp, CKPT)

# ============================================================
# ENTRENAMIENTO
# ============================================================

def train_micro(ids, labels=None, divisor=1):
    ids = ids.to(
        dev,
        non_blocking=True,
    )

    if labels is None:
        labels = ids
    else:
        labels = labels.to(
            dev,
            non_blocking=True,
        )

    with torch.autocast(
        device_type=dev,
        dtype=AMP_DTYPE,
        enabled=usa_amp,
    ):
        loss = fwd(
            input_ids=ids,
            labels=labels,
        ).loss

    scaler.scale(
        loss / divisor
    ).backward()

    return loss.detach()

def apply_optimizer(lr):
    for g in opt.param_groups:
        g["lr"] = lr

    scaler.unscale_(opt)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    scaler.step(opt)
    scaler.update()

    opt.zero_grad(
        set_to_none=True
    )

    estado["paso"] += 1
    estado["lr"] = lr

# ============================================================
# DATASET STREAMING
# ============================================================

def chat_ids_stream(seed=0):
    rng = random.Random(seed)

    indices = list(
        range(len(train_chat))
    )

    while True:
        rng.shuffle(indices)

        for start in range(
            0,
            len(indices),
            512,
        ):
            batch = [
                train_chat[i]
                for i in indices[
                    start:start + 512
                ]
            ]

            for c in batch:
                ids, _ = encode_conversation(c)

                if len(ids) < 8:
                    continue

                yield ids

def web_ids_stream(seed=0):
    for text in textos_web(seed):
        ids = encode_plain(text)

        if len(ids) >= 8:
            yield ids

def block_stream(generation):
    buffer = []

    for ids in generation:
        buffer.extend(ids)

        while len(buffer) >= TRAIN_BLOCK:
            yield buffer[:TRAIN_BLOCK]
            buffer = buffer[TRAIN_BLOCK:]

def chat_web_mix(seed=0):
    rng = random.Random(seed)

    chat = block_stream(
        chat_ids_stream(seed + 10)
    )

    web = block_stream(
        web_ids_stream(seed + 20)
    )

    web_alive = bool(FUENTES_OK)

    while True:
        progress = min(
            1.0,
            estado["paso"] / 10000,
        )

        chat_prob = (
            CHAT_MIX0
            + (CHAT_MIX1 - CHAT_MIX0)
            * progress
        )

        use_chat = (
            not web_alive
            or rng.random() < chat_prob
        )

        try:
            block = next(
                chat if use_chat else web
            )

        except StopIteration:
            if use_chat:
                return

            web_alive = False
            continue

        yield torch.tensor(
            [block],
            dtype=torch.long,
        )

# ============================================================
# EVALUACIÓN
# ============================================================

@torch.no_grad()
def evaluar(n=500):
    if not val_p:
        return float("nan")

    model.eval()

    subset = val_p[:n]
    total_loss = 0.0
    total_tokens = 0

    for start in range(
        0,
        len(subset),
        8,
    ):
        items = prepare(
            subset[start:start + 8]
        )

        if not items:
            continue

        max_len = max(
            len(x[0])
            for x in items
        )

        ids = torch.full(
            (len(items), max_len),
            PAD,
            dtype=torch.long,
        )

        labels = torch.full(
            (len(items), max_len),
            -100,
            dtype=torch.long,
        )

        for i, (x, y) in enumerate(items):
            ids[i, :len(x)] = torch.tensor(x)
            labels[i, :len(y)] = torch.tensor(y)

        ids = ids.to(dev)
        labels = labels.to(dev)

        with torch.autocast(
            device_type=dev,
            dtype=AMP_DTYPE,
            enabled=usa_amp,
        ):
            logits = model(
                input_ids=ids
            ).logits

        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(
                -1,
                logits.size(-1),
            ),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        valid = (
            labels[:, 1:] != -100
        ).sum().item()

        total_loss += loss.item()
        total_tokens += valid

    model.train()

    return total_loss / max(
        1,
        total_tokens,
    )

# ============================================================
# GENERACIÓN + TOOL CALLING
# ============================================================

def validate_tool_call(call):
    """Valida una llamada sin ejecutar ninguna herramienta."""
    if not isinstance(call, dict):
        return False
    name = call.get("name")
    args = call.get("arguments")
    if not isinstance(name, str) or not isinstance(args, dict):
        return False
    schema = next(
        (x for x in TOOL_SCHEMAS if x["name"] == name),
        None,
    )
    if schema is None:
        return False
    required = schema.get("parameters", {}).get("required", [])
    return all(key in args for key in required)

def parse_tool_call(text):
    match = re.search(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        text,
        flags=re.S,
    )

    if not match:
        return None

    try:
        obj = json.loads(
            match.group(1)
        )

        if not isinstance(obj, dict):
            return None

        name = obj.get("name")
        arguments = obj.get(
            "arguments",
            {},
        )

        if not isinstance(name, str):
            return None

        if not isinstance(arguments, dict):
            return None

        valid_tools = {x["name"] for x in TOOL_SCHEMAS}
        if name not in valid_tools:
            return None

        return {
            "name": name,
            "arguments": arguments,
        }

    except Exception:
        return None

@torch.no_grad()
def generar(turnos, max_new=180):
    if isinstance(turnos, str):
        turnos = [turnos]

    ids = prompt_ids(turnos)

    # Nunca permitir que el prompt + generación
    # supere el contexto configurado.
    max_input = max(
        32,
        CONTEXT - max_new,
    )

    ids = ids[-max_input:]

    model.eval()

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=dev,
    )

    with torch.autocast(
        device_type=dev,
        dtype=AMP_DTYPE,
        enabled=usa_amp,
    ):
        out = model.generate(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            max_new_tokens=max_new,
            do_sample=True,
            **GEN,
            eos_token_id=EOS,
            pad_token_id=PAD,
            use_cache=True,
        )

    model.train()

    generated = out[
        0,
        x.shape[1]:,
    ].tolist()

    text = tok.decode(
        generated,
        skip_special_tokens=False,
    ).strip()

    # Si genera una llamada, se devuelve intacta
    # para que un runtime externo pueda ejecutarla.
    call = parse_tool_call(text)

    if call:
        return text, call

    return (
        text.split(
            f"\n{ROL_USER}:"
        )[0].strip(),
        None,
    )

def execute_tool_local(call):
    """Runtime mínimo opcional para probar el protocolo.

    No ejecuta herramientas peligrosas automáticamente.
    Solo demuestra cómo conectar Aqua con un executor externo.
    """
    if not call:
        return None

    name = call["name"]
    args = call["arguments"]

    if name == "calculator":
        return {
            "ok": False,
            "tool": name,
            "error": "Conecta aquí un evaluador matemático seguro."
        }

    if name == "web_search":
        return {
            "ok": False,
            "tool": name,
            "error": "Conecta aquí tu buscador."
        }

    return {
        "ok": False,
        "tool": name,
        "error": "Tool no conectada al runtime."
    }

# ============================================================
# PREFETCH
# ============================================================

class Prefetch:
    def __init__(self, fn, size=16):
        self.q = queue.Queue(
            maxsize=size
        )
        self.stop = False
        self.error = None

        import threading

        threading.Thread(
            target=self._run,
            args=(fn,),
            daemon=True,
        ).start()

    def _put(self, value):
        while not self.stop:
            try:
                self.q.put(
                    value,
                    timeout=1,
                )
                return True
            except queue.Full:
                continue

        return False

    def _run(self, fn):
        try:
            for value in fn():
                if not self._put(value):
                    return

        except Exception as e:
            self.error = e

        self._put(None)

    def __iter__(self):
        while True:
            value = self.q.get(
                timeout=1800
            )

            if value is None:
                if self.error:
                    raise self.error
                return

            yield value

    def close(self):
        self.stop = True

# ============================================================
# SEÑALES
# ============================================================

parar = False

def stop_handler(*_):
    global parar
    parar = True

for sig in (
    signal.SIGTERM,
    signal.SIGINT,
):
    try:
        signal.signal(
            sig,
            stop_handler,
        )
    except Exception:
        pass

# ============================================================
# FASE 1 — PRETRAINING
# ============================================================

LIMITE = LIMITE_H * 3600
fin_pre = (
    T0
    + LIMITE
    - HORAS_SFT * 3600
)

if estado["fase"] == 1 and not parar:
    print("\n=== FASE 1: PRETRAINING ===")

    pre_start = time.time()

    lr_start = (
        LR_PRE_MAX
        if estado["paso"] < WARMUP
        else max(
            LR_PRE_MIN,
            min(
                LR_PRE_MAX,
                estado["lr"],
            ),
        )
    )

    stream = Prefetch(
        lambda: chat_web_mix(
            SEED + estado["paso"]
        ),
        size=12,
    )

    loss_acc = 0.0
    micro_count = 0
    accum_count = 0

    last_log = time.time()
    last_ckpt = time.time()

    try:
        for batch in stream:
            now = time.time()

            if now >= fin_pre or parar:
                break

            loss = train_micro(
                batch,
                None,
                ACUM,
            )

            loss_acc += loss.item()
            micro_count += 1
            accum_count += 1

            if accum_count >= ACUM:
                elapsed = max(
                    1.0,
                    now - pre_start,
                )

                progress = min(
                    1.0,
                    elapsed
                    / max(
                        1.0,
                        fin_pre - pre_start,
                    ),
                )

                if estado["paso"] < WARMUP:
                    lr = (
                        LR_PRE_MAX
                        * (estado["paso"] + 1)
                        / WARMUP
                    )
                else:
                    lr = (
                        LR_PRE_MIN
                        + 0.5
                        * (
                            lr_start
                            - LR_PRE_MIN
                        )
                        * (
                            1
                            + math.cos(
                                math.pi
                                * progress
                            )
                        )
                    )

                apply_optimizer(lr)
                accum_count = 0

            if now - last_log >= 180:
                elapsed_log = max(
                    1.0,
                    now - last_log,
                )

                tokens_sec = (
                    micro_count
                    * MB
                    * TRAIN_BLOCK
                    / elapsed_log
                )

                print(
                    f"[{(now-T0)/3600:.2f}h] "
                    f"paso={estado['paso']} | "
                    f"loss={loss_acc/max(1,micro_count):.4f} | "
                    f"lr={estado['lr']:.2e} | "
                    f"{tokens_sec/1000:.1f}K tok/s"
                )

                loss_acc = 0.0
                micro_count = 0
                last_log = now

            if now - last_ckpt >= CKPT_MIN * 60:
                guardar_ckpt()
                last_ckpt = now

    except Exception:
        traceback.print_exc()
        parar = True

    finally:
        stream.close()

    opt.zero_grad(
        set_to_none=True
    )

    if not parar:
        try:
            val = evaluar()
            print(
                f"val tras pretraining: {val:.4f}"
            )
        except Exception as e:
            print(
                f"evaluación omitida: {e}"
            )

        estado["fase"] = 2

    guardar_ckpt()

# ============================================================
# FASE 2 — SFT
# ============================================================

def sft_batches(epoch):
    rng = random.Random(
        SEED + 100 + epoch
    )

    indices = list(
        range(len(train_chat))
    )

    rng.shuffle(indices)

    for start in range(
        0,
        len(indices),
        2048,
    ):
        batch_indices = indices[
            start:start + 2048
        ]

        encoded = prepare(
            [
                train_chat[i]
                for i in batch_indices
            ],
            max_len=TRAIN_BLOCK,
        )

        if not encoded:
            continue

        encoded.sort(
            key=lambda x: len(x[0])
        )

        current = []
        current_tokens = 0

        for item in encoded:
            length = len(item[0])

            if (
                current
                and (
                    current_tokens + length
                    > TOK_PASO
                    or len(current)
                    >= SFT_MAXB
                )
            ):
                yield make_batch(current)
                current = []
                current_tokens = 0

            current.append(item)
            current_tokens += length

        if current:
            yield make_batch(current)

def make_batch(items):
    max_len = max(
        len(x[0])
        for x in items
    )

    max_len = min(
        TRAIN_BLOCK,
        ((max_len + 7) // 8) * 8,
    )

    ids = torch.full(
        (len(items), max_len),
        PAD,
        dtype=torch.long,
    )

    labels = torch.full(
        (len(items), max_len),
        -100,
        dtype=torch.long,
    )

    for i, (x, y) in enumerate(items):
        length = min(
            len(x),
            max_len,
        )

        ids[i, :length] = torch.tensor(
            x[:length]
        )

        labels[i, :length] = torch.tensor(
            y[:length]
        )

    return ids, labels

def run_sft():
    global parar

    end_time = T0 + LIMITE
    best = estado.get(
        "mejor",
        float("inf"),
    )

    print("\n=== FASE 2: SFT CHAT + TOOLS ===")

    for epoch in range(
        estado["epoca_sft"],
        SFT_EPOCAS_MAX,
    ):
        if parar or time.time() >= end_time:
            break

        stream = Prefetch(
            lambda e=epoch: sft_batches(e),
            size=8,
        )

        local_loss = 0.0
        local_n = 0
        accum = 0
        steps = 0

        complete = True
        last_ckpt = time.time()

        try:
            for ids, labels in stream:
                now = time.time()

                if now >= end_time or parar:
                    complete = False
                    break

                loss = train_micro(
                    ids,
                    labels,
                    ACUM,
                )

                local_loss += loss.item()
                local_n += 1
                accum += 1

                if accum < ACUM:
                    continue

                progress = min(
                    1.0,
                    (
                        steps
                        / max(
                            1,
                            len(train_chat)
                            // max(
                                1,
                                SFT_MAXB,
                            ),
                        )
                    ),
                )

                lr = (
                    LR_SFT_MIN
                    + 0.5
                    * (
                        LR_SFT_MAX
                        - LR_SFT_MIN
                    )
                    * (
                        1
                        + math.cos(
                            math.pi
                            * progress
                        )
                    )
                )

                if steps < SFT_WARM:
                    lr *= (
                        (steps + 1)
                        / SFT_WARM
                    )

                apply_optimizer(lr)

                accum = 0
                steps += 1

                if steps % 100 == 0:
                    print(
                        f"SFT ep={epoch+1} "
                        f"paso={steps} "
                        f"loss={local_loss/max(1,local_n):.4f} "
                        f"lr={lr:.2e}"
                    )

                    local_loss = 0.0
                    local_n = 0

                if (
                    steps % EVAL_CADA == 0
                ):
                    val = evaluar()
                    print(
                        f"  val={val:.4f}"
                    )

                    if val < best:
                        best = val
                        estado["mejor"] = best
                        guardar_ckpt()

                if (
                    now - last_ckpt
                    >= CKPT_MIN * 60
                ):
                    guardar_ckpt()
                    last_ckpt = now

        except Exception:
            traceback.print_exc()
            parar = True
            complete = False

        finally:
            stream.close()

        if complete:
            estado["epoca_sft"] = (
                epoch + 1
            )

            val = evaluar()

            print(
                f"SFT epoch={epoch+1} "
                f"completa | val={val:.4f}"
            )

            if val < best:
                best = val

            estado["mejor"] = best
            guardar_ckpt()

        else:
            break

if (
    not parar
    and estado["fase"] == 2
    and time.time() < T0 + LIMITE
):
    run_sft()

guardar_ckpt()

# ============================================================
# EXPORTAR MODELO
# ============================================================

print("\n=== EXPORTANDO MODELO ===")

if os.path.exists(FINAL):
    shutil.rmtree(FINAL)

os.makedirs(FINAL)

model.to(torch.float32)

model.config.max_position_embeddings = CONTEXT
model.config.use_cache = True

for key, value in dict(
    eos_token_id=EOS,
    pad_token_id=PAD,
    bos_token_id=BOS,
    do_sample=True,
    max_new_tokens=256,
    **GEN,
).items():
    setattr(
        model.generation_config,
        key,
        value,
    )

model.save_pretrained(
    FINAL,
    safe_serialization=True,
)

# Tokenizer compatible con Transformers.
try:
    from transformers import (
        PreTrainedTokenizerFast
    )

    kwargs = dict(
        tokenizer_object=tok._tokenizer,
        eos_token="<eos>",
        pad_token="<pad>",
        model_input_names=[
            "input_ids",
            "attention_mask",
        ],
    )

    tokenizer = PreTrainedTokenizerFast(
        **kwargs
    )

    tokenizer.add_special_tokens(
        {
            "bos_token": "<bos>",
            "additional_special_tokens": [
                "<tool>",
                "</tool>",
                "<tool_call>",
                "</tool_call>",
                "<tool_result>",
                "</tool_result>",
                "<tools>",
                "</tools>",
            ],
        }
    )

    tokenizer.save_pretrained(
        FINAL
    )

except Exception as e:
    print(
        f"tokenizer HF aviso: {e}"
    )

for filename in (
    "vocab.json",
    "merges.txt",
):
    src = os.path.join(
        TOK_DIR,
        filename,
    )

    if os.path.exists(src):
        shutil.copy2(
            src,
            os.path.join(
                FINAL,
                filename,
            ),
        )

# Guardar definición de tools.
with open(
    os.path.join(
        FINAL,
        "tools.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        TOOL_SCHEMAS,
        f,
        ensure_ascii=False,
        indent=2,
    )

# Configuración explícita del modelo.
with open(
    os.path.join(
        FINAL,
        "aqua_config.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "name": "Aqua 1.0.0",
            "architecture": ARCH,
            "parameters": n_par,
            "context_length": CONTEXT,
            "train_block": TRAIN_BLOCK,
            "tool_calling": True,
            "tool_calling_mode": "single_or_sequential",
            "supports_multi_tool": True,
            "supports_tool_results": True,
            "supports_no_tool_decision": True,
            "tool_count": len(TOOL_SCHEMAS),
            "tool_format": (
                "<tool_call>{\"name\":"
                "\"...\",\"arguments\":{...}}"
                "</tool_call>"
            ),
            "tools": [
                x["name"]
                for x in TOOL_SCHEMAS
            ],
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

# ============================================================
# ZIP
# ============================================================

print("\n=== EMPAQUETANDO ===")

try:
    with zipfile.ZipFile(
        ZIP,
        "w",
        zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as z:

        for root, _, files in os.walk(FINAL):
            for filename in files:
                path = os.path.join(
                    root,
                    filename,
                )

                z.write(
                    path,
                    os.path.relpath(
                        path,
                        SALIDA,
                    ),
                )

        for extra in (
            "ckpt.pt",
            "aqua_dataset.jsonl",
        ):
            path = os.path.join(
                SALIDA,
                extra,
            )

            if os.path.exists(path):
                z.write(
                    path,
                    extra,
                )

    print(
        f"ZIP: {ZIP} | "
        f"{os.path.getsize(ZIP)/1e6:.1f} MB"
    )

except Exception as e:
    print(
        f"ZIP fallo: {type(e).__name__}: {e}"
    )

# ============================================================
# PRUEBAS
# ============================================================

print("\n=== PRUEBAS ===")

tests = [
    "Hola, ¿quién eres?",
    "¿Cuánto es 25 * 37?",
    "Explícame qué es Python.",
    "Busca información sobre inteligencia artificial.",
    "Busca la hora de Madrid y después dime qué tiempo hace allí.",
    "Lee /tmp/datos.txt y dime qué contiene.",
]

for query in tests:
    print(f"\n{ROL_USER}: {query}")

    try:
        text, tool = generar(
            query,
            max_new=160,
        )

        print(
            f"{ROL_BOT}: {text}"
        )

        if tool:
            print(
                "TOOL CALL DETECTADO:",
                json.dumps(
                    tool,
                    ensure_ascii=False,
                ),
            )

    except Exception as e:
        print(
            f"Prueba falló: {type(e).__name__}: {e}"
        )

# ============================================================
# RESUMEN
# ============================================================

elapsed = (
    time.time() - T0
) / 3600

print(
    f"\nTERMINADO: {elapsed:.2f}h | "
    f"pasos={estado['paso']} | "
    f"parámetros={n_par/1e6:.1f}M | "
    f"contexto={CONTEXT}"
)

print(
    "\nModelo:",
    FINAL,
)

print(
    "ZIP:",
    ZIP,
)

print(
    "\nTool calling: ACTIVO"
)
