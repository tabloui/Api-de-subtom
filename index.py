from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import aiohttp


# ============================================================
# CONFIG
# ============================================================

def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v.strip().strip('"').strip("'")


class Config:
    def __init__(self):
        self.ai_slots = []
        default_model = _env("AI_MODEL", "gemini-flash-lite-latest")
        n = 1
        empty = 0
        while n <= 20:
            key = _env(f"AI_API_KEY_{n}")
            if not key:
                empty += 1
                if empty >= 3:
                    break
                n += 1
                continue
            empty = 0
            self.ai_slots.append({
                "index": n, "key": key,
                "model": _env(f"AI_MODEL_{n}", default_model) or default_model,
                "base_url": _env(f"AI_API_BASE_URL_{n}", ""),
            })
            n += 1

        main = _env("AI_API_KEY")
        if main:
            self.ai_slots.insert(0, {
                "index": 0, "key": main,
                "model": default_model,
                "base_url": _env("AI_API_BASE_URL", ""),
            })

        self.temperature = float(_env("AI_TEMPERATURE", "0.7"))
        self.max_tokens = int(_env("AI_MAX_TOKENS", "4096"))
        self.timeout = int(_env("AI_TIMEOUT_SECONDS", "55"))
        self.github_token = _env("GITHUB_TOKEN")
        self.vercel_token = _env("VERCEL_TOKEN")
        self.phone_url = _env("PHONE_SERVER_URL").rstrip("/")
        self.phone_secret = _env("PHONE_SERVER_SECRET")
        self.workspace = Path(_env("SUBTOM_WORKSPACE", "/tmp/subtom"))
        self.workspace.mkdir(parents=True, exist_ok=True)


config = Config()


def detect_provider(key: str) -> tuple[str, str]:
    key = (key or "").strip()
    if key.startswith("sk-bl-"):
        return "openai", "https://api.bazaarlink.ai/v1"
    if key.startswith("AQ.") or key.startswith("AIza"):
        return "gemini", "https://generativelanguage.googleapis.com/v1beta"
    if key.startswith("gsk_"):
        return "groq", "https://api.groq.com/openai/v1"
    if key.startswith("sk-ant-"):
        return "anthropic", "https://api.anthropic.com/v1"
    if key.startswith("sk-or-v1-"):
        return "openrouter", "https://openrouter.ai/api/v1"
    if key.startswith("csk-"):
        return "cerebras", "https://api.cerebras.ai/v1"
    if key.startswith("nvapi-"):
        return "nvidia", "https://integrate.api.nvidia.com/v1"
    if key.startswith("xai-"):
        return "xai", "https://api.x.ai/v1"
    if key.startswith("sk-proj-") or key.startswith("sk-"):
        return "openai", "https://api.openai.com/v1"
    return "openrouter", "https://openrouter.ai/api/v1"


SYSTEM_PROMPT = (
    "Eres Subtom IA, el asistente personal de Amin. Hablas siempre en español y eres "
    "súper amable, cálido y cercano, como un buen amigo que sabe programar. Te gusta "
    "conversar: das contexto, explicas con detalle, y tus respuestas son largas y "
    "completas, nunca de una línea seca. Usas un tono natural, con humor seco cuando "
    "encaja, sin exagerar con emojis. Eres técnico cuando hace falta pero sin ser "
    "pedante.\n\n"
    "FORMATO DE RESPUESTA: Separa tus ideas en párrafos cortos con líneas en blanco "
    "entre ellos. Usa listas con guiones cuando enumeres cosas. Pon el código en "
    "bloques con ```. No metas todo en un solo bloque de texto.\n\n"
    "CONTROL DEL TELÉFONO (MUY IMPORTANTE):\n"
    "Tienes acceso al teléfono de Amin a través de herramientas que empiezan por 'phone_'. "
    "Puedes hacer MUCHAS cosas reales en su móvil:\n"
    "- phone_status: ver batería, RAM, disco, modelo, Android, temperatura\n"
    "- phone_battery: solo la batería\n"
    "- phone_location: ubicación GPS\n"
    "- phone_wifi: info de la WiFi\n"
    "- phone_sms: leer los últimos SMS\n"
    "- phone_vibrate: hacer vibrar el teléfono\n"
    "- phone_notify: enviar una notificación\n"
    "- phone_torch: encender/apagar la linterna\n"
    "- phone_brightness: ajustar el brillo\n"
    "- phone_photo: sacar una foto (0=trasera, 1=frontal)\n"
    "- phone_volume: ajustar el volumen\n"
    "- phone_clipboard_get / phone_clipboard_set: leer/escribir el portapapeles\n"
    "- phone_apps: listar apps instaladas\n"
    "- phone_processes: ver procesos activos\n"
    "- phone_shell: ejecutar CUALQUIER comando shell en el teléfono\n\n"
    "ÚSALAS cuando el usuario pida algo del teléfono. Ejemplos:\n"
    "- 'cuánta batería tengo' → phone_battery\n"
    "- 'vibra 2 segundos' → phone_vibrate con ms=2000\n"
    "- 'enciende la linterna' → phone_torch con state='on'\n"
    "- 'qué apps tengo' → phone_apps\n"
    "- 'notifícame: beber agua' → phone_notify\n"
    "- 'sácame una foto' → phone_photo con camera=0\n"
    "- 'dónde estoy' → phone_location\n\n"
    "GITHUB: Tienes herramientas para listar repos, leer/escribir archivos, crear repos, "
    "issues, buscar código, ver commits y árboles de archivos. Úsalas cuando el usuario "
    "mencione GitHub.\n\n"
    "VERCEL: Puedes listar proyectos y deployments de Vercel.\n\n"
    "ARCHIVOS: Puedes leer, escribir, listar, borrar y ver el árbol del workspace.\n\n"
    "WEB: Puedes buscar en internet y descargar URLs.\n\n"
    "IMÁGENES: Solo debes generar imágenes con generate_image cuando el usuario lo pida "
    "EXPLÍCITAMENTE ('genera una imagen', 'hazme un dibujo', 'créame un logo', 'dibuja', "
    "'ilustra'). NUNCA generes imágenes por iniciativa propia ni en conversación normal.\n\n"
    "REGLA CRÍTICA DE PROYECTOS NUEVOS:\n"
    "Cuando el usuario pida 'haz una web', 'crea una app', 'hazme un proyecto', 'un bot', "
    "'una landing' o algo NUEVO, ANTES de tocar nada PREGUNTA en qué repositorio lo quiere. "
    "NO uses repos de conversaciones anteriores sin que él lo pida. Cuando te dé el nombre, "
    "crea el repo y sube los archivos.\n\n"
    "Tienes herramientas reales. Úsalas cuando toca. Nunca inventes contenido: si una "
    "herramienta falla, dilo claramente.\n\n"
    "Eres Subtom, no finjas ser ChatGPT, Claude ni Gemini."
)


# ============================================================
# CONNECTOR
# ============================================================

class Connector:
    def __init__(self):
        self._session = None
        self._cooldowns = {}
        self._fails = {}

    async def _session_get(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.timeout)
            )
        return self._session

    def _slots(self):
        now = time.monotonic()
        avail = [s for s in config.ai_slots if now >= self._cooldowns.get(s["index"], 0)]
        if not avail:
            self._cooldowns.clear()
            avail = list(config.ai_slots)
        avail.sort(key=lambda s: self._fails.get(s["index"], 0))
        return avail

    def _fail(self, slot, cd=60):
        self._cooldowns[slot["index"]] = time.monotonic() + cd
        self._fails[slot["index"]] = self._fails.get(slot["index"], 0) + 1

    def _ok(self, slot):
        self._fails[slot["index"]] = 0
        self._cooldowns.pop(slot["index"], None)

    @staticmethod
    def _tools_to_gemini(tools):
        if not tools:
            return None
        decls = []
        for t in tools:
            fn = t.get("function", {})
            decls.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return [{"functionDeclarations": decls}]

    @staticmethod
    def _messages_to_gemini(messages):
        system_parts = []
        contents = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append({"text": content})
                continue
            if role == "tool":
                name = m.get("name", "tool")
                try:
                    payload = json.loads(content) if isinstance(content, str) else content
                except Exception:
                    payload = {"result": str(content)}
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": name,
                    "response": payload if isinstance(payload, dict) else {"result": payload},
                }}]})
                continue
            if role == "assistant" and m.get("tool_calls"):
                parts = []
                if content:
                    parts.append({"text": content})
                for idx, call in enumerate(m["tool_calls"]):
                    fn = call.get("function", {})
                    try:
                        args_dict = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args_dict = {}
                    part = {"functionCall": {"name": fn.get("name", ""), "args": args_dict}}
                    if idx == 0:
                        part["thoughtSignature"] = "skip_thought_signature_validator"
                    parts.append(part)
                contents.append({"role": "model", "parts": parts})
                continue
            g_role = "model" if role == "assistant" else "user"
            if isinstance(content, list):
                parts = []
                for b in content:
                    if b.get("type") == "text":
                        parts.append({"text": b["text"]})
                contents.append({"role": g_role, "parts": parts})
            else:
                contents.append({"role": g_role, "parts": [{"text": content or ""}]})
        return {"parts": system_parts}, contents

    @staticmethod
    def _gemini_to_openai(data, model):
        text = ""
        tool_calls = []
        finish = "stop"
        for c in data.get("candidates", []):
            finish = str(c.get("finishReason", "stop")).lower()
            for p in c.get("content", {}).get("parts", []):
                if "text" in p:
                    text += p["text"]
                if "functionCall" in p:
                    fc = p["functionCall"]
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                        },
                    })
        if not text and not tool_calls:
            if finish in ("safety", "recitation", "blocked", "prohibited_content"):
                text = f"⚠️ Gemini bloqueó la respuesta por filtros de seguridad ({finish})."
            elif finish == "max_tokens":
                text = "⚠️ Respuesta cortada por límite de tokens."
            else:
                text = f"⚠️ Respuesta vacía del modelo ({finish})."
        msg = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "model": model,
            "choices": [{
                "index": 0, "message": msg,
                "finish_reason": "tool_calls" if tool_calls else finish,
            }],
        }

    async def _call_gemini(self, slot, messages, tools, session):
        base = (slot["base_url"] or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        if base.endswith("/openai"):
            base = base[:-7]
        model = slot["model"] or "gemini-flash-lite-latest"
        endpoint = f"{base}/models/{model}:generateContent?key={slot['key']}"

        sys_instr, contents = self._messages_to_gemini(messages)
        payload = {"contents": contents}
        if sys_instr["parts"]:
            payload["systemInstruction"] = sys_instr
        payload["generationConfig"] = {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
        }
        if tools:
            payload["tools"] = self._tools_to_gemini(tools)

        async with session.post(endpoint, json=payload) as r:
            body = await r.text()
            if r.status == 200:
                self._ok(slot)
                return self._gemini_to_openai(json.loads(body), model)
            if r.status == 429:
                self._fail(slot, 60); raise RuntimeError("429 quota")
            if r.status in (401, 403):
                self._fail(slot, 300); raise RuntimeError(f"{r.status} auth")
            if r.status == 400:
                self._fail(slot, 30); raise RuntimeError(f"400 payload: {body[:200]}")
            if r.status == 404:
                self._fail(slot, 120); raise RuntimeError("404 modelo")
            self._fail(slot, 60)
            raise RuntimeError(f"gemini {r.status}: {body[:200]}")

    async def _call_openai(self, slot, messages, tools, session):
        provider, base = detect_provider(slot["key"])
        if slot["base_url"]:
            base = slot["base_url"]
        endpoint = f"{base.rstrip('/')}/chat/completions"
        payload = {
            "model": slot["model"], "messages": messages,
            "temperature": config.temperature, "max_tokens": config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        async with session.post(endpoint, headers={
            "Authorization": f"Bearer {slot['key']}",
            "Content-Type": "application/json",
        }, json=payload) as r:
            body = await r.text()
            if r.status == 200:
                self._ok(slot)
                return json.loads(body)
            if r.status == 429:
                self._fail(slot, 30); raise RuntimeError("429")
            if r.status in (401, 403):
                self._fail(slot, 300); raise RuntimeError(f"{r.status} auth")
            if r.status == 404:
                self._fail(slot, 120); raise RuntimeError("404 modelo")
            self._fail(slot, 60)
            raise RuntimeError(f"{provider} {r.status}: {body[:200]}")

    async def complete(self, messages, tools=None):
        if not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

        slots = self._slots()
        if not slots:
            raise RuntimeError("No hay slots de IA configurados")

        session = await self._session_get()
        last_err = None
        for slot in slots:
            provider, _ = detect_provider(slot["key"])
            try:
                if provider == "gemini":
                    return await self._call_gemini(slot, messages, tools, session)
                return await self._call_openai(slot, messages, tools, session)
            except Exception as e:
                last_err = e
                continue
        raise last_err or RuntimeError("Todos los slots fallaron")

    def stats(self):
        return {
            "total_slots": len(config.ai_slots),
            "slots": [{
                "index": s["index"],
                "provider": detect_provider(s["key"])[0],
                "model": s["model"],
                "fails": self._fails.get(s["index"], 0),
                "cooldown": max(0, round(self._cooldowns.get(s["index"], 0) - time.monotonic(), 1)),
            } for s in config.ai_slots],
        }


connector = Connector()


# ============================================================
# TOOLS
# ============================================================

def tool_schemas():
    return [
        {"type": "function", "function": {"name": "phone_status", "description": "Estado del teléfono (batería, RAM, disco, modelo, Android, temperatura).", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "phone_battery", "description": "Batería del teléfono.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "phone_location", "description": "Ubicación GPS actual del teléfono.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "phone_wifi", "description": "Info de la red WiFi actual.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "phone_sms", "description": "Últimos SMS recibidos.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}}}},
        {"type": "function", "function": {"name": "phone_vibrate", "description": "Hacer vibrar el teléfono.", "parameters": {"type": "object", "properties": {"ms": {"type": "integer", "default": 1000}}}}},
        {"type": "function", "function": {"name": "phone_notify", "description": "Mostrar notificación.", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]}}},
        {"type": "function", "function": {"name": "phone_torch", "description": "Encender/apagar linterna.", "parameters": {"type": "object", "properties": {"state": {"type": "string", "enum": ["on", "off"]}}, "required": ["state"]}}},
        {"type": "function", "function": {"name": "phone_brightness", "description": "Ajustar brillo (0-255).", "parameters": {"type": "object", "properties": {"level": {"type": "integer", "default": 200}}}}},
        {"type": "function", "function": {"name": "phone_photo", "description": "Sacar foto (0=trasera, 1=frontal).", "parameters": {"type": "object", "properties": {"camera": {"type": "integer", "default": 0}}}}},
        {"type": "function", "function": {"name": "phone_volume", "description": "Ajustar volumen.", "parameters": {"type": "object", "properties": {"stream": {"type": "string", "default": "music"}, "level": {"type": "integer", "default": 10}}}}},
        {"type": "function", "function": {"name": "phone_clipboard_get", "description": "Leer portapapeles.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "phone_clipboard_set", "description": "Escribir portapapeles.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
        {"type": "function", "function": {"name": "phone_apps", "description": "Apps instaladas.", "parameters": {"type": "object", "properties": {"user_only": {"type": "boolean", "default": True}}}}},
        {"type": "function", "function": {"name": "phone_processes", "description": "Procesos activos.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "phone_shell", "description": "Ejecutar comando shell en el teléfono.", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},

        {"type": "function", "function": {"name": "github_list", "description": "Listar repos de GitHub.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "github_read", "description": "Leer archivo de un repo.", "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "path": {"type": "string"}, "ref": {"type": "string", "default": "main"}}, "required": ["repo", "path"]}}},
        {"type": "function", "function": {"name": "github_write", "description": "Escribir archivo en GitHub.", "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}, "message": {"type": "string"}, "branch": {"type": "string", "default": "main"}}, "required": ["repo", "path", "content", "message"]}}},
        {"type": "function", "function": {"name": "github_create_repo", "description": "Crear repositorio.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "private": {"type": "boolean", "default": False}}, "required": ["name"]}}},
        {"type": "function", "function": {"name": "github_delete_repo", "description": "ELIMINA un repositorio. IRREVERSIBLE.", "parameters": {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]}}},
        {"type": "function", "function": {"name": "github_tree", "description": "Árbol de archivos del repo.", "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "ref": {"type": "string", "default": "main"}}, "required": ["repo"]}}},
        {"type": "function", "function": {"name": "github_list_commits", "description": "Historial de commits.", "parameters": {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]}}},
        {"type": "function", "function": {"name": "github_create_issue", "description": "Crear issue.", "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}}, "required": ["repo", "title"]}}},
        {"type": "function", "function": {"name": "github_search_code", "description": "Buscar código en GitHub.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},

        {"type": "function", "function": {"name": "vercel_projects", "description": "Listar proyectos de Vercel.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "vercel_deployments", "description": "Deployments recientes de Vercel.", "parameters": {"type": "object", "properties": {}}}},

        {"type": "function", "function": {"name": "file_list", "description": "Listar archivos del workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}}}}},
        {"type": "function", "function": {"name": "file_read", "description": "Leer archivo.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "file_write", "description": "Escribir archivo.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
        {"type": "function", "function": {"name": "file_delete", "description": "Borrar archivo.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "file_tree", "description": "Árbol de archivos.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}, "max_depth": {"type": "integer", "default": 3}}}}},

        {"type": "function", "function": {"name": "web_search", "description": "Buscar en internet.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "web_fetch", "description": "Descargar URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "max_chars": {"type": "integer", "default": 8000}}, "required": ["url"]}}},

        {"type": "function", "function": {"name": "generate_image", "description": "Generar imagen con IA. Solo si el usuario lo pide explícitamente.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}}},
    ]


# ============================================================
# EJECUTORES
# ============================================================

async def _phone_call(endpoint, method="GET", params=None):
    if not config.phone_url or not config.phone_secret:
        return {"error": "PHONE_SERVER_URL o PHONE_SERVER_SECRET no configurados"}
    url = config.phone_url + endpoint
    headers = {"X-Secret": config.phone_secret}
    timeout = aiohttp.ClientTimeout(total=25)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            fn = s.post if method == "POST" else s.get
            async with fn(url, headers=headers, params=params or {}) as r:
                body = await r.text()
                if r.status == 200:
                    try:
                        return json.loads(body)
                    except Exception:
                        return {"result": body[:2000]}
                return {"error": f"HTTP {r.status}", "detail": body[:300]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def _github_request(method, path, **kwargs):
    if not config.github_token:
        return {"error": "GITHUB_TOKEN no configurado"}
    headers = {
        "Authorization": f"Bearer {config.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.request(method, "https://api.github.com" + path, headers=headers, **kwargs) as r:
            text = await r.text()
            if r.status >= 400:
                return {"error": f"GitHub {r.status}", "detail": text[:500]}
            try:
                return json.loads(text)
            except Exception:
                return {"content": text[:2000]}


async def _vercel_request(method, path):
    if not config.vercel_token:
        return {"error": "VERCEL_TOKEN no configurado"}
    headers = {"Authorization": f"Bearer {config.vercel_token}"}
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.request(method, "https://api.vercel.com" + path, headers=headers) as r:
            data = await r.json(content_type=None)
            if r.status >= 400:
                return {"error": f"Vercel {r.status}", "detail": data}
            return data


async def _web_search(query, max_results=5):
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {"error": "beautifulsoup4 no instalado"}
    url = f"https://html.duckduckgo.com/html/?q={query}"
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.get(url, headers={"User-Agent": "Mozilla/5.0"}) as r:
            html = await r.text()
    soup = BeautifulSoup(html, "lxml")
    out = []
    for res in soup.select(".result")[:max_results]:
        t = res.select_one(".result__title")
        u = res.select_one(".result__url")
        sn = res.select_one(".result__snippet")
        if t and u:
            out.append({
                "title": t.get_text(strip=True),
                "url": u.get_text(strip=True),
                "snippet": sn.get_text(strip=True) if sn else "",
            })
    return {"query": query, "results": out}


async def _web_fetch(url, max_chars=8000):
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {"error": "beautifulsoup4 no instalado"}
    timeout = aiohttp.ClientTimeout(total=20)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url, headers={"User-Agent": "Mozilla/5.0"}) as r:
                if r.status >= 400:
                    return {"error": f"HTTP {r.status}"}
                html = await r.text()
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return {"url": url, "text": text[:max_chars]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def _gen_image(prompt):
    if not prompt.strip():
        return {"error": "prompt vacío"}
    import urllib.parse
    enc = urllib.parse.quote(prompt)
    for model in ["flux", "turbo", "flux-realism"]:
        url = f"https://image.pollinations.ai/prompt/{enc}?model={model}&width=1024&height=1024&nologo=true"
        try:
            timeout = aiohttp.ClientTimeout(total=50)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(url) as r:
                    if r.status != 200:
                        continue
                    data = await r.read()
                    if len(data) < 1000:
                        continue
                    return {
                        "ok": True,
                        "model": model,
                        "image_base64": base64.b64encode(data).decode(),
                        "prompt": prompt,
                    }
        except Exception:
            continue
    return {"error": "No se pudo generar la imagen tras varios intentos"}


def _safe_path(p):
    target = (config.workspace / p).resolve()
    if not str(target).startswith(str(config.workspace.resolve())):
        raise ValueError("Ruta fuera del workspace")
    return target


def _file_write(path, content):
    t = _safe_path(path)
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text(content, encoding="utf-8")
    return str(t.relative_to(config.workspace))


def _file_read(path):
    return _safe_path(path).read_text(encoding="utf-8", errors="replace")


def _file_list(path="."):
    t = _safe_path(path)
    if not t.exists():
        return []
    return sorted([str(p.relative_to(config.workspace)) for p in t.iterdir()])


def _file_delete(path):
    t = _safe_path(path)
    if not t.exists():
        return False
    if t.is_dir():
        import shutil
        shutil.rmtree(t)
    else:
        t.unlink()
    return True


def _file_tree(path=".", max_depth=3):
    def walk(p, d):
        if d > max_depth:
            return "..."
        if p.is_file():
            return p.stat().st_size
        return {c.name: walk(c, d + 1) for c in sorted(p.iterdir())}
    return walk(_safe_path(path), 0)


# ============================================================
# RUN TOOL
# ============================================================

async def run_tool(name, args):
    try:
        if name.startswith("phone_"):
            mapping = {
                "phone_status": ("/status", "GET", {}),
                "phone_battery": ("/battery", "GET", {}),
                "phone_location": ("/location", "GET", {}),
                "phone_wifi": ("/wifi", "GET", {}),
                "phone_sms": ("/sms", "GET", {"limit": args.get("limit", 10)}),
                "phone_vibrate": ("/vibrate", "POST", {"ms": args.get("ms", 1000)}),
                "phone_notify": ("/notify", "POST", {"title": args.get("title", "Subtom"), "content": args.get("content", "")}),
                "phone_torch": ("/torch", "POST", {"state": args.get("state", "on")}),
                "phone_brightness": ("/brightness", "POST", {"level": args.get("level", 200)}),
                "phone_photo": ("/photo", "POST", {"camera": args.get("camera", 0)}),
                "phone_volume": ("/volume", "POST", {"stream": args.get("stream", "music"), "level": args.get("level", 10)}),
                "phone_clipboard_get": ("/clipboard/get", "GET", {}),
                "phone_clipboard_set": ("/clipboard/set", "POST", {"text": args.get("text", "")}),
                "phone_apps": ("/apps", "GET", {"user_only": "1" if args.get("user_only", True) else "0"}),
                "phone_processes": ("/processes", "GET", {}),
                "phone_shell": ("/shell", "POST", {"cmd": args.get("cmd", "")}),
            }
            if name in mapping:
                ep, m, p = mapping[name]
                return await _phone_call(ep, m, p)
            return {"error": f"Phone tool desconocida: {name}"}

        if name == "github_list":
            return await _github_request("GET", "/user/repos?per_page=50")
        if name == "github_read":
            repo = args["repo"].strip("/")
            path = args["path"].lstrip("/")
            ref = args.get("ref", "main")
            return await _github_request("GET", f"/repos/{repo}/contents/{path}?ref={ref}")
        if name == "github_tree":
            repo = args["repo"].strip("/")
            ref = args.get("ref", "main")
            return await _github_request("GET", f"/repos/{repo}/git/trees/{ref}?recursive=1")
        if name == "github_list_commits":
            repo = args["repo"].strip("/")
            return await _github_request("GET", f"/repos/{repo}/commits?per_page=20")
        if name == "github_create_repo":
            return await _github_request("POST", "/user/repos", json={
                "name": args["name"],
                "description": args.get("description", ""),
                "private": bool(args.get("private", False)),
                "auto_init": True,
            })
        if name == "github_delete_repo":
            return await _github_request("DELETE", f"/repos/{args['repo'].strip('/')}")
        if name == "github_create_issue":
            return await _github_request("POST", f"/repos/{args['repo'].strip('/')}/issues",
                                         json={"title": args["title"], "body": args.get("body", "")})
        if name == "github_search_code":
            return await _github_request("GET", f"/search/code?q={args['query']}")
        if name == "github_write":
            repo = args["repo"].strip("/")
            path = args["path"].lstrip("/")
            branch = args.get("branch", "main")
            sha = None
            existing = await _github_request("GET", f"/repos/{repo}/contents/{path}?ref={branch}")
            if isinstance(existing, dict) and existing.get("sha"):
                sha = existing["sha"]
            payload = {
                "message": args["message"],
                "content": base64.b64encode(args["content"].encode()).decode(),
                "branch": branch,
            }
            if sha:
                payload["sha"] = sha
            return await _github_request("PUT", f"/repos/{repo}/contents/{path}", json=payload)

        if name == "vercel_projects":
            return await _vercel_request("GET", "/v9/projects?limit=50")
        if name == "vercel_deployments":
            return await _vercel_request("GET", "/v6/deployments?limit=20")

        if name == "file_list":
            return {"files": _file_list(args.get("path", "."))}
        if name == "file_read":
            return {"content": _file_read(args["path"])}
        if name == "file_write":
            return {"path": _file_write(args["path"], args["content"])}
        if name == "file_delete":
            return {"deleted": _file_delete(args["path"])}
        if name == "file_tree":
            return _file_tree(args.get("path", "."), int(args.get("max_depth", 3)))

        if name == "web_search":
            return await _web_search(args["query"], int(args.get("max_results", 5)))
        if name == "web_fetch":
            return await _web_fetch(args["url"], int(args.get("max_chars", 8000)))

        if name == "generate_image":
            return await _gen_image(args["prompt"])

        return {"error": f"Herramienta desconocida: {name}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:300]}"}


# ============================================================
# AGENTE
# ============================================================

async def agent_ask(prompt, history=None, max_rounds=8):
    messages = list(history or [])
    messages.append({"role": "user", "content": prompt})

    used = []
    for _ in range(max_rounds):
        resp = await connector.complete(messages, tool_schemas())
        choice = resp["choices"][0]
        msg = choice["message"]
        tool_calls = msg.get("tool_calls") or []

        a_msg = {"role": "assistant", "content": msg.get("content")}
        if tool_calls:
            a_msg["tool_calls"] = tool_calls
        messages.append(a_msg)

        if not tool_calls:
            return {
                "response": msg.get("content") or "",
                "history": messages,
                "tools_used": used,
            }

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            result = await run_tool(name, args)
            used.append(name)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": json.dumps(result, ensure_ascii=False, default=str)[:20000],
            })

    return {"response": "[Se alcanzó el límite de rondas]", "history": messages, "tools_used": used}


# ============================================================
# HANDLER HTTP (Vercel)
# ============================================================

class handler(BaseHTTPRequestHandler):
    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode())

    def do_OPTIONS(self):
        self._json(200, {"ok": True})

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path in ("", "/api"):
            return self._json(200, {
                "service": "Subtom IA API",
                "version": "1.0",
                "endpoints": [
                    "GET  /api          → info",
                    "GET  /api/status   → estado",
                    "GET  /api/tools    → lista de herramientas",
                    "POST /api/chat     → {prompt, history}",
                    "POST /api/tool     → {tool, args}",
                ],
                "stats": connector.stats(),
            })

        if path == "/api/status":
            return self._json(200, {
                "ok": True,
                "stats": connector.stats(),
                "phone_configured": bool(config.phone_url and config.phone_secret),
                "github_configured": bool(config.github_token),
                "vercel_configured": bool(config.vercel_token),
            })

        if path == "/api/tools":
            names = [s["function"]["name"] for s in tool_schemas()]
            return self._json(200, {"total": len(names), "tools": names})

        return self._json(404, {"error": "Not found", "path": path})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}

        path = self.path.split("?")[0].rstrip("/")

        if path == "/api/chat":
            prompt = body.get("prompt", "").strip()
            history = body.get("history", [])
            if not prompt:
                return self._json(400, {"error": "prompt vacío"})
            try:
                result = asyncio.run(agent_ask(prompt, history))
                return self._json(200, result)
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        if path == "/api/tool":
            tool = body.get("tool", "").strip()
            args = body.get("args", {})
            if not tool:
                return self._json(400, {"error": "tool vacío"})
            result = asyncio.run(run_tool(tool, args))
            return self._json(200, {"tool": tool, "result": result})

        return self._json(404, {"error": "Not found", "path": path})
