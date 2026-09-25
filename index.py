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
                self._fail(slot, 120); raise RuntimeError(f"404 modelo")
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
