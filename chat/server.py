from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import json
import mimetypes
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.chat_engine import OwnAIEngine
from documents.ingest import ingest
from storage.app_store import AppStore
from vision.image_analyzer import LocalVision


HOST = "127.0.0.1"
PORT = 8000
MAX_BODY = 20 * 1024 * 1024

STATIC = ROOT / "chat" / "static"

store = AppStore(ROOT)
engine = OwnAIEngine()
vision = LocalVision()


def json_response(handler, status, payload):
    data = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    handler.send_response(status)
    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8",
    )
    handler.send_header(
        "Content-Length",
        str(len(data)),
    )
    handler.send_header(
        "Cache-Control",
        "no-store",
    )
    handler.end_headers()
    handler.wfile.write(data)


def auth_user(handler):
    token = handler.headers.get(
        "X-OwnAI-Token",
        "",
    )
    return store.user_from_token(token)


def safe_name(name):
    value = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        Path(name).name,
    )
    return value[:120] or "upload.bin"


class Handler(BaseHTTPRequestHandler):
    server_version = "OwnAI/7.0"

    def do_GET(self):
        path = urlparse(
            self.path
        ).path

        if path == "/":
            return self.serve_static(
                "index.html",
                "text/html; charset=utf-8",
            )

        if path.startswith("/static/"):
            relative = path.removeprefix(
                "/static/"
            )

            return self.serve_static(
                relative,
                mimetypes.guess_type(
                    relative
                )[0]
                or "application/octet-stream",
            )

        if path == "/api/info":
            return json_response(
                self,
                200,
                engine.info(),
            )

        username = auth_user(self)

        if path == "/api/me":
            if not username:
                return json_response(
                    self,
                    401,
                    {"authenticated": False},
                )

            return json_response(
                self,
                200,
                {
                    "authenticated": True,
                    "username": username,
                },
            )

        if path == "/api/chats":
            if not username:
                return json_response(
                    self,
                    401,
                    {"error": "Login required."},
                )

            return json_response(
                self,
                200,
                {
                    "chats": store.list_chats(
                        username
                    ),
                },
            )

        match = re.fullmatch(
            r"/api/chats/([A-Za-z0-9_-]+)",
            path,
        )

        if match:
            if not username:
                return json_response(
                    self,
                    401,
                    {"error": "Login required."},
                )

            try:
                chat = store.get_chat(
                    username,
                    match.group(1),
                )
            except ValueError as exc:
                return json_response(
                    self,
                    404,
                    {"error": str(exc)},
                )

            return json_response(
                self,
                200,
                {"chat": chat},
            )

        return json_response(
            self,
            404,
            {"error": "Not found."},
        )

    def do_POST(self):
        path = urlparse(
            self.path
        ).path

        try:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )
        except ValueError:
            length = 0

        if length > MAX_BODY:
            return json_response(
                self,
                413,
                {
                    "error":
                        "Request is too large."
                },
            )

        try:
            raw = self.rfile.read(length)
            payload = json.loads(
                raw.decode("utf-8")
            )
        except (ValueError, json.JSONDecodeError):
            return json_response(
                self,
                400,
                {
                    "error":
                        "Invalid JSON request."
                },
            )

        if path == "/api/register":
            try:
                username = store.register(
                    str(
                        payload.get(
                            "username",
                            "",
                        )
                    ),
                    str(
                        payload.get(
                            "password",
                            "",
                        )
                    ),
                )

                token = store.login(
                    username,
                    str(
                        payload.get(
                            "password",
                            "",
                        )
                    ),
                )

                return json_response(
                    self,
                    200,
                    {
                        "token": token,
                        "username": username,
                    },
                )

            except ValueError as exc:
                return json_response(
                    self,
                    400,
                    {"error": str(exc)},
                )

        if path == "/api/login":
            try:
                token = store.login(
                    str(
                        payload.get(
                            "username",
                            "",
                        )
                    ),
                    str(
                        payload.get(
                            "password",
                            "",
                        )
                    ),
                )

                return json_response(
                    self,
                    200,
                    {
                        "token": token,
                        "username": store.user_from_token(
                            token
                        ),
                    },
                )

            except ValueError as exc:
                return json_response(
                    self,
                    401,
                    {"error": str(exc)},
                )

        if path == "/api/logout":
            store.logout(
                self.headers.get(
                    "X-OwnAI-Token",
                    "",
                )
            )

            return json_response(
                self,
                200,
                {"ok": True},
            )

        username = auth_user(self)

        if not username:
            return json_response(
                self,
                401,
                {
                    "error":
                        "Login required."
                },
            )

        if path == "/api/chats/new":
            chat = store.new_chat(
                username,
                str(
                    payload.get(
                        "title",
                        "New chat",
                    )
                ),
            )

            return json_response(
                self,
                200,
                {"chat": chat},
            )

        if path == "/api/chat":
            message = str(
                payload.get(
                    "message",
                    "",
                )
            ).strip()

            chat_id = str(
                payload.get(
                    "chat_id",
                    "",
                )
            ).strip()

            use_rag = bool(
                payload.get(
                    "use_rag",
                    True,
                )
            )

            mode = str(
                payload.get(
                    "mode",
                    "auto",
                )
            )

            if not message:
                return json_response(
                    self,
                    400,
                    {
                        "error":
                            "Message is empty."
                    },
                )

            try:
                if not chat_id:
                    chat = store.new_chat(
                        username,
                        message[:60],
                    )

                    chat_id = chat["id"]
                else:
                    chat = store.get_chat(
                        username,
                        chat_id,
                    )

                recent_history = [
                    (
                        item["role"],
                        item["content"],
                    )
                    for item
                    in chat["messages"][-8:]
                ]

                recalled = store.search_memory(
                    username,
                    message,
                    top_k=4,
                )

                memory_history = [
                    (
                        "Memory",
                        item["content"],
                    )
                    for item in recalled
                ]

                # Keep the current conversation recent turns first, then
                # add a few relevant older memories when available.
                history = (
                    recent_history
                    + memory_history
                )[:12]

                prompt = message

                if (
                    mode == "web"
                    and not re.match(
                        r"^(search the web|search online|look this up)\b",
                        message,
                        re.I,
                    )
                ):
                    prompt = (
                        "search the web "
                        + message
                    )

                store.append_message(
                    username,
                    chat_id,
                    "user",
                    message,
                )

                answer = engine.generate(
                    prompt,
                    use_rag=use_rag,
                    history=history,
                )

                store.append_message(
                    username,
                    chat_id,
                    "assistant",
                    answer,
                )

                return json_response(
                    self,
                    200,
                    {
                        "answer": answer,
                        "chat_id": chat_id,
                    },
                )

            except ValueError as exc:
                return json_response(
                    self,
                    404,
                    {"error": str(exc)},
                )

            except Exception as exc:
                return json_response(
                    self,
                    500,
                    {"error": str(exc)},
                )

        if path == "/api/delete-chat":
            chat_id = str(
                payload.get(
                    "chat_id",
                    "",
                )
            ).strip()

            try:
                store.delete_chat(
                    username,
                    chat_id,
                )

                return json_response(
                    self,
                    200,
                    {"ok": True},
                )

            except ValueError as exc:
                return json_response(
                    self,
                    404,
                    {"error": str(exc)},
                )

        if path == "/api/upload":
            name = safe_name(
                str(
                    payload.get(
                        "name",
                        "upload.bin",
                    )
                )
            )

            encoded = str(
                payload.get(
                    "content_base64",
                    "",
                )
            )

            if not encoded:
                return json_response(
                    self,
                    400,
                    {
                        "error":
                            "No file data."
                    },
                )

            if len(encoded) > 19 * 1024 * 1024:
                return json_response(
                    self,
                    413,
                    {
                        "error":
                            "File is too large."
                    },
                )

            try:
                file_bytes = base64.b64decode(
                    encoded,
                    validate=True,
                )
            except ValueError:
                return json_response(
                    self,
                    400,
                    {
                        "error":
                            "Invalid file encoding."
                    },
                )

            if len(file_bytes) > 15 * 1024 * 1024:
                return json_response(
                    self,
                    413,
                    {
                        "error":
                            "File limit is 15 MB."
                    },
                )

            upload_dir = (
                ROOT
                / "data"
                / "uploads"
                / username
            )

            upload_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output = upload_dir / name
            output.write_bytes(
                file_bytes
            )

            try:
                ext = output.suffix.lower()

                if ext in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".bmp",
                    ".gif",
                }:
                    info = vision.metadata(
                        file_bytes
                    )

                    return json_response(
                        self,
                        200,
                        {
                            "type": "image",
                            "name": name,
                            "metadata": info,
                            "message":
                                "Image uploaded.",
                        },
                    )

                result = ingest(
                    output,
                    upload_dir,
                )

                engine.retriever.build()

                return json_response(
                    self,
                    200,
                    {
                        "type":
                            "document",
                        **result,
                        "message":
                            "Document indexed for local Q&A.",
                    },
                )

            except Exception as exc:
                return json_response(
                    self,
                    400,
                    {"error": str(exc)},
                )

        if path == "/api/vision":
            encoded = str(
                payload.get(
                    "content_base64",
                    "",
                )
            )

            prompt = str(
                payload.get(
                    "prompt",
                    "",
                )
            ).strip()

            try:
                file_bytes = base64.b64decode(
                    encoded,
                    validate=True,
                )

                if len(file_bytes) > 15 * 1024 * 1024:
                    raise ValueError(
                        "Image limit is 15 MB."
                    )

                answer = vision.describe(
                    file_bytes,
                    prompt=prompt or None,
                )

                return json_response(
                    self,
                    200,
                    {"answer": answer},
                )

            except Exception as exc:
                return json_response(
                    self,
                    400,
                    {"error": str(exc)},
                )

        return json_response(
            self,
            404,
            {"error": "Not found."},
        )

    def serve_static(self, relative, content_type):
        try:
            target = (
                STATIC / relative
            ).resolve()

            if (
                STATIC.resolve()
                not in target.parents
            ):
                raise ValueError(
                    "Invalid path."
                )

            if (
                not target.exists()
                or not target.is_file()
            ):
                raise FileNotFoundError()

            data = target.read_bytes()

            self.send_response(200)
            self.send_header(
                "Content-Type",
                content_type,
            )
            self.send_header(
                "Content-Length",
                str(len(data)),
            )
            self.end_headers()
            self.wfile.write(data)

        except (
            OSError,
            ValueError,
        ):
            json_response(
                self,
                404,
                {"error": "Not found."},
            )


def main():
    STATIC.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("OWN AI v7 LOCAL WEB APP")
    print("=" * 60)
    print("Stage:", engine.stage)
    print("Device:", engine.device)
    print(
        "URL:",
        f"http://{HOST}:{PORT}",
    )
    print("Local-only: True")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Own AI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
