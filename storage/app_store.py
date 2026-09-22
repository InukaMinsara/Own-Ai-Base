from pathlib import Path
import hashlib
import hmac
import json
import secrets
import time


class AppStore:
    """Small local-only persistent store for accounts and saved chats."""

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "data" / "runtime" / "app_store.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self):
        if not self.path.exists():
            return {
                "users": {},
                "sessions": {},
                "chats": {},
            }

        try:
            return json.loads(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return {
                "users": {},
                "sessions": {},
                "chats": {},
            }

    def _save(self):
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.path)

    @staticmethod
    def _hash_password(password, salt_hex=None):
        salt = (
            bytes.fromhex(salt_hex)
            if salt_hex
            else secrets.token_bytes(16)
        )
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            180_000,
        )
        return salt.hex(), digest.hex()

    def register(self, username, password):
        username = username.strip().lower()

        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters.")

        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters.")

        if username in self.data["users"]:
            raise ValueError("Username already exists.")

        salt, digest = self._hash_password(password)

        self.data["users"][username] = {
            "password_salt": salt,
            "password_hash": digest,
            "created_at": time.time(),
        }

        self._save()
        return username

    def login(self, username, password):
        username = username.strip().lower()
        user = self.data["users"].get(username)

        if not user:
            raise ValueError("Invalid username or password.")

        salt = user["password_salt"]
        _, digest = self._hash_password(
            password,
            salt_hex=salt,
        )

        if not hmac.compare_digest(
            digest,
            user["password_hash"],
        ):
            raise ValueError("Invalid username or password.")

        token = secrets.token_urlsafe(32)
        self.data["sessions"][token] = {
            "username": username,
            "created_at": time.time(),
        }
        self._save()
        return token

    def logout(self, token):
        self.data["sessions"].pop(token, None)
        self._save()

    def user_from_token(self, token):
        session = self.data["sessions"].get(token)
        return session["username"] if session else None

    def new_chat(self, username, title="New chat"):
        chat_id = secrets.token_urlsafe(12)

        self.data["chats"][chat_id] = {
            "id": chat_id,
            "username": username,
            "title": title[:80] or "New chat",
            "created_at": time.time(),
            "updated_at": time.time(),
            "messages": [],
        }

        self._save()
        return self.data["chats"][chat_id]

    def list_chats(self, username):
        chats = [
            chat
            for chat in self.data["chats"].values()
            if chat["username"] == username
        ]
        return sorted(
            chats,
            key=lambda c: c["updated_at"],
            reverse=True,
        )

    def get_chat(self, username, chat_id):
        chat = self.data["chats"].get(chat_id)

        if not chat or chat["username"] != username:
            raise ValueError("Chat not found.")

        return chat

    def append_message(self, username, chat_id, role, content):
        chat = self.get_chat(username, chat_id)

        chat["messages"].append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })

        if role == "user" and chat["title"] == "New chat":
            chat["title"] = content.strip()[:60] or "New chat"

        chat["updated_at"] = time.time()
        self._save()

        return chat

    def delete_chat(self, username, chat_id):
        chat = self.get_chat(username, chat_id)
        del self.data["chats"][chat_id]
        self._save()
        return chat_id
