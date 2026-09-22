from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.chat_engine import OwnAIEngine


HOST = "127.0.0.1"
PORT = 8000


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Own AI</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:#0b0d10;color:#eef2f7}
.app{height:100vh;display:flex;overflow:hidden}
.side{width:270px;background:#10141a;border-right:1px solid #202631;padding:18px;display:flex;flex-direction:column;gap:16px}
.brand{font-size:22px;font-weight:700}
.sub{font-size:12px;color:#8d98a8}
.btn{border:1px solid #2a3240;background:#171c24;color:#fff;border-radius:10px;padding:11px 12px;cursor:pointer}
.btn:hover{background:#1d2430}
.info{margin-top:auto;font-size:12px;line-height:1.6;color:#9da7b6;white-space:pre-line}
.main{flex:1;display:flex;flex-direction:column;min-width:0}
.top{padding:16px 22px;border-bottom:1px solid #202631;background:#0f1217;display:flex;justify-content:space-between;gap:12px;align-items:center}
.status{font-size:12px;color:#8d98a8}
.chat{flex:1;overflow:auto;padding:28px max(18px,calc((100vw - 940px)/2))}
.msg{max-width:850px;margin:0 auto 20px;padding:16px 18px;border-radius:16px;line-height:1.55;white-space:pre-wrap}
.user{background:#182231}
.ai{background:#11161e;border:1px solid #222b37}
.role{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#8d98a8;margin-bottom:7px}
.composer{padding:14px max(18px,calc((100vw - 940px)/2));border-top:1px solid #202631;background:#0f1217}
.row{display:flex;gap:10px}
textarea{flex:1;resize:none;min-height:54px;max-height:180px;border-radius:14px;border:1px solid #293241;background:#151a21;color:#fff;padding:14px;outline:none}
textarea:focus{border-color:#4b5d77}
.send{width:90px}
.controls{display:flex;gap:12px;align-items:center;margin-top:8px;font-size:12px;color:#8d98a8}
label{display:flex;align-items:center;gap:6px}
@media(max-width:700px){.side{display:none}.chat{padding:18px}.composer{padding:12px}.top{padding:14px}}
</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <div>
      <div class="brand">Own AI</div>
      <div class="sub">Your locally trained AI</div>
    </div>
    <button class="btn" onclick="newChat()">＋ New chat</button>
    <button class="btn" onclick="showInfo()">Model info</button>
    <div class="info" id="info">Loading model…</div>
  </aside>

  <main class="main">
    <header class="top">
      <div>
        <b>Own AI</b>
        <div class="status" id="status">Ready</div>
      </div>
      <button class="btn" onclick="newChat()">Clear memory</button>
    </header>

    <section class="chat" id="chat">
      <div class="msg ai">
        <div class="role">Own AI</div>
        Hello! I am your local Own AI model. Ask me something.
      </div>
    </section>

    <footer class="composer">
      <div class="row">
        <textarea id="input" placeholder="Message Own AI…" onkeydown="key(event)"></textarea>
        <button class="btn send" id="send" onclick="send()">Send</button>
      </div>
      <div class="controls">
        <label><input id="rag" type="checkbox" checked> use local knowledge</label>
        <span>Enter = send · Shift+Enter = new line</span>
      </div>
    </footer>
  </main>
</div>

<script>
const input=document.getElementById('input');
const chat=document.getElementById('chat');
const statusEl=document.getElementById('status');
const sendBtn=document.getElementById('send');
const infoEl=document.getElementById('info');

function escapeHtml(s){
  return s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function add(role,text){
  const box=document.createElement('div');
  box.className='msg '+(role==='You'?'user':'ai');
  box.innerHTML='<div class="role">'+role+'</div>'+escapeHtml(text);
  chat.appendChild(box);
  chat.scrollTop=chat.scrollHeight;
}
function key(e){
  if(e.key==='Enter'&&!e.shiftKey){
    e.preventDefault();
    send();
  }
}
async function send(){
  const text=input.value.trim();
  if(!text)return;

  add('You',text);
  input.value='';
  sendBtn.disabled=true;
  statusEl.textContent='Thinking…';

  try{
    const r=await fetch('/api/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        message:text,
        use_rag:document.getElementById('rag').checked
      })
    });

    const data=await r.json();

    if(data.error) throw new Error(data.error);

    add('Own AI',data.answer);
  }catch(e){
    add('Own AI','Error: '+e.message);
  }finally{
    sendBtn.disabled=false;
    statusEl.textContent='Ready';
    input.focus();
  }
}
async function newChat(){
  await fetch('/api/reset',{method:'POST'});
  chat.innerHTML='';
  add('Own AI','New conversation started.');
}
async function showInfo(){
  const r=await fetch('/api/info');
  const d=await r.json();
  alert(JSON.stringify(d,null,2));
}
fetch('/api/info').then(r=>r.json()).then(d=>{
  infoEl.textContent=
    d.stage+' · '+Number(d.parameters).toLocaleString()+' params
'+
    d.layers+' layers · '+d.retrieval_documents+' local chunks';
});
</script>
</body>
</html>"""


engine = OwnAIEngine()


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        data = body.encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            self._send(
                200,
                HTML,
                "text/html; charset=utf-8",
            )
            return

        if path == "/api/info":
            self._send(
                200,
                json.dumps(engine.info()),
            )
            return

        self._send(
            404,
            json.dumps({"error": "Not found"}),
        )

    def do_POST(self):
        path = urlparse(self.path).path

        try:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )
            raw = self.rfile.read(length)
            payload = json.loads(
                raw.decode("utf-8")
            )
        except (ValueError, json.JSONDecodeError):
            self._send(
                400,
                json.dumps({"error": "Invalid request"}),
            )
            return

        if path == "/api/reset":
            engine.reset_memory()

            self._send(
                200,
                json.dumps({"ok": True}),
            )
            return

        if path == "/api/chat":
            message = str(
                payload.get(
                    "message",
                    "",
                )
            ).strip()

            use_rag = bool(
                payload.get(
                    "use_rag",
                    True,
                )
            )

            if not message:
                self._send(
                    400,
                    json.dumps(
                        {"error": "Message is empty"}
                    ),
                )
                return

            try:
                answer = engine.generate(
                    message,
                    use_rag=use_rag,
                )

                self._send(
                    200,
                    json.dumps(
                        {"answer": answer},
                        ensure_ascii=False,
                    ),
                )
            except Exception as exc:
                self._send(
                    500,
                    json.dumps(
                        {"error": str(exc)}
                    ),
                )

            return

        self._send(
            404,
            json.dumps({"error": "Not found"}),
        )


def main():
    print("=" * 56)
    print("OWN AI WEB CHAT")
    print("=" * 56)
    print("Stage:", engine.stage)
    print("Device:", engine.device)
    print("URL: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    print("=" * 56)

    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("
Stopping Own AI…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
