let token = localStorage.getItem("ownai_token") || "";
let username = localStorage.getItem("ownai_user") || "";
let currentChatId = null;
let webMode = false;
let lastAnswer = "";
let registerMode = false;

const $ = id => document.getElementById(id);

function api(path, options={}){
  const headers = Object.assign(
    {"Content-Type":"application/json"},
    options.headers || {}
  );

  if(token){
    headers["X-OwnAI-Token"] = token;
  }

  return fetch(
    path,
    Object.assign({}, options, {headers})
  );
}

function setAuthMode(register){
  registerMode = register;
  $("loginTab").classList.toggle("active", !register);
  $("registerTab").classList.toggle("active", register);
  $("authButton").textContent =
    register ? "Create account" : "Login";
  $("authStatus").textContent = "";
}

$("loginTab").onclick = () => setAuthMode(false);
$("registerTab").onclick = () => setAuthMode(true);

$("authButton").onclick = async ()=>{
  $("authStatus").textContent = "";

  const user = $("authUser").value.trim();
  const pass = $("authPass").value;

  if(!user || !pass){
    $("authStatus").textContent =
      "Enter username and password.";
    return;
  }

  const endpoint =
    registerMode
      ? "/api/register"
      : "/api/login";

  try{
    const r = await api(endpoint,{
      method:"POST",
      body:JSON.stringify({
        username:user,
        password:pass
      })
    });

    const data = await r.json();

    if(data.error){
      throw new Error(data.error);
    }

    token = data.token;
    username = data.username;

    localStorage.setItem(
      "ownai_token",
      token
    );

    localStorage.setItem(
      "ownai_user",
      username
    );

    await boot();
  }catch(error){
    $("authStatus").textContent =
      error.message;
  }
};

$("authPass").addEventListener(
  "keydown",
  event=>{
    if(event.key === "Enter"){
      $("authButton").click();
    }
  }
);

async function boot(){
  if(!token){
    $("loginScreen").classList.remove("hidden");
    $("app").classList.add("hidden");
    return;
  }

  try{
    const r = await api("/api/me");

    if(!r.ok){
      throw new Error();
    }

    const me = await r.json();

    if(!me.authenticated){
      throw new Error();
    }

    $("loginScreen").classList.add("hidden");
    $("app").classList.remove("hidden");
    $("userLabel").textContent =
      "Signed in as " + username;

    const chats = await getChats();

    if(chats.length){
      await openChat(chats[0].id);
    }else{
      await newChat();
    }
  }catch{
    token = "";
    username = "";
    localStorage.removeItem("ownai_token");
    localStorage.removeItem("ownai_user");

    $("loginScreen").classList.remove("hidden");
    $("app").classList.add("hidden");
  }
}

async function getChats(){
  const r = await api("/api/chats");
  const data = await r.json();
  return data.chats || [];
}

async function loadChats(){
  const chats = await getChats();
  $("chatList").innerHTML = "";

  chats.forEach(chat=>{
    const button =
      document.createElement("button");

    button.className =
      "chat-item"
      + (
        chat.id === currentChatId
          ? " active"
          : ""
      );

    button.innerHTML =
      '<div class="chat-title"></div>' +
      '<div class="chat-date"></div>';

    button.querySelector(
      ".chat-title"
    ).textContent = chat.title;

    button.querySelector(
      ".chat-date"
    ).textContent =
      new Date(
        chat.updated_at * 1000
      ).toLocaleString();

    button.onclick = () =>
      openChat(chat.id);

    $("chatList").appendChild(button);
  });
}

function clearMessages(){
  $("messages").innerHTML = "";
}

function addMessage(role,text){
  const box =
    document.createElement("div");

  box.className =
    "message " + role;

  const label =
    document.createElement("div");

  label.className = "role";
  label.textContent =
    role === "user"
      ? "You"
      : "Own AI";

  const body =
    document.createElement("div");

  body.textContent = text;

  box.appendChild(label);
  box.appendChild(body);

  $("messages").appendChild(box);
  $("messages").scrollTop =
    $("messages").scrollHeight;

  if(role === "assistant"){
    lastAnswer = text;
  }

  return body;
}

async function openChat(id){
  const r = await api(
    "/api/chats/" +
    encodeURIComponent(id)
  );

  const data = await r.json();

  if(data.error){
    return;
  }

  currentChatId = id;

  clearMessages();

  const list =
    data.chat.messages || [];

  if(!list.length){
    addMessage(
      "assistant",
      "New conversation started."
    );
  }

  list.forEach(item=>{
    addMessage(
      item.role === "user"
        ? "user"
        : "assistant",
      item.content
    );
  });

  await loadChats();
  $("sidebar").classList.remove("open");
}

$("newChat").onclick = newChat;

async function newChat(){
  const r = await api(
    "/api/chats/new",
    {
      method:"POST",
      body:JSON.stringify({
        title:"New chat"
      })
    }
  );

  const data = await r.json();

  if(data.error){
    return;
  }

  currentChatId =
    data.chat.id;

  clearMessages();

  addMessage(
    "assistant",
    "New conversation started. Ask me anything."
  );

  await loadChats();
}

$("logout").onclick = async ()=>{
  await api(
    "/api/logout",
    {
      method:"POST"
    }
  );

  token = "";
  username = "";

  localStorage.removeItem(
    "ownai_token"
  );

  localStorage.removeItem(
    "ownai_user"
  );

  location.reload();
};

$("modelInfo").onclick = async ()=>{
  const r = await api(
    "/api/info"
  );

  const data = await r.json();

  $("modelInfoText").textContent =
    JSON.stringify(
      data,
      null,
      2
    );

  $("modal").classList.remove(
    "hidden"
  );
};

$("closeModal").onclick = () =>
  $("modal").classList.add("hidden");

$("openSide").onclick = () =>
  $("sidebar").classList.add("open");

$("closeSide").onclick = () =>
  $("sidebar").classList.remove("open");

$("webMode").onclick = ()=>{
  webMode = !webMode;

  $("webMode").style.borderColor =
    webMode ? "#4f77b2" : "";

  $("status").textContent =
    webMode
      ? "Web mode"
      : "Ready";
};

async function send(){
  const text =
    $("input").value.trim();

  if(!text || !currentChatId){
    return;
  }

  $("input").value = "";
  $("input").style.height =
    "auto";

  addMessage(
    "user",
    text
  );

  $("status").textContent =
    "Thinking...";

  $("send").disabled = true;

  try{
    const r = await api(
      "/api/chat",
      {
        method:"POST",
        body:JSON.stringify({
          chat_id:
            currentChatId,
          message:text,
          use_rag:
            $("rag").checked,
          mode:
            webMode
              ? "web"
              : "auto"
        })
      }
    );

    const data = await r.json();

    if(data.error){
      throw new Error(
        data.error
      );
    }

    lastAnswer =
      data.answer || "";

    addMessage(
      "assistant",
      lastAnswer
    );

    await loadChats();
  }catch(error){
    addMessage(
      "assistant",
      "Error: " +
      error.message
    );
  }finally{
    $("send").disabled = false;

    $("status").textContent =
      webMode
        ? "Web mode"
        : "Ready";

    $("input").focus();
  }
}

$("send").onclick = send;

$("input").addEventListener(
  "keydown",
  event=>{
    if(
      event.key === "Enter"
      && !event.shiftKey
    ){
      event.preventDefault();
      send();
    }
  }
);

$("input").addEventListener(
  "input",
  ()=>{
    $("input").style.height =
      "auto";

    $("input").style.height =
      Math.min(
        $("input").scrollHeight,
        180
      ) + "px";
  }
);

document.querySelectorAll(
  "[data-quick]"
).forEach(button=>{
  button.onclick = ()=>{
    $("input").value =
      button.dataset.quick;

    $("input").focus();
    $("input").dispatchEvent(
      new Event("input")
    );
  };
});

async function toBase64(file){
  const bytes =
    await file.arrayBuffer();

  let binary = "";
  const chunk = 0x8000;

  for(
    let i=0;
    i<bytes.byteLength;
    i+=chunk
  ){
    binary += String.fromCharCode(
      ...new Uint8Array(
        bytes.slice(
          i,
          i+chunk
        )
      )
    );
  }

  return btoa(binary);
}

$("fileInput").onchange =
  async event=>{
    const file =
      event.target.files[0];

    if(!file){
      return;
    }

    $("attachment").textContent =
      "Uploading " +
      file.name +
      "...";

    $("attachment").classList.remove(
      "hidden"
    );

    try{
      const encoded =
        await toBase64(file);

      const r = await api(
        "/api/upload",
        {
          method:"POST",
          body:JSON.stringify({
            name:file.name,
            content_base64:encoded
          })
        }
      );

      const data =
        await r.json();

      if(data.error){
        throw new Error(
          data.error
        );
      }

      addMessage(
        "assistant",
        data.message ||
        "File processed."
      );

      $("attachment").textContent =
        file.name +
        " · indexed";
    }catch(error){
      $("attachment").textContent =
        "Upload error: " +
        error.message;
    }

    event.target.value = "";
  };

$("imageInput").onchange =
  async event=>{
    const file =
      event.target.files[0];

    if(!file){
      return;
    }

    $("attachment").textContent =
      "Analyzing " +
      file.name +
      "...";

    $("attachment").classList.remove(
      "hidden"
    );

    try{
      const encoded =
        await toBase64(file);

      const r = await api(
        "/api/vision",
        {
          method:"POST",
          body:JSON.stringify({
            name:file.name,
            content_base64:encoded,
            prompt:$("input").value.trim()
          })
        }
      );

      const data =
        await r.json();

      if(data.error){
        throw new Error(
          data.error
        );
      }

      addMessage(
        "assistant",
        "Image understanding: " +
        data.answer
      );

      $("attachment").textContent =
        file.name +
        " · analyzed";
    }catch(error){
      addMessage(
        "assistant",
        "Vision unavailable: " +
        error.message
      );

      $("attachment").textContent =
        file.name +
        " · vision unavailable";
    }

    event.target.value = "";
  };

let recognition = null;

$("voiceIn").onclick = ()=>{
  const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

  if(!SpeechRecognition){
    alert(
      "Voice input is not supported by this browser."
    );
    return;
  }

  if(recognition){
    recognition.stop();
    return;
  }

  recognition =
    new SpeechRecognition();

  recognition.lang = "si-LK";
  recognition.interimResults =
    false;

  recognition.onstart = ()=>{
    $("voiceIn").textContent =
      "⏹ Stop";
  };

  recognition.onresult = event=>{
    $("input").value =
      event.results[0][0]
        .transcript;

    $("input").dispatchEvent(
      new Event("input")
    );
  };

  recognition.onerror = event=>{
    $("status").textContent =
      "Voice error: " +
      event.error;
  };

  recognition.onend = ()=>{
    recognition = null;

    $("voiceIn").textContent =
      "🎙 Voice";
  };

  recognition.start();
};

$("voiceOut").onclick = ()=>{
  if(!lastAnswer){
    alert(
      "There is no AI answer to read yet."
    );
    return;
  }

  if(!("speechSynthesis" in window)){
    alert(
      "Voice output is not supported."
    );
    return;
  }

  speechSynthesis.cancel();

  const utterance =
    new SpeechSynthesisUtterance(
      lastAnswer
    );

  utterance.lang =
    /[අ-෴]/.test(lastAnswer)
      ? "si-LK"
      : "en-US";

  speechSynthesis.speak(
    utterance
  );
};

let deferredPrompt = null;

window.addEventListener(
  "beforeinstallprompt",
  event=>{
    event.preventDefault();
    deferredPrompt = event;
  }
);

$("installApp").onclick =
  async ()=>{
    if(deferredPrompt){
      deferredPrompt.prompt();
      deferredPrompt = null;
    }else{
      alert(
        "Use your browser's Install App / Add to Home screen option."
      );
    }
  };

boot();
