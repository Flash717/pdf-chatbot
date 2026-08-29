/**
 * Embeddable PDF chatbot widget.
 *
 * Drop into any page with:
 *   <link rel="stylesheet" href="widget.css">
 *   <script src="widget.js" data-api-url="https://your-backend.example.com"></script>
 *
 * The widget renders a floating chat bubble that expands into a small
 * panel, posts questions to POST {apiUrl}/chat, and displays the answer
 * plus its source citations. No build step, no framework dependency.
 */
(function () {
  var scriptTag = document.currentScript;
  var API_URL = (scriptTag && scriptTag.getAttribute("data-api-url")) || "http://localhost:8000";

  function el(tag, className, text) {
    var e = document.createElement(tag);
    if (className) e.className = className;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function buildWidget() {
    var bubble = el("button", "pdfchat-bubble", "💬");
    bubble.setAttribute("aria-label", "Open chat");

    var panel = el("div", "pdfchat-panel");
    var header = el("div", "pdfchat-header", "Ask about our documents");
    var messages = el("div", "pdfchat-messages");
    var inputRow = el("div", "pdfchat-inputrow");
    var input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Ask a question...";
    var sendBtn = el("button", "", "Send");

    inputRow.appendChild(input);
    inputRow.appendChild(sendBtn);
    panel.appendChild(header);
    panel.appendChild(messages);
    panel.appendChild(inputRow);

    document.body.appendChild(bubble);
    document.body.appendChild(panel);

    bubble.addEventListener("click", function () {
      panel.classList.toggle("open");
      if (panel.classList.contains("open")) input.focus();
    });

    function addMessage(role, text) {
      var msg = el("div", "pdfchat-msg " + role, text);
      messages.appendChild(msg);
      messages.scrollTop = messages.scrollHeight;
      return msg;
    }

    function addSources(container, sources) {
      if (!sources || !sources.length) return;
      var seen = {};
      var parts = [];
      sources.forEach(function (s) {
        var key = s.source + "#" + s.page;
        if (seen[key]) return;
        seen[key] = true;
        parts.push((s.source || "unknown") + (s.page ? ", p." + s.page : ""));
      });
      var srcEl = el("div", "pdfchat-sources", "Sources: " + parts.join("; "));
      container.appendChild(srcEl);
    }

    function ask() {
      var question = input.value.trim();
      if (!question) return;
      input.value = "";
      sendBtn.disabled = true;
      addMessage("user", question);
      var thinking = addMessage("bot", "Thinking...");

      fetch(API_URL + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
      })
        .then(function (res) {
          if (!res.ok) throw new Error("Request failed: " + res.status);
          return res.json();
        })
        .then(function (data) {
          thinking.textContent = data.answer;
          addSources(thinking, data.sources);
        })
        .catch(function (err) {
          thinking.textContent = "Sorry, something went wrong talking to the backend (" + err.message + ").";
        })
        .finally(function () {
          sendBtn.disabled = false;
        });
    }

    sendBtn.addEventListener("click", ask);
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") ask();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", buildWidget);
  } else {
    buildWidget();
  }
})();
