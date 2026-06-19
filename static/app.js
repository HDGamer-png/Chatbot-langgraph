const messagesContainer = document.getElementById("messages");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const attachBtn = document.getElementById("attachBtn");
const fileInput = document.getElementById("fileInput");
const attachmentList = document.getElementById("attachmentList");
const newSessionBtn = document.getElementById("newSessionBtn");
const loadSessionBtn = document.getElementById("loadSessionBtn");
const refreshSessionsBtn = document.getElementById("refreshSessionsBtn");
const providerSelect = document.getElementById("providerSelect");
const sessionSelect = document.getElementById("sessionSelect");
const providerStatus = document.getElementById("providerStatus");
const modelStatus = document.getElementById("modelStatus");
const sessionStatus = document.getElementById("sessionStatus");
const userStatus = document.getElementById("userStatus");
const userIdInput = document.getElementById("userIdInput");
const userNameInput = document.getElementById("userNameInput");
const userEmailInput = document.getElementById("userEmailInput");
const saveUserBtn = document.getElementById("saveUserBtn");
const openProcessBtn = document.getElementById("openProcessBtn");
const clearBtn = document.getElementById("clearBtn");

let queuedAttachments = [];

let sessionId = null;
let provider = null;
let availableProviders = [];
let modelFast = null;
let modelMain = null;
let isLoading = false;
let currentUserId = "";
let currentUserName = "";
let currentUserEmail = "";

function updateStatus() {
    providerStatus.textContent = `Provider: ${provider || "không xác định"}`;
    modelStatus.textContent = `Model: ${modelMain || "n/a"}`;
    sessionStatus.textContent = `Phiên: ${sessionId || "chưa có"}`;
    userStatus.textContent = `Người dùng: ${currentUserId || "chưa có"}${currentUserName ? ` (${currentUserName})` : ""}`;
}

async function initialize() {
    loadStoredUser();
    if (currentUserId) {
        await fetchUserProfile(currentUserId);
    }
    await fetchHealth();
    await fetchSessions();
    await createNewSession();
}

// Auto-scroll to bottom
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Display user message
function displayUserMessage(text, attachments = []) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message user";
    msgDiv.innerHTML = `
        <div>
            <div class="message-content">${escapeHtml(text)}</div>
            ${renderAttachmentsHtml(attachments)}
            <div class="metadata">${new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}</div>
        </div>
    `;
    messagesContainer.appendChild(msgDiv);
    // Post-render safety: remove any <strong> or <b> elements that might still exist
    try {
        msgDiv.querySelectorAll('strong, b').forEach(el => {
            const txt = document.createTextNode(el.textContent || '');
            el.parentNode.replaceChild(txt, el);
        });
    } catch (e) {
        // ignore if operation fails
    }
    scrollToBottom();
}

function renderAttachmentsHtml(attachments) {
    if (!attachments || !attachments.length) {
        return "";
    }
    return `
        <div class="attachment-panel">
            ${attachments
                .map((item) => {
                    const safeName = escapeHtml(item.filename || item.url || "file");
                    const safeUrl = escapeHtml(item.url || "#");
                    const isImage = item.type?.startsWith("image/");
                    return `<div class="attachment-pill">
                        ${isImage ? `<img src="${safeUrl}" alt="${safeName}" style="max-width:36px; max-height:36px; border-radius:8px;" />` : "📎"}
                        <a href="${safeUrl}" target="_blank" rel="noreferrer noopener">${safeName}</a>
                    </div>`;
                })
                .join("")}
        </div>
    `;
}

// Display loading indicator
function displayLoading() {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message bot";
    msgDiv.id = "loading-msg";
    msgDiv.innerHTML = `
        <div>
            <div class="message-content loading">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
}

// Display bot reply
function displayBotMessage(text, intent = "", elapsed = 0) {
    const loadingMsg = document.getElementById("loading-msg");
    if (loadingMsg) {
        loadingMsg.remove();
    }

    const msgDiv = document.createElement("div");
    msgDiv.className = "message bot";

    // Render with controlled bolding using marker system:
    // - LLM should mark important text with `!!important: ...`
    // - We convert those markers to <span class="important">...</span>
    // - We still strip raw <strong>/<b> and **bold** elsewhere
    let raw = (text || "")
        // remove HTML <strong> or <b> tags if present
        .replace(/<\/?strong[^>]*>/gi, '')
        .replace(/<\/?b[^>]*>/gi, '');
    // remove escaped HTML bold tags like &lt;strong&gt; or &lt;b&gt;
    raw = raw.replace(/&lt;\/?strong[^&]*&gt;/gi, '').replace(/&lt;\/?b[^&]*&gt;/gi, '');

    // Extract next-action hint lines if present and remove them from the main body
    let hintText = "";
    const hintRegex = /(?:^|\n)(?:gợi ý hành động tiếp theo|gợi ý)[:：]\s*([\s\S]*)$/i;
    const hintMatch = raw.match(hintRegex);
    if (hintMatch) {
        hintText = hintMatch[1].trim();
        raw = raw.slice(0, hintMatch.index).trim();
    }

    // Extract !!important: markers and replace with placeholders to preserve content
    const impMap = [];
    let impIdx = 0;
    raw = raw.replace(/!!important:\s*(.*?)(?=$|\n)/g, function(_, m) {
        const key = `@@IMP${impIdx}@@`;
        impMap.push({ key, text: m.trim() });
        impIdx += 1;
        return key;
    });

    const escaped = escapeHtml(raw || "");
    const lines = escaped.split('\n');
    let renderedParts = [];
    let openList = null;

    function closeList() {
        if (openList === 'ul') renderedParts.push('</ul>');
        else if (openList === 'ol') renderedParts.push('</ol>');
        openList = null;
    }

    function formatInline(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '$1')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }

    for (let i = 0; i < lines.length; i++) {
        const rawLine = lines[i];
        const line = rawLine.trim();
        const nextLine = lines[i + 1] ? lines[i + 1].trim() : '';

        if (!line) {
            closeList();
            renderedParts.push('<br>');
            continue;
        }

        const hMatch = line.match(/^(#{1,6})\s*(.*)$/);
        if (hMatch) {
            closeList();
            const level = Math.min(hMatch[1].length, 3);
            const content = formatInline(hMatch[2]);
            if (level === 1) renderedParts.push(`<h2>${content}</h2>`);
            else if (level === 2) renderedParts.push(`<h3>${content}</h3>`);
            else renderedParts.push(`<h4>${content}</h4>`);
            continue;
        }

        const tableSeparator = /^\s*\|?\s*(:?-+:?)\s*(\|\s*(:?-+:?)\s*)+\|?\s*$/;
        if (line.includes('|') && tableSeparator.test(nextLine)) {
            closeList();
            const headerCells = line.replace(/^\||\|$/g, '').split('|').map(c => formatInline(c.trim()));
            const alignments = nextLine.replace(/^\||\|$/g, '').split('|').map(cell => {
                const t = cell.trim();
                if (/^:-+:$/.test(t)) return 'center';
                if (/^-+:$/.test(t)) return 'right';
                return 'left';
            });
            renderedParts.push('<table>');
            renderedParts.push('<thead><tr>' + headerCells.map((cell, idx) => `<th style="text-align:${alignments[idx]||'left'}">${cell}</th>`).join('') + '</tr></thead>');
            renderedParts.push('<tbody>');
            i += 1;
            while (i + 1 < lines.length && lines[i + 1].trim().includes('|')) {
                i += 1;
                const row = lines[i].replace(/^\||\|$/g, '').split('|').map(c => formatInline(c.trim()));
                renderedParts.push('<tr>' + row.map((cell, idx) => `<td style="text-align:${alignments[idx]||'left'}">${cell}</td>`).join('') + '</tr>');
            }
            renderedParts.push('</tbody></table>');
            continue;
        }

        const olMatch = line.match(/^\d+\.\s+(.*)$/);
        if (olMatch) {
            const content = formatInline(olMatch[1]);
            if (openList !== 'ol') {
                closeList();
                renderedParts.push('<ol>');
                openList = 'ol';
            }
            renderedParts.push(`<li>${content}</li>`);
            continue;
        }

        const liMatch = line.match(/^[-\*]\s+(.*)$/);
        if (liMatch) {
            const content = formatInline(liMatch[1]);
            if (openList !== 'ul') {
                closeList();
                renderedParts.push('<ul>');
                openList = 'ul';
            }
            renderedParts.push(`<li>${content}</li>`);
            continue;
        }

        const paragraph = formatInline(line);
        closeList();
        renderedParts.push(`<p>${paragraph}</p>`);
    }

    closeList();
    let renderedContent = renderedParts.join('\n');
    // Final safety: remove any remaining <strong> or <b> elements while preserving their text
    try {
        const tmpDiv = document.createElement('div');
        tmpDiv.innerHTML = renderedContent;
        tmpDiv.querySelectorAll('strong, b').forEach(el => {
            const txt = document.createTextNode(el.textContent || '');
            el.parentNode.replaceChild(txt, el);
        });
        const paragraphs = tmpDiv.querySelectorAll('p');
        if (paragraphs.length > 1) {
            paragraphs[0].classList.add('summary');
        }
        renderedContent = tmpDiv.innerHTML;
    } catch (e) {
        // fallback to regex removal if DOM ops fail
        renderedContent = renderedContent.replace(/<\/?strong[^>]*>/gi, '').replace(/<\/?b[^>]*>/gi, '');
    }

    // Replace importance placeholders with styled spans (escaped inner text)
    for (const item of impMap) {
        const safe = escapeHtml(item.text || '');
        renderedContent = renderedContent.split(item.key).join(`<span class="important">${safe}</span>`);
    }

    msgDiv.innerHTML = `
        <div>
            <div class="message-content markdown-content">${renderedContent}</div>
            ${hintText ? `<div class="hint-card"><div class="hint-title">Gợi ý hành động tiếp theo</div><div class="hint-text">${escapeHtml(hintText).replace(/\n/g, '<br>')}</div></div>` : ''}
            <div class="metadata">
                ${intent ? `Intent: ${escapeHtml(intent)}` : ''} 
                ${elapsed ? `⏱ ${elapsed.toFixed(2)}s` : ''}
            </div>
        </div>
    `;
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
}


// Display error message
function displayError(errorText, hintText = null) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "error-message";
    msgDiv.innerHTML = `❌ Lỗi: ${escapeHtml(errorText)}`;
    if (hintText) {
        const hintDiv = document.createElement("div");
        hintDiv.className = "error-hint";
        hintDiv.textContent = `Gợi ý: ${hintText}`;
        msgDiv.appendChild(hintDiv);
    }
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function clearMessages(showWelcome = true) {
    messagesContainer.innerHTML = "";
    if (showWelcome) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message bot";
        msgDiv.innerHTML = `
            <div>
                <div class="message-content">👋 Xin chào! Tôi là trợ lý AI. Bắt đầu chat bằng cách gửi câu hỏi.</div>
                <div class="metadata">Sẵn sàng</div>
            </div>
        `;
        messagesContainer.appendChild(msgDiv);
    }
    scrollToBottom();
}

async function fetchHealth() {
    try {
        const response = await fetch("/api/health");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data.status !== "ok") throw new Error(data.message || "Không thể lấy trạng thái.");

        provider = data.provider || "unknown";
        availableProviders = data.available_providers || [];
        modelFast = data.model_fast || null;
        modelMain = data.model_main || null;
        updateProviderSelect();
        updateStatus();
    } catch (error) {
        console.error("Health error:", error);
        providerStatus.textContent = "Provider: lỗi";
        modelStatus.textContent = "Model: lỗi";
    }
}

function updateProviderSelect() {
    providerSelect.innerHTML = "";
    if (!availableProviders.length) {
        const opt = document.createElement("option");
        opt.textContent = "Không có provider";
        opt.disabled = true;
        providerSelect.appendChild(opt);
        providerSelect.disabled = true;
        return;
    }

    availableProviders.forEach((item) => {
        const opt = document.createElement("option");
        opt.value = item;
        opt.textContent = item === provider ? `${item} (đang dùng)` : item;
        opt.selected = item === provider;
        providerSelect.appendChild(opt);
    });
    providerSelect.disabled = false;
}

async function fetchSessions() {
    try {
        const response = await fetch("/api/sessions");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        sessionSelect.innerHTML = "";

        const defaultOption = document.createElement("option");
        defaultOption.value = "";
        defaultOption.textContent = "Chọn phiên hiện có";
        defaultOption.disabled = true;
        defaultOption.selected = true;
        sessionSelect.appendChild(defaultOption);

        (data.sessions || []).forEach((item) => {
            const opt = document.createElement("option");
            opt.value = item.session_id;
            opt.textContent = item.session_id;
            sessionSelect.appendChild(opt);
        });
    } catch (error) {
        console.error("Sessions error:", error);
    }
}

async function createNewSession() {
    try {
        const response = await fetch("/api/session/new", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: currentUserId || null }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        sessionId = data.session_id;
        updateStatus();
        clearMessages();
        await fetchSessions();
    } catch (error) {
        console.error("New session error:", error);
        displayError(error.message);
    }
}

async function loadSession(sessionToLoad) {
    if (!sessionToLoad) return;
    try {
        const response = await fetch(`/api/history/${sessionToLoad}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        sessionId = data.session_id;
        currentUserId = data.user_id || "";
        userIdInput.value = currentUserId;
        if (currentUserId) {
            await fetchUserProfile(currentUserId);
        }
        updateStatus();
        queuedAttachments = [];
        renderAttachmentQueue();
        clearMessages(false);
        renderHistory(data.turns || []);
    } catch (error) {
        console.error("Load session error:", error);
        displayError(error.message);
    }
}

async function fetchUserProfile(userId) {
    if (!userId) return;
    try {
        const response = await fetch(`/api/users/${encodeURIComponent(userId)}`);
        if (!response.ok) {
            return;
        }
        const data = await response.json();
        if (data.error) {
            return;
        }

        currentUserName = (data.profile && data.profile.name) || "";
        currentUserEmail = (data.profile && data.profile.email) || "";
        userNameInput.value = currentUserName;
        userEmailInput.value = currentUserEmail;
        localStorage.setItem("chatbot_user_id", currentUserId);
        localStorage.setItem("chatbot_user_name", currentUserName);
        localStorage.setItem("chatbot_user_email", currentUserEmail);
        updateStatus();
    } catch (error) {
        console.warn("Không thể lấy profile user:", error);
    }
}

function renderHistory(turns) {
    if (!turns.length) {
        const info = document.createElement("div");
        info.className = "message bot";
        info.innerHTML = `
            <div>
                <div class="message-content">Phiên này chưa có lịch sử hoặc chưa có lượt chat nào.</div>
            </div>
        `;
        messagesContainer.appendChild(info);
        scrollToBottom();
        return;
    }

    turns.forEach((turn) => {
        const userText = turn.user_query || "";
        const botText = turn.final_answer || "";
        if (userText) displayUserMessage(userText, turn.attachments || []);
        if (botText) displayBotMessage(botText, "", 0);
    });
}

async function sendMessage() {
    const message = messageInput.value.trim();
    const hasAttachments = queuedAttachments.length > 0;

    if (!message && !hasAttachments) return;
    if (isLoading) return;
    if (!sessionId) {
        await createNewSession();
    }

    displayUserMessage(message || "(đã đính kèm file)", queuedAttachments);
    messageInput.value = "";
    isLoading = true;
    sendBtn.disabled = true;
    displayLoading();

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                message,
                attachments: queuedAttachments,
                session_id: sessionId,
                provider: providerSelect.value || provider,
                user_id: currentUserId || null,
            }),
        });

        const data = await response.json();
        if (!response.ok) {
            const errorMessage = data.error || `HTTP ${response.status}`;
            const hintText = data.hint || null;
            displayError(errorMessage, hintText);
            return;
        }
        if (data.error) {
            const hintText = data.hint || null;
            throw new Error(`${data.error}${hintText ? ` — ${hintText}` : ''}`);
        }

        provider = data.provider || provider;
        updateStatus();
        queuedAttachments = [];
        renderAttachmentQueue();
        displayBotMessage(data.reply || "Không nhận được phản hồi", data.intent, data.elapsed || 0);
    } catch (error) {
        console.error("Error:", error);
        displayError(error.message);
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

function renderAttachmentQueue() {
    attachmentList.innerHTML = "";
    if (!queuedAttachments.length) {
        return;
    }
    queuedAttachments.forEach((item, index) => {
        const pill = document.createElement("div");
        pill.className = "attachment-pill";
        const safeName = escapeHtml(item.filename || item.url || "file");
        const isImage = item.type?.startsWith("image/");
        pill.innerHTML = `
            ${isImage ? `<img src="${escapeHtml(item.url)}" alt="${safeName}" style="max-width:24px; max-height:24px; border-radius:6px;" />` : "📎"}
            <a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer noopener">${safeName}</a>
            <button type="button" aria-label="Xoá" data-index="${index}">×</button>
        `;
        pill.querySelector("button").addEventListener("click", () => {
            queuedAttachments.splice(index, 1);
            renderAttachmentQueue();
        });
        attachmentList.appendChild(pill);
    });
}

async function uploadFile(file) {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/api/upload", {
        method: "POST",
        body: form,
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}`);
    }
    return response.json();
}

async function addAttachment(file) {
    try {
        const uploaded = await uploadFile(file);
        queuedAttachments.push(uploaded);
        renderAttachmentQueue();
    } catch (error) {
        console.error("Upload error:", error);
        displayError(`Không thể đính kèm tệp: ${error.message}`);
    }
}

function handlePasteAttachment(event) {
    if (!event.clipboardData) return;
    const items = Array.from(event.clipboardData.items || []);
    const imageItem = items.find((item) => item.kind === "file" && item.type.startsWith("image/"));
    if (!imageItem) return;
    const file = imageItem.getAsFile();
    if (!file) return;
    event.preventDefault();
    addAttachment(file);
}

function handleFileInputChange(event) {
    const files = Array.from(event.target.files || []);
    files.forEach(addAttachment);
    fileInput.value = "";
}

function handleFileDrop(event) {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files || []);
    files.forEach(addAttachment);
}

async function saveUserProfile() {
    const userId = userIdInput.value.trim();
    if (!userId) {
        displayError("Vui lòng nhập User ID trước khi lưu.");
        return;
    }
    const payload = {
        name: userNameInput.value.trim(),
        email: userEmailInput.value.trim(),
    };

    try {
        const response = await fetch(`/api/users/${encodeURIComponent(userId)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP ${response.status}`);
        }
        const data = await response.json();
        currentUserId = data.user_id;
        currentUserName = data.profile.name || "";
        currentUserEmail = data.profile.email || "";
        localStorage.setItem("chatbot_user_id", currentUserId);
        localStorage.setItem("chatbot_user_name", currentUserName);
        localStorage.setItem("chatbot_user_email", currentUserEmail);
        updateStatus();
        displayBotMessage(`User profile saved: ${currentUserId}`, "profile", 0);
    } catch (error) {
        console.error("Save user error:", error);
        displayError(error.message);
    }
}

function loadStoredUser() {
    currentUserId = localStorage.getItem("chatbot_user_id") || "";
    currentUserName = localStorage.getItem("chatbot_user_name") || "";
    currentUserEmail = localStorage.getItem("chatbot_user_email") || "";
    userIdInput.value = currentUserId;
    userNameInput.value = currentUserName;
    userEmailInput.value = currentUserEmail;
    updateStatus();
}

function openProcessPage() {
    if (!sessionId) {
        displayError("Vui lòng tạo hoặc chọn phiên trước.");
        return;
    }
    window.open(`/process/${encodeURIComponent(sessionId)}`, "_blank");
}

// Event listeners
sendBtn.addEventListener("click", sendMessage);
attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", handleFileInputChange);
messageInput.addEventListener("paste", handlePasteAttachment);
messageInput.addEventListener("dragover", (e) => e.preventDefault());
messageInput.addEventListener("drop", handleFileDrop);
messageInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Initialize
newSessionBtn.addEventListener("click", createNewSession);
refreshSessionsBtn.addEventListener("click", fetchSessions);
loadSessionBtn.addEventListener("click", () => {
    const selected = sessionSelect.value;
    if (selected) loadSession(selected);
});
saveUserBtn.addEventListener("click", saveUserProfile);
openProcessBtn.addEventListener("click", openProcessPage);
providerSelect.addEventListener("change", (event) => {
    provider = event.target.value;
    updateStatus();
});
clearBtn.addEventListener("click", () => clearMessages());

initialize();
messageInput.focus();