/**
 * Thư viện kết nối API Máy chủ AI Chatbot v4.0
 */
class ChatbotClient {
    constructor(baseUrl = "http://127.0.0.1:5000") {
        this.baseUrl = baseUrl;
    }

    async sendQuery(message, attachments = []) {
        const response = await fetch(`${this.baseUrl}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, attachments })
        });
        return await response.json();
    }
}
