package com.chatbot.ai.model;

/**
 * Lớp DTO truyền tải thông tin giữa các AI Agent.
 */
public class AgentPayload {
    private String sessionId;
    private String userQuery;
    private double confidenceScore;

    public AgentPayload(String sessionId, String userQuery) {
        this.sessionId = sessionId;
        this.userQuery = userQuery;
        this.confidenceScore = 1.0;
    }

    public String getSessionId() { return sessionId; }
    public String getUserQuery() { return userQuery; }
    public double getConfidenceScore() { return confidenceScore; }
}
