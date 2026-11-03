-- Bảng quản lý sinh viên thực tập
CREATE TABLE IF NOT EXISTS internship_students (
    student_id VARCHAR(20) PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    topic_name VARCHAR(255) NOT NULL,
    supervisor VARCHAR(100),
    status VARCHAR(20) DEFAULT 'IN_PROGRESS',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO internship_students (student_id, full_name, topic_name, supervisor, status)
VALUES ('SV2026_01', 'Hoàng Minh Đức', 'Nghiên cứu kiến trúc Multi-Agent LangGraph', 'TS. Nguyễn Văn A', 'COMPLETED');
