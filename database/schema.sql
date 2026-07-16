-- DocuMind database schema

CREATE DATABASE IF NOT EXISTS documind;
USE documind;

CREATE TABLE IF NOT EXISTS invoices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    raw_text LONGTEXT,
    status VARCHAR(50) DEFAULT 'uploaded'
);

CREATE TABLE IF NOT EXISTS extracted_fields (
    id INT AUTO_INCREMENT PRIMARY KEY,
    invoice_id INT NOT NULL,
    method VARCHAR(50) NOT NULL,        -- 'rule_based', 'classical_ml', 'llm_prompting', 'fine_tuned'
    vendor_name VARCHAR(255),
    invoice_number VARCHAR(100),
    invoice_date DATE,
    total_amount DECIMAL(12,2),
    tax_amount DECIMAL(12,2),
    confidence_score FLOAT,
    latency_ms INT,
    FOREIGN KEY (invoice_id) REFERENCES invoices(id)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    invoice_id INT NOT NULL,
    decision VARCHAR(50),               -- 'auto_approved', 'flagged'
    reason VARCHAR(255),
    decided_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (invoice_id) REFERENCES invoices(id)
);

CREATE TABLE IF NOT EXISTS human_corrections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    invoice_id INT NOT NULL,
    field_name VARCHAR(100),
    original_value VARCHAR(255),
    corrected_value VARCHAR(255),
    corrected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (invoice_id) REFERENCES invoices(id)
);
