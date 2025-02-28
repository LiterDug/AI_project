import sqlite3


# Kết nối hoặc tạo mới database
conn = sqlite3.connect("database/farm_inventory.db")
cursor = conn.cursor()
print(f"Đã tạo database thành công")


# 1. Bảng vật tư
cursor.execute("""
CREATE TABLE IF NOT EXISTS materials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT,
    unit TEXT NOT NULL,  -- Đơn vị tính (kg, lít, cái,...)
    description TEXT
)
""")

# 2. Bảng nhà cung cấp
cursor.execute("""
CREATE TABLE IF NOT EXISTS suppliers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    contact TEXT,
    address TEXT
)
""")

# 3. Bảng tồn kho
cursor.execute("""
CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id INTEGER,
    quantity REAL NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (material_id) REFERENCES materials(id)
)
""")

# 4. Bảng giao dịch nhập/xuất kho
cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id INTEGER,
    supplier_id INTEGER,
    quantity REAL NOT NULL,
    transaction_type TEXT CHECK(transaction_type IN ('import', 'export')),
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (material_id) REFERENCES materials(id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id)
)
""")

# Thêm vật tư mẫu
cursor.execute("INSERT INTO materials (name, category, unit, description) VALUES ('Phân bón NPK', 'Phân bón', 'kg', 'Dùng để bón cây')")
cursor.execute("INSERT INTO materials (name, category, unit, description) VALUES ('Thuốc trừ sâu', 'Bảo vệ thực vật', 'lít', 'Dùng để diệt sâu bệnh')")

# Thêm nhà cung cấp
cursor.execute("INSERT INTO suppliers (name, contact, address) VALUES ('Công ty A', '0123456789', 'Hà Nội')")
cursor.execute("INSERT INTO suppliers (name, contact, address) VALUES ('Công ty B', '0987654321', 'TP Hồ Chí Minh')")

# Thêm tồn kho
cursor.execute("INSERT INTO inventory (material_id, quantity) VALUES (1, 100)")  # 100 kg phân bón
cursor.execute("INSERT INTO inventory (material_id, quantity) VALUES (2, 50)")   # 50 lít thuốc trừ sâu

# Thêm giao dịch nhập kho
cursor.execute("INSERT INTO transactions (material_id, supplier_id, quantity, transaction_type) VALUES (1, 1, 50, 'import')")
cursor.execute("INSERT INTO transactions (material_id, supplier_id, quantity, transaction_type) VALUES (2, 2, 30, 'import')")

# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()

print("Database farm_inventory.db đã được tạo thành công và dữ liệu mẫu đã được thêm vào!")