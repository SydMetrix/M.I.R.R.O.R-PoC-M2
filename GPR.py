import json
import pickle
import faiss
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from datetime import datetime
import os

# ====== Load SpaCy NLP model ======
print("Loading SpaCy model...")
nlp = spacy.load("en_core_web_sm")  # hoặc mô hình tiếng Việt nếu cần

# ====== Load Transformer model ======
print("Loading Transformer model for embeddings...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# ====== Load FAISS index và thư viện vector ======
print("Loading FAISS index and vector library...")
index = faiss.read_index("concept_index.faiss")
with open("concept_metadata.pkl", "rb") as f:
    glitch_lib = pickle.load(f)

# ====== Hàm tạo embedding câu ======
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
    return embeddings[0].cpu().numpy()

# ====== Hàm Semantic Role Parsing đơn giản ======
def extract_roles(text):
    doc = nlp(text)
    roles = {"Agent": None, "Action": None, "Patient": None, "Location": None, "Manner": None}
    for token in doc:
        if token.dep_ == "nsubj":
            roles["Agent"] = token.text
        if token.dep_ == "ROOT":
            roles["Action"] = token.text
        if token.dep_ == "dobj":
            roles["Patient"] = token.text
        if token.dep_ == "prep":
            roles["Location"] = token.head.text + " " + token.text
    return roles

# ====== Hàm tính Component Semantic Similarity ======
def compare_roles(roles1, roles2):
    scores = []
    for key in roles1:
        if roles1[key] and roles2[key]:
            vec1 = embed_text(roles1[key])
            vec2 = embed_text(roles2[key])
            sim = cosine_similarity(vec1, vec2)
            scores.append(sim)
    if scores:
        return sum(scores) / len(scores), len(scores) / len(roles1)
    return 0.0, 0.0

# ====== Hàm cosine similarity ======
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ====== Hàm chuyển đổi numpy types sang Python native types ======
def convert_numpy_types(obj):
    """Chuyển đổi các kiểu dữ liệu numpy sang kiểu Python thuần túy để có thể serialize JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):  # Xử lý numpy boolean
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# ====== Hàm debug JSON file ======
def debug_json_file(filepath: str):
    """Debug file JSON để tìm lỗi"""
    if not os.path.exists(filepath):
        print(f"File {filepath} không tồn tại.")
        return
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"📄 File size: {len(content)} characters")
            print(f"📝 First 200 chars: {content[:200]}")
            print(f"📝 Last 200 chars: {content[-200:]}")
            
            # Thử parse JSON
            f.seek(0)
            data = json.load(f)
            print(f"✅ JSON is valid! Type: {type(data)}")
            if isinstance(data, dict):
                print(f"🔑 Keys: {list(data.keys())}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error: {e}")
        print(f"📍 Error at line {e.lineno}, column {e.colno}")
        # Hiển thị context xung quanh lỗi
        lines = content.split('\n')
        if e.lineno <= len(lines):
            print(f"🔍 Problem line: {lines[e.lineno-1]}")
            if e.colno > 0:
                print(f"🔍 Problem char: '{lines[e.lineno-1][e.colno-1:e.colno+10]}'")
    except Exception as e:
        print(f"❌ File Error: {e}")

# ====== Hàm save_to_json hoàn chỉnh ======
def save_to_json(filepath: str, new_entry: dict):
    """
    Lưu entry mới vào file JSON. Nếu file đã tồn tại, load và append.
    Nếu file không tồn tại hoặc bị lỗi, tạo mới từ đầu.
    
    Args:
        filepath (str): Đường dẫn đến file JSON
        new_entry (dict): Entry mới cần thêm vào
    """
    # Tạo cấu trúc database mặc định
    default_db = {
        "metadata": {
            "last_updated": None,
            "total_entries": 0,
            "version": "M2_GPR_v2"
        },
        "entries": []
    }
    
    # Thử load file hiện tại
    db = default_db.copy()
    if os.path.exists(filepath):
        print(f"📁 Found existing file: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and "entries" in loaded and "metadata" in loaded:
                    db = loaded
                    print(f"✅ Loaded existing database with {len(db['entries'])} entries.")
                else:
                    print("⚠️ Invalid format. Starting new database.")
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print("🔍 Debugging JSON file...")
            debug_json_file(filepath)
            # Tạo backup và tạo file mới
            backup_path = filepath + ".backup"
            if os.path.exists(filepath):
                os.rename(filepath, backup_path)
                print(f"📄 Corrupted file backed up to: {backup_path}")
            print("🆕 Creating new database...")
        except Exception as e:
            print(f"⚠️ Warning: Could not load existing JSON. Reason: {e}")
            print("Proceeding with a new empty database.")
    else:
        print(f"🆕 Creating new database file: {filepath}")
    
    # Tạo entry_id mới
    new_entry["entry_id"] = len(db["entries"]) + 1
    new_entry["timestamp"] = datetime.now().isoformat()
    
    # Chuyển đổi tất cả numpy types sang Python native types
    new_entry = convert_numpy_types(new_entry)
    
    # Thêm entry mới
    db["entries"].append(new_entry)
    
    # Cập nhật metadata
    db["metadata"]["last_updated"] = datetime.now().isoformat()
    db["metadata"]["total_entries"] = len(db["entries"])
    
    # Chuyển đổi toàn bộ database để đảm bảo JSON serializable
    db = convert_numpy_types(db)
    
    # Ghi vào file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=4, ensure_ascii=False)
        print(f"✅ Entry saved successfully to {filepath}")
        print(f"📊 Total entries: {db['metadata']['total_entries']}")
    except Exception as e:
        print(f"❌ Error saving to JSON: {e}")
        print("Data types in new_entry:")
        for key, value in new_entry.get("results", {}).items():
            print(f"  {key}: {type(value)} = {value}")
        raise

# ====== Nhận input ======
print("\nEnter Prompt:")
prompt = input()
print("\nEnter Response:")
response = input()

# ====== Semantic Embedding ======
print("Embedding prompt and response...")
prompt_vec = embed_text(prompt)
response_vec = embed_text(response)

# ====== Tính toán Semantic Similarity ======
semantic_score = cosine_similarity(prompt_vec, response_vec)

# ====== Semantic Role Matching ======
prompt_roles = extract_roles(prompt)
response_roles = extract_roles(response)
component_score, role_match_ratio = compare_roles(prompt_roles, response_roles)
aggregated_srl_score = round(component_score * role_match_ratio, 3)

# ====== Glitch Signature Logic ======
glitch_types = []
if semantic_score < 0.75:
    glitch_types.append("SemanticSubLoop")
if aggregated_srl_score < 0.4:
    glitch_types.append("SkewedMirror")
if semantic_score > 0.7 and aggregated_srl_score > 0.3:
    glitch_types.append("SelfAffirmingTrap") 

glitch_signature = "GTP::" + "+".join(glitch_types) if glitch_types else "GTP::None"
csi_score = round(1.0 - ((semantic_score + aggregated_srl_score) / 2) * 0.5, 3)
# ====== CSI Reflection Labeling ======
if csi_score >= 0.8:
    csi_label = "🔴 High Divergence"
elif csi_score >= 0.6:
    csi_label = "🟠 Medium Divergence"
elif csi_score >= 0.4:
    csi_label = "🟡 Mild Divergence"
else:
    csi_label = "🟢 Stable"
    # Optional print to console
print(f"CSI Score: {csi_score} → {csi_label}")
# ====== Route Decision Logic ======
route_sahl = csi_score < 0.5
route_arp_x = len(glitch_types) >= 2

# ====== In kết quả phân tích ra màn hình ======
print("\n==== ANALYSIS RESULT ====")
print(f"Glitch Signature: {glitch_signature}")
print(f"Glitch Types: {glitch_types}")
print(f"CSI Score: {csi_score:.3f}")
print(f"Aggregated SRL Score: {aggregated_srl_score:.3f}")
print(f"Route SAHL: {route_sahl}")
print(f"Route ARP-X: {route_arp_x}")

# ====== Hỏi người dùng có muốn lưu không ======
save = input("\nSave to database? (y/n): ").strip().lower()
if save != 'y':
    print("\nResult discarded.")
    exit()

# ====== Tạo kết quả mới ======
new_entry = {
    "input": {
        "prompt": prompt,
        "response": response
    },
    "results": {
        "glitch_signature": glitch_signature,
        "glitch_types": glitch_types,
        "csi_score": csi_score,
        "aggregated_srl_score": aggregated_srl_score,
        "route_sahl": route_sahl,
        "route_arp_x": route_arp_x
    }
}

# ====== Sử dụng hàm save_to_json ======
json_path = "M2_results.json"
save_to_json(json_path, new_entry)