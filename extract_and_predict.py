import re
import socket
import dns.resolver
import whois
from urllib.parse import urlparse
from datetime import datetime
from collections import Counter
from math import log2
import ipwhois
import numpy as np
import tensorflow as tf

VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


# ===== Feature extraction =====
def get_life_time(domain: str):
    try:
        w = whois.whois(domain)
        creation_date = (
            w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        )
        if creation_date and isinstance(creation_date, datetime):
            return (datetime.now() - creation_date).days
    except:
        pass
    return 0


def get_ns_similarity(domain: str):
    try:
        ns_records = dns.resolver.resolve(domain, "NS")
        ns_names = sorted([str(r.target).lower() for r in ns_records])
        if not ns_names:
            return 0.0
        base = ns_names[0]
        scores = [
            sum(1 for a, b in zip(base, ns) if a == b) / max(len(base), len(ns))
            for ns in ns_names[1:]
        ]
        return sum(scores) / len(scores) if scores else 0.0
    except:
        return 0.0


def get_n_countries(domain: str):
    try:
        ip = socket.gethostbyname(domain)
        obj = ipwhois.IPWhois(ip)
        res = obj.lookup_rdap(depth=1)
        countries = set()
        if "network" in res and "country" in res["network"]:
            countries.add(res["network"]["country"])
        for entity in res.get("objects", {}).values():
            contact = entity.get("contact", {})
            if "address" in contact:
                for addr in contact["address"]:
                    if "value" in addr and isinstance(addr["value"], str):
                        match = re.search(r"[A-Z]{2}", addr["value"])
                        if match:
                            countries.add(match.group(0))
        return len(countries)
    except:
        return 0


def normalize_domain(domain: str) -> str:
    # Loại bỏ http://, https://, www.
    domain = domain.strip().lower()
    if domain.startswith("http://"):
        domain = domain[7:]
    elif domain.startswith("https://"):
        domain = domain[8:]
    if domain.startswith("www."):
        domain = domain[4:]
    # Chỉ lấy phần domain chính (trước dấu '/')
    domain = domain.split("/")[0]
    return domain


def extract_features(domain: str):
    features = {}

    # Chuẩn hóa domain trước khi xử lý
    domain = normalize_domain(domain)

    parsed = urlparse(domain)
    netloc = parsed.netloc if parsed.netloc else parsed.path
    domain = netloc.lower()

    features["length"] = len(domain)

    try:
        answers = dns.resolver.resolve(domain, "NS")
        features["n_ns"] = len(answers)
    except:
        features["n_ns"] = 0

    features["n_vowels"] = sum(c in VOWELS for c in domain)
    features["n_vowel_chars"] = features["n_vowels"]
    features["n_constant_chars"] = len(re.findall(r"[a-z]", domain))
    features["n_num"] = sum(c.isdigit() for c in domain)
    features["n_other_chars"] = sum(not c.isalnum() for c in domain)

    probs = [c / len(domain) for c in Counter(domain).values()]
    features["entropy"] = -sum(p * log2(p) for p in probs if p > 0)

    features["ns_similarity"] = get_ns_similarity(domain)
    features["n_countries"] = get_n_countries(domain)

    try:
        answers = dns.resolver.resolve(domain, "MX")
        features["n_mx"] = len(answers)
    except:
        features["n_mx"] = 0

    features["n_labels"] = len(domain.split("."))
    features["life_time"] = get_life_time(domain)

    # Đảm bảo đúng thứ tự và đủ 13 đặc trưng
    feature_list = [
        features["length"],
        features["n_ns"],
        features["n_vowels"],
        features["n_vowel_chars"],
        features["n_constant_chars"],
        features["n_num"],
        features["n_other_chars"],
        features["entropy"],
        features["ns_similarity"],
        features["n_countries"],
        features["n_mx"],
        features["n_labels"],
        features["life_time"],
    ]
    print(f"Extracted features for {domain}: {feature_list}")
    return feature_list


# ===== Prediction using TFLite model =====
def predict_domain(domain):
    features = extract_features(domain)
    interpreter = tf.lite.Interpreter(
        model_path="madonna_model_pruned_quantized.tflite"
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zp = input_details[0]["quantization"]
    inp = np.expand_dims(np.array(features, dtype=np.float32), axis=0)
    inp_quant = (inp / input_scale + input_zp).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], inp_quant)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]["index"])[0][0]
    out_scale, out_zp = output_details[0]["quantization"]
    prob = (raw_output - out_zp) * out_scale
    label = "Malicious" if prob > 0.5 else "Benign"

    return {"domain": domain, "probability": float(prob) * 100, "label": label}


if __name__ == "__main__":
    test_domain = "google.com"
    result = predict_domain(test_domain)
    print(result)
