import json, os
from flask import Flask, request, jsonify, send_from_directory

CONFIG_PATH = "./training_config.json"

def get_config():
    config =[]
    try:
        if not os.path.exists(get_config_path()):
            return jsonify({"status": "error", "message": "Config file not found"}), 404

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print("❌ Error reading config:", e)
    return config

def get_config_path() :
    return CONFIG_PATH
