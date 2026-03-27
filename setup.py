"""
Quick setup script for AIVA AI Virtual Assistant Backend
"""
import os
import sys
import subprocess


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def run_command(command, description):
    print(f"\n🔄 {description}...")
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def main():
    print_header("AIVA Backend — Setup")

    # Python version
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print("❌ Python 3.10+ required")
        return False
    print(f"✅ Python {v.major}.{v.minor}.{v.micro}")

    # Check project structure
    print_header("Checking Project Structure")
    required = [
        ("requirements.txt",           "Requirements"),
        ("main.py",                     "FastAPI app"),
        ("rag_faiss/build_index.py",    "Index builder"),
        ("rag_faiss/retriever.py",      "FAISS retriever"),
        ("rag_faiss/config.py",         "RAG config"),
        ("agent/groq_llama_agent.py",   "LLM agent"),
        ("audio/stt.py",               "STT processor"),
        ("audio/tts.py",               "TTS processor"),
    ]
    ok = True
    for path, desc in required:
        if os.path.exists(path):
            print(f"  ✅ {desc}")
        else:
            print(f"  ❌ {desc} missing: {path}")
            ok = False
    if not ok:
        return False

    # Install deps
    print_header("Installing Dependencies")
    if not run_command("pip install -r requirements.txt", "pip install"):
        return False

    # .env check
    print_header("Environment Configuration")
    if not os.path.exists(".env"):
        print("⚠️  No .env file found. Create one with these keys:")
        print("")
        print("  GEMINI_API_KEY_1=...")
        print("  GEMINI_API_KEY_2=...       # optional, for rate-limit rotation")
        print("  GROQ_API_KEY_1=...")
        print("  GROQ_API_KEY_2=...         # optional, for rate-limit rotation")
        print("  SARVAM_API_KEY=...         # optional, for Tamil STT/TTS")
        return False

    # Validate keys
    with open(".env") as f:
        env = f.read()

    gemini_keys = sum(1 for i in range(1, 3) if f"GEMINI_API_KEY_{i}=" in env)
    groq_keys   = sum(1 for i in range(1, 3) if f"GROQ_API_KEY_{i}=" in env)
    sarvam      = "SARVAM_API_KEY=" in env

    print(f"  Gemini embedding keys : {gemini_keys}")
    print(f"  Groq LLM/STT keys    : {groq_keys}")
    print(f"  Sarvam Tamil key      : {'✅' if sarvam else '—'}")

    if gemini_keys == 0:
        print("\n❌ At least one GEMINI_API_KEY_* is required for embeddings")
        return False
    if groq_keys == 0:
        print("\n❌ At least one GROQ_API_KEY_* is required for LLM + STT")
        return False

    # Test imports
    print_header("Testing Imports")
    test_modules = [
        ("fastapi",             "FastAPI"),
        ("faiss",               "FAISS"),
        ("google.generativeai", "Google Generative AI"),
        ("groq",                "Groq SDK"),
        ("edge_tts",            "Edge TTS"),
    ]
    for mod, name in test_modules:
        try:
            __import__(mod)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} — run: pip install -r requirements.txt")
            ok = False

    if not ok:
        return False

    # Build FAISS index
    print_header("Building FAISS Index")
    if not run_command("python -m rag_faiss.build_index", "FAISS index build"):
        print("⚠️  Index build failed — RAG will return empty context until fixed")

    # Done
    print_header("Setup Complete!")
    print("🚀 AIVA Backend ready!")
    print("")
    print("  Start server  : python main.py")
    print("  Health check   : http://localhost:8000/")
    print("  API docs       : http://localhost:8000/docs")
    print("")
    print("  Stack:")
    print("  • Embedding : Gemini embedding-001 (cloud, dual-key rotation)")
    print("  • LLM       : Groq llama-3.1-8b-instant (dual-key rotation)")
    print("  • STT       : Groq whisper-large-v3-turbo (en) / Sarvam saarika:v2 (ta)")
    print("  • TTS       : Edge TTS JennyNeural (en) / Sarvam bulbul:v2 (ta)")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)