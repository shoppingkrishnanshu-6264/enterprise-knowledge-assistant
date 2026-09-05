import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GROQ_API_KEY")

if key is None:
    print("GROQ_API_KEY is not set at all.")
else:
    print(f"Length: {len(key)}")
    print(f"repr (safe, shows hidden chars): {key[:6]!r} ... {key[-6:]!r}")
    non_ascii = [(i, c, hex(ord(c))) for i, c in enumerate(key) if ord(c) > 127]
    if non_ascii:
        print("Non-ASCII characters found:")
        for i, c, h in non_ascii:
            print(f"  position {i}: {c!r} ({h})")
    else:
        print("No non-ASCII characters found.")
    if key != key.strip():
        print("WARNING: key has leading/trailing whitespace (or invisible chars).")
