import argparse
import json
import urllib.request

NORMAL = [
    "NASA is planning a rocket launch into lunar orbit.",
    "The baseball pitcher helped the team win the game.",
    "My car engine makes a noise when driving on the road.",
    "How can I render a graphics image with better pixels?",
]
SHIFTED = ["zzqqvv " * 400, "qzxwv blrpt xnqvz", "unrelated vocabulary zzxxqq"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--shifted", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    args = parser.parse_args()
    texts = SHIFTED if args.shifted else NORMAL
    last = None
    for index in range(args.count):
        body = json.dumps({"text": texts[index % len(texts)]}).encode()
        request = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            last = json.load(response)
    print(json.dumps({"requests": args.count, "last_response": last}, indent=2))


if __name__ == "__main__":
    main()
