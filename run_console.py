"""Console client for O.L.I.V.I.A. — talks to the running FastAPI backend."""

import argparse
import asyncio
import sys

from src.flet_app.services.api_client import OliviaAPIClient

DEFAULT_URL = "http://localhost:8000"


def print_banner():
    print("=" * 60)
    print("  O.L.I.V.I.A. Console Client")
    print("=" * 60)
    print("  Commands: /health /clear /quit")
    print()


def print_health(health: dict | None):
    if not health:
        print("  [!] Could not reach backend")
        return
    print()
    print("  Service Status:")
    print("  " + "-" * 30)
    for svc, status in health.get("services", {}).items():
        mark = "OK" if status else "--"
        print(f"  {svc:<16} [{mark}]")
    print("  " + "-" * 30)
    print()


async def input_loop(client: OliviaAPIClient):
    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")
        except EOFError:
            break

        text = user_input.strip()
        if not text:
            continue

        cmd = text.lower()
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/health":
            print_health(await client.get_health())
            continue
        if cmd == "/clear":
            ok = await client.clear_history()
            print("  History cleared." if ok else "  [!] Clear failed.")
            continue

        # Stream response
        sys.stdout.write("Olivia: ")
        sys.stdout.flush()
        async for tok in client.send_message_stream(text):
            sys.stdout.write(tok)
            sys.stdout.flush()
        print()


async def main():
    parser = argparse.ArgumentParser(description="O.L.I.V.I.A. console client")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend URL (default: %(default)s)")
    args = parser.parse_args()

    print_banner()

    client = OliviaAPIClient(base_url=args.url)
    try:
        print(f"  Connecting to {args.url} ...")
        connected = await client.check_connection(max_retries=5, retry_delay=1.0)
        if not connected:
            print(f"  [!] Backend not reachable at {args.url}")
            print("  Start it first: python run_olivia.py --api-only")
            return

        print("  Connected!")
        print_health(await client.get_health())

        await input_loop(client)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  Goodbye.")
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
