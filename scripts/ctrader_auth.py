#!/usr/bin/env python3
"""cTrader Open API OAuth2 authentication helper.

Run this once to get access/refresh tokens, then store them in .env.
Refresh tokens are long-lived - you won't need to re-auth unless revoked.

Usage:
    python scripts/ctrader_auth.py

Requires CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET in .env or environment.
"""
import os
import sys
import json
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# Load .env if exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from ctrader_open_api import Auth, EndPoints

CLIENT_ID = os.environ.get("CTRADER_CLIENT_ID")
CLIENT_SECRET = os.environ.get("CTRADER_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:5000/callback"

if not CLIENT_ID or not CLIENT_SECRET:
    print("ERROR: Set CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET in .env file")
    print()
    print("Create a .env file in the project root with:")
    print("  CTRADER_CLIENT_ID=your_id_here")
    print("  CTRADER_CLIENT_SECRET=your_secret_here")
    sys.exit(1)

auth = Auth(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)
auth_code = None
server_done = threading.Event()


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        parsed = urlparse(self.path)
        if parsed.path == "/callback":
            params = parse_qs(parsed.query)
            if "code" in params:
                auth_code = params["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>Authorization successful!</h1><p>You can close this tab.</p>")
                server_done.set()
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"No auth code received")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


def main():
    # Step 1: Get auth URL
    auth_url = auth.getAuthUri(scope="trading")
    print("=" * 60)
    print("cTrader Open API Authentication")
    print("=" * 60)
    print()
    print("Opening browser for authorization...")
    print("If browser doesn't open, visit this URL manually:")
    print()
    print(f"  {auth_url}")
    print()

    # Step 2: Start local server to catch callback
    server = HTTPServer(("localhost", 5000), CallbackHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Open browser
    webbrowser.open(auth_url)

    print("Waiting for authorization callback...")
    server_done.wait(timeout=120)
    server.shutdown()

    if not auth_code:
        print("ERROR: Timed out waiting for authorization")
        sys.exit(1)

    print(f"Auth code received: {auth_code[:10]}...")

    # Step 3: Exchange code for tokens
    print("Exchanging code for tokens...")
    token_response = auth.getToken(auth_code)

    if "access_token" not in token_response:
        print(f"ERROR: Token exchange failed: {token_response}")
        sys.exit(1)

    access_token = token_response["access_token"]
    refresh_token = token_response.get("refresh_token", "")
    expires_in = token_response.get("expires_in", "unknown")

    print()
    print("SUCCESS! Tokens received.")
    print(f"  Access token expires in: {expires_in} seconds")
    print()

    # Step 4: Get account list
    print("Fetching trading accounts...")
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints as EP
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOAGetAccountListByAccessTokenRes,
    )
    from twisted.internet import reactor, defer

    accounts_found = []

    def on_connected(client):
        req = ProtoOAGetAccountListByAccessTokenReq()
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAApplicationAuthReq
        app_req = ProtoOAApplicationAuthReq()
        app_req.clientId = CLIENT_ID
        app_req.clientSecret = CLIENT_SECRET
        d = client.send(app_req)
        d.addCallback(lambda _: _get_accounts(client))
        d.addErrback(on_error)

    def _get_accounts(client):
        req = ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = access_token
        d = client.send(req)
        d.addCallback(lambda msg: _on_accounts(msg, client))
        d.addErrback(on_error)

    def _on_accounts(msg, client):
        result = Protobuf.extract(msg)
        for acc in result.ctidTraderAccount:
            accounts_found.append({
                "ctidTraderAccountId": acc.ctidTraderAccountId,
                "isLive": acc.isLive,
                "traderLogin": acc.traderLogin,
            })
        client.stopService()
        reactor.callFromThread(reactor.stop)

    def on_error(failure):
        print(f"Error: {failure}")
        reactor.callFromThread(reactor.stop)

    def on_disconnected(client, reason):
        pass

    client = Client(EP.PROTOBUF_DEMO_HOST, EP.PROTOBUF_PORT, TcpProtocol)
    client.setConnectedCallback(on_connected)
    client.setDisconnectedCallback(on_disconnected)
    client.setMessageReceivedCallback(lambda c, m: None)
    client.startService()
    reactor.run()

    # Step 5: Save to .env
    print()
    if accounts_found:
        print("Trading accounts found:")
        for acc in accounts_found:
            live_str = "LIVE" if acc["isLive"] else "DEMO"
            print(f"  [{live_str}] Account ID: {acc['ctidTraderAccountId']}, Login: {acc['traderLogin']}")
        demo_accounts = [a for a in accounts_found if not a["isLive"]]
        account_id = demo_accounts[0]["ctidTraderAccountId"] if demo_accounts else accounts_found[0]["ctidTraderAccountId"]
    else:
        print("No accounts found. You'll need to set CTRADER_ACCOUNT_ID manually.")
        account_id = "SET_ME"

    # Write/update .env
    env_lines = []
    if env_path.exists():
        env_lines = env_path.read_text().splitlines()

    env_vars = {
        "CTRADER_CLIENT_ID": CLIENT_ID,
        "CTRADER_CLIENT_SECRET": CLIENT_SECRET,
        "CTRADER_ACCESS_TOKEN": access_token,
        "CTRADER_REFRESH_TOKEN": refresh_token,
        "CTRADER_ACCOUNT_ID": str(account_id),
    }

    # Update existing or append
    existing_keys = set()
    for i, line in enumerate(env_lines):
        for key in env_vars:
            if line.startswith(f"{key}="):
                env_lines[i] = f"{key}={env_vars[key]}"
                existing_keys.add(key)

    for key, value in env_vars.items():
        if key not in existing_keys:
            env_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(env_lines) + "\n")
    print()
    print(f"Credentials saved to {env_path}")
    print()
    print("You're all set! The live engine can now connect to cTrader.")


if __name__ == "__main__":
    main()
