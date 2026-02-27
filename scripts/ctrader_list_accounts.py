#!/usr/bin/env python3
"""List all cTrader accounts linked to the current access token."""
import os
import sys
from pathlib import Path

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAGetAccountListByAccessTokenReq,
)
from twisted.internet import reactor

CLIENT_ID = os.environ.get("CTRADER_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("CTRADER_CLIENT_SECRET", "")
ACCESS_TOKEN = os.environ.get("CTRADER_ACCESS_TOKEN", "")

if not all([CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN]):
    print("ERROR: Missing CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, or CTRADER_ACCESS_TOKEN in .env")
    sys.exit(1)

accounts_found = []

def on_connected(client):
    req = ProtoOAApplicationAuthReq()
    req.clientId = CLIENT_ID
    req.clientSecret = CLIENT_SECRET
    d = client.send(req)
    d.addCallback(lambda _: get_accounts(client))
    d.addErrback(on_error)

def get_accounts(client):
    req = ProtoOAGetAccountListByAccessTokenReq()
    req.accessToken = ACCESS_TOKEN
    d = client.send(req)
    d.addCallback(lambda msg: on_accounts(msg, client))
    d.addErrback(on_error)

def on_accounts(msg, client):
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

client = Client(EndPoints.PROTOBUF_DEMO_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
client.setConnectedCallback(on_connected)
client.setDisconnectedCallback(lambda c, r: None)
client.setMessageReceivedCallback(lambda c, m: None)
client.startService()
reactor.run()

if accounts_found:
    print("\nAll cTrader accounts linked to your token:")
    print(f"{'Type':<6} {'Login':<12} {'ctidTraderAccountId':<22}")
    print("-" * 40)
    for acc in accounts_found:
        live_str = "LIVE" if acc["isLive"] else "DEMO"
        print(f"{live_str:<6} {acc['traderLogin']:<12} {acc['ctidTraderAccountId']}")
else:
    print("No accounts found. Token may be expired - run ctrader_auth.py to refresh.")
