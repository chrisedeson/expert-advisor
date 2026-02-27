#!/usr/bin/env python3
"""Try to authenticate and authorize a specific cTrader account by login number.

If the account isn't linked to the current access token, we need to
use ProtoOAAccountAuthReq with the login credentials.
"""
import os
import sys
from pathlib import Path

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

results = []

def on_connected(client):
    req = ProtoOAApplicationAuthReq()
    req.clientId = CLIENT_ID
    req.clientSecret = CLIENT_SECRET
    d = client.send(req)
    d.addCallback(lambda _: get_accounts(client))
    d.addErrback(on_error)

def get_accounts(client):
    # First list accounts
    req = ProtoOAGetAccountListByAccessTokenReq()
    req.accessToken = ACCESS_TOKEN
    d = client.send(req)
    d.addCallback(lambda msg: on_accounts(msg, client))
    d.addErrback(on_error)

def on_accounts(msg, client):
    result = Protobuf.extract(msg)
    print(f"\nAccounts found: {len(result.ctidTraderAccount)}")
    for acc in result.ctidTraderAccount:
        print(f"  Login: {acc.traderLogin}, ctidTraderAccountId: {acc.ctidTraderAccountId}, isLive: {acc.isLive}")
        results.append({
            'login': acc.traderLogin,
            'ctid': acc.ctidTraderAccountId,
            'live': acc.isLive,
        })

    # Check if 9888090 is there
    logins = [a['login'] for a in results]
    if 9888090 in logins:
        print("\nAccount 9888090 FOUND!")
    else:
        print("\nAccount 9888090 NOT found in linked accounts.")
        print("This account was likely created under a different cTrader ID.")
        print("\nTo fix this, you need to:")
        print("1. Log into https://id.ctrader.com with the SAME email as your first account")
        print("2. Link the new demo account to the same cTrader ID")
        print("3. Then re-run this script")
        print("\nOR: The new account may already be under the same cID but needs a fresh OAuth.")
        print("Try running: python scripts/ctrader_auth.py")

    client.stopService()
    reactor.callFromThread(reactor.stop)

def on_error(failure):
    print(f"Error: {failure}")
    # Check if token expired
    if "INVALID" in str(failure) or "expired" in str(failure).lower():
        print("\nToken may be expired. Run: python scripts/ctrader_auth.py")
    reactor.callFromThread(reactor.stop)

client = Client(EndPoints.PROTOBUF_DEMO_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
client.setConnectedCallback(on_connected)
client.setDisconnectedCallback(lambda c, r: None)
client.setMessageReceivedCallback(lambda c, m: None)
client.startService()
reactor.run()
