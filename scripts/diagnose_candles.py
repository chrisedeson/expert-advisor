#!/usr/bin/env python3
"""Diagnostic script to verify trendbar price decoding.

Connects to cTrader, fetches one H1 bar for each of our 5 symbols,
and prints raw vs decoded prices alongside live spot prices for comparison.

Usage:
    python scripts/diagnose_candles.py
"""
import os
import sys
import time
import threading
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Load .env
env_path = project_root / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOASymbolsListReq,
    ProtoOASymbolByIdReq,
    ProtoOAGetTrendbarsReq,
    ProtoOASubscribeSpotsReq,
)
from twisted.internet import reactor

SYMBOLS = ["EURUSD", "GBPUSD", "EURJPY", "XAGUSD", "US500"]

CLIENT_ID = os.environ.get("CTRADER_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("CTRADER_CLIENT_SECRET", "")
ACCESS_TOKEN = os.environ.get("CTRADER_ACCESS_TOKEN", "")
ACCOUNT_ID = int(os.environ.get("CTRADER_ACCOUNT_ID", "0"))

if not all([CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID]):
    print("ERROR: Missing credentials in .env")
    print("Need: CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, CTRADER_ACCOUNT_ID")
    sys.exit(1)

# State
symbol_map = {}
symbol_details = {}
spot_prices = {}
results_ready = threading.Event()
done = threading.Event()


def on_connected(client):
    print("Connected, authenticating...")
    req = ProtoOAApplicationAuthReq()
    req.clientId = CLIENT_ID
    req.clientSecret = CLIENT_SECRET
    d = client.send(req)
    d.addCallback(lambda _: on_app_auth(client))
    d.addErrback(lambda f: print(f"Error: {f}"))


def on_app_auth(client):
    print("App authenticated, authenticating account...")
    req = ProtoOAAccountAuthReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    req.accessToken = ACCESS_TOKEN
    d = client.send(req)
    d.addCallback(lambda _: on_account_auth(client))
    d.addErrback(lambda f: print(f"Error: {f}"))


def on_account_auth(client):
    print(f"Account {ACCOUNT_ID} authenticated, loading symbols...")
    req = ProtoOASymbolsListReq()
    req.ctidTraderAccountId = ACCOUNT_ID
    d = client.send(req)
    d.addErrback(lambda f: print(f"Error: {f}"))


def on_message(client, msg):
    try:
        result = Protobuf.extract(msg)
        msg_type = type(result).__name__

        if msg_type == "ProtoOASymbolsListRes":
            for sym in result.symbol:
                symbol_map[sym.symbolName] = sym.symbolId
            print(f"Loaded {len(result.symbol)} symbols")
            our_ids = [symbol_map[n] for n in SYMBOLS if n in symbol_map]
            missing = [n for n in SYMBOLS if n not in symbol_map]
            if missing:
                print(f"WARNING: Not found: {missing}")
            if our_ids:
                req = ProtoOASymbolByIdReq()
                req.ctidTraderAccountId = ACCOUNT_ID
                for sid in our_ids:
                    req.symbolId.append(sid)
                client.send(req)

        elif msg_type == "ProtoOASymbolByIdRes":
            for sym in result.symbol:
                symbol_details[sym.symbolId] = {
                    "symbolId": sym.symbolId,
                    "digits": sym.digits,
                    "pipPosition": sym.pipPosition if hasattr(sym, "pipPosition") else None,
                    "lotSize": sym.lotSize if hasattr(sym, "lotSize") else None,
                    "stepVolume": sym.stepVolume if hasattr(sym, "stepVolume") else None,
                    "minVolume": sym.minVolume if hasattr(sym, "minVolume") else None,
                }
            print(f"Got details for {len(result.symbol)} symbols")

            # Subscribe to spot prices
            for name in SYMBOLS:
                sid = symbol_map.get(name)
                if sid:
                    req = ProtoOASubscribeSpotsReq()
                    req.ctidTraderAccountId = ACCOUNT_ID
                    req.symbolId.append(sid)
                    client.send(req)

            # Fetch H1 candles for each symbol
            now_ms = int(time.time() * 1000)
            for name in SYMBOLS:
                sid = symbol_map.get(name)
                if sid:
                    req = ProtoOAGetTrendbarsReq()
                    req.ctidTraderAccountId = ACCOUNT_ID
                    req.symbolId = sid
                    req.period = 9  # H1
                    req.fromTimestamp = now_ms - (5 * 3600 * 1000)  # last 5 hours
                    req.toTimestamp = now_ms
                    client.send(req)

        elif msg_type == "ProtoOAGetTrendbarsRes":
            sid = result.symbolId if hasattr(result, "symbolId") else None
            name = next((n for n, s in symbol_map.items() if s == sid), f"ID:{sid}")
            details = symbol_details.get(sid, {})
            digits = details.get("digits", 5)

            if hasattr(result, "trendbar") and len(result.trendbar) > 0:
                bar = result.trendbar[-1]  # last bar
                raw_low = bar.low
                raw_delta_open = bar.deltaOpen if bar.deltaOpen else 0
                raw_delta_high = bar.deltaHigh if bar.deltaHigh else 0
                raw_delta_close = bar.deltaClose if bar.deltaClose else 0

                # Method 1: Always /100000 (CORRECT per cTrader docs)
                price_low_correct = round(raw_low / 100000.0, digits)
                price_open_correct = round((raw_low + raw_delta_open) / 100000.0, digits)
                price_high_correct = round((raw_low + raw_delta_high) / 100000.0, digits)
                price_close_correct = round((raw_low + raw_delta_close) / 100000.0, digits)

                # Method 2: /10^digits (OLD BUGGY code)
                old_divisor = 10 ** digits
                price_low_old = raw_low / old_divisor
                price_close_old = (raw_low + raw_delta_close) / old_divisor

                print(f"\n{'='*60}")
                print(f"SYMBOL: {name}")
                print(f"  digits={digits}, pipPosition={details.get('pipPosition')}")
                print(f"  lotSize={details.get('lotSize')}, stepVolume={details.get('stepVolume')}")
                print(f"  RAW: low={raw_low}, dOpen={raw_delta_open}, dHigh={raw_delta_high}, dClose={raw_delta_close}")
                print(f"  CORRECT (/100000): O={price_open_correct} H={price_high_correct} L={price_low_correct} C={price_close_correct}")
                print(f"  OLD BUG (/10^{digits}={old_divisor}): L={price_low_old:.6f} C={price_close_old:.6f}")

                spot = spot_prices.get(sid, {})
                if spot:
                    print(f"  SPOT: bid={spot.get('bid')}, ask={spot.get('ask')}")
                else:
                    print(f"  SPOT: (waiting for data...)")

            results_ready.set()

        elif msg_type == "ProtoOASpotEvent":
            sid = result.symbolId
            details = symbol_details.get(sid, {})
            digits = details.get("digits", 5)
            if sid not in spot_prices:
                spot_prices[sid] = {}
            if result.HasField("bid"):
                spot_prices[sid]["bid"] = round(result.bid / 100000.0, digits)
            if result.HasField("ask"):
                spot_prices[sid]["ask"] = round(result.ask / 100000.0, digits)

        elif msg_type == "ProtoOAErrorRes":
            print(f"ERROR: {result.errorCode} - {getattr(result, 'description', '')}")

    except Exception as e:
        pass  # Ignore parse errors for non-relevant messages


def on_disconnected(client, reason):
    print(f"Disconnected: {reason}")
    done.set()


def main():
    print("cTrader Candle Diagnostic")
    print("=" * 60)

    client = Client(EndPoints.PROTOBUF_DEMO_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
    client.setConnectedCallback(on_connected)
    client.setDisconnectedCallback(on_disconnected)
    client.setMessageReceivedCallback(on_message)

    reactor_thread = threading.Thread(target=lambda: reactor.run(installSignalHandlers=False), daemon=True)
    reactor_thread.start()

    reactor.callFromThread(client.startService)

    # Wait for results
    print("Waiting for candle data (15s timeout)...")
    results_ready.wait(timeout=15)
    time.sleep(3)  # extra time for spot prices

    # Print spot prices summary
    print(f"\n{'='*60}")
    print("SPOT PRICES SUMMARY (correct /100000 decoding):")
    for name in SYMBOLS:
        sid = symbol_map.get(name)
        if sid and sid in spot_prices:
            print(f"  {name}: bid={spot_prices[sid].get('bid')}, ask={spot_prices[sid].get('ask')}")
        else:
            print(f"  {name}: no spot data")

    print(f"\n{'='*60}")
    print("DONE. Compare CORRECT prices with actual market prices on cTrader/TradingView.")
    print("If they match, the /100000 fix is correct.")

    reactor.callFromThread(client.stopService)
    time.sleep(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
