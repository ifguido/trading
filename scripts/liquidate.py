"""Liquida todas las posiciones spot a USDC (venta de mercado).

USO:
    python scripts/liquidate.py            # DRY-RUN: solo muestra qué vendería
    python scripts/liquidate.py --confirm  # EJECUTA las ventas reales

IMPORTANTE:
    - Detené el bot ANTES de correr esto, o recomprará lo que vendas.
    - Las ventas de mercado son IRREVERSIBLES.
    - Vende solo los activos base de los pares configurados en settings.yaml.
"""

from __future__ import annotations

import asyncio
import sys
from decimal import Decimal

import ccxt.async_support as ccxt

from src.core.config_loader import load_config


async def main(confirm: bool) -> None:
    config = load_config("config/settings.yaml")

    if config.exchange.mode != "live":
        print(f"Exchange mode = {config.exchange.mode}, no es 'live'. Abortando.")
        return

    exchange = ccxt.binance({
        "apiKey": config.exchange.api_key,
        "secret": config.exchange.api_secret,
        "enableRateLimit": True,
        "options": config.exchange.options,
    })

    try:
        await exchange.load_markets()
        balance = await exchange.fetch_balance()

        print(f"\n{'DRY-RUN (no vende nada)' if not confirm else 'MODO REAL (vende de verdad)'}")
        print("=" * 60)

        plan: list[tuple[str, str, Decimal, float]] = []
        for pair in config.pairs:
            base, quote = pair.symbol.split("/")
            free = Decimal(str((balance.get(base) or {}).get("free") or 0))
            if free <= Decimal("0"):
                print(f"  {pair.symbol:12} sin saldo ({base})")
                continue

            try:
                ticker = await exchange.fetch_ticker(pair.symbol)
                price = float(ticker["last"])
            except Exception as exc:
                print(f"  {pair.symbol:12} ERROR ticker: {exc}")
                continue

            notional = float(free) * price
            market = exchange.market(pair.symbol)
            min_notional = 0.0
            try:
                min_notional = float(market["limits"]["cost"]["min"] or 0)
            except (KeyError, TypeError):
                pass

            if notional < max(min_notional, 1.0):
                print(f"  {pair.symbol:12} {free} {base} ~${notional:.2f} -> DUST (debajo del mínimo, se omite)")
                continue

            print(f"  {pair.symbol:12} VENDER {free} {base} ~${notional:.2f}")
            plan.append((pair.symbol, base, free, notional))

        print("=" * 60)
        total = sum(n for _, _, _, n in plan)
        print(f"Total estimado a liquidar: ~${total:.2f} en {len(plan)} venta(s)\n")

        if not plan:
            print("Nada para vender.")
            return

        if not confirm:
            print("DRY-RUN: no se ejecutó ninguna venta. Corré con --confirm para vender.")
            return

        for symbol, base, free, _ in plan:
            amount = float(exchange.amount_to_precision(symbol, float(free)))
            print(f"Vendiendo {amount} {base} en {symbol} (market)...")
            order = await exchange.create_market_sell_order(symbol, amount)
            filled = order.get("filled") or amount
            avg = order.get("average") or order.get("price")
            print(f"  OK -> id={order.get('id')} filled={filled} avg={avg}")

        print("\nLiquidación completa. Verificá el balance en Binance.")

    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main(confirm="--confirm" in sys.argv))
