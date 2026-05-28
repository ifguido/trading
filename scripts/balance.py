"""Muestra el balance completo del wallet SPOT de Binance (todos los activos > 0).

USO:
    python scripts/balance.py

Solo lectura. Incluye USDC y cualquier activo, no solo los pares configurados.
Nota: el bot opera SOLO el wallet spot; fondos en Funding/Earn/Futures no aparecen.
"""

from __future__ import annotations

import asyncio

import ccxt.async_support as ccxt

from src.core.config_loader import load_config


async def main() -> None:
    config = load_config("config/settings.yaml")
    exchange = ccxt.binance({
        "apiKey": config.exchange.api_key,
        "secret": config.exchange.api_secret,
        "enableRateLimit": True,
        "options": config.exchange.options,
    })

    try:
        await exchange.load_markets()
        balance = await exchange.fetch_balance()

        print("\nBalance SPOT (activos con saldo > 0):")
        print("=" * 50)

        total_usdc = 0.0
        rows: list[tuple[str, float, float]] = []
        for asset, amount in (balance.get("total") or {}).items():
            amt = float(amount or 0)
            if amt <= 0:
                continue

            if asset in ("USDC", "USDT", "BUSD", "FDUSD"):
                usd_value = amt
            else:
                try:
                    ticker = await exchange.fetch_ticker(f"{asset}/USDC")
                    usd_value = amt * float(ticker["last"])
                except Exception:
                    usd_value = 0.0  # sin par /USDC: no se puede valuar

            total_usdc += usd_value
            rows.append((asset, amt, usd_value))

        for asset, amt, usd in sorted(rows, key=lambda r: -r[2]):
            val = f"~${usd:.2f}" if usd > 0 else "(sin par /USDC)"
            print(f"  {asset:8} {amt:<20} {val}")

        print("=" * 50)
        print(f"Total valuado en SPOT: ~${total_usdc:.2f}")
        print("\nNota: fondos en Funding / Earn / Futures NO aparecen acá.")

    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
