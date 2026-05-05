"""
trading_bot.py — Bot de trading Binance (spot).
Démarre en paper trading (DRY_RUN=true par défaut).
Passer en live : BINANCE_API_KEY + BINANCE_SECRET + DRY_RUN=false dans .env
"""
import os
import json
import uuid
import time
import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
import structlog

log = structlog.get_logger()


@dataclass
class Order:
    id:           str
    symbol:       str
    side:         str
    type:         str
    amount:       float
    price:        float
    sl_price:     float
    tp1_price:    float
    tp2_price:    float
    strategy:     str
    dry_run:      bool = True
    status:       str = "pending"
    exchange_id:  Optional[str] = None
    filled_price: Optional[float] = None
    pnl_pct:      Optional[float] = None
    pnl_usd:      Optional[float] = None
    tp1_hit:      bool = False
    opened_at:    str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TradingBot:
    """
    Bot de trading Binance spot.
    Paper trading par défaut (DRY_RUN=true).
    """

    ORDERS_FILE = "data/orders.json"

    def __init__(self):
        self.dry_run  = os.getenv("DRY_RUN", "true").lower() == "true"
        self.exchange = self._init_exchange()
        os.makedirs("data", exist_ok=True)
        log.info("trading_bot_init",
                 mode="PAPER" if self.dry_run else "LIVE",
                 exchange_ok=self.exchange is not None)

    def _init_exchange(self):
        api_key    = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_SECRET", "")

        if not api_key or not api_secret:
            log.info("binance_no_keys — paper trading only")
            return None

        try:
            import ccxt
            exchange = ccxt.binance({
                "apiKey":          api_key,
                "secret":          api_secret,
                "enableRateLimit": True,
                "options":         {"defaultType": "spot"},
            })
            exchange.fetch_balance()
            log.info("binance_connected")
            return exchange
        except Exception as e:
            log.error("binance_init_failed", error=str(e))
            return None

    def _get_price(self, symbol: str) -> float:
        if self.exchange:
            ticker = self.exchange.fetch_ticker(f"{symbol}/USDT")
            return float(ticker["last"])
        import requests
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": f"{symbol}USDT"},
            timeout=5,
        )
        return float(r.json()["price"])

    def get_account_info(self) -> dict:
        if not self.exchange or self.dry_run:
            return {
                "mode":    "PAPER TRADING",
                "balance": {"USDT": 10000.0},
                "note":    "Configurez BINANCE_API_KEY pour le live",
            }
        try:
            balance   = self.exchange.fetch_balance()
            usdt      = balance["USDT"]["free"]
            positions = {
                k: v for k, v in balance["total"].items()
                if v > 0 and k not in ["USDT", "BUSD", "FDUSD"]
            }
            return {
                "mode":      "LIVE",
                "usdt_free": usdt,
                "positions": positions,
                "ts":        datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}

    def place_order(self, symbol: str, side: str,
                    amount_usdt: float,
                    sl_pct: float = 10.0,
                    tp1_pct: float = 15.0,
                    tp2_pct: float = 30.0,
                    strategy: str = "GEM_SWING") -> Order:
        price  = self._get_price(symbol)
        amount = amount_usdt / price

        if side == "buy":
            sl_price  = price * (1 - sl_pct / 100)
            tp1_price = price * (1 + tp1_pct / 100)
            tp2_price = price * (1 + tp2_pct / 100)
        else:
            sl_price  = price * (1 + sl_pct / 100)
            tp1_price = price * (1 - tp1_pct / 100)
            tp2_price = price * (1 - tp2_pct / 100)

        order = Order(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            type="market",
            amount=amount,
            price=price,
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            strategy=strategy,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            order.status       = "filled_simulated"
            order.filled_price = price
            log.info("order_simulated", symbol=symbol, side=side,
                     amount_usdt=amount_usdt, price=price)
        else:
            max_order = float(os.getenv("MAX_ORDER_USDT", "200"))
            if amount_usdt > max_order:
                raise ValueError(
                    f"Montant {amount_usdt} > MAX_ORDER_USDT ({max_order})"
                )
            result = self.exchange.create_market_order(
                f"{symbol}/USDT", side, amount
            )
            order.exchange_id  = result["id"]
            order.status       = "filled"
            order.filled_price = result.get("average", price)
            log.info("order_live", symbol=symbol, side=side,
                     price=order.filled_price, amount_usdt=amount_usdt)

        self._save_order(order)
        return order

    def update_positions(self) -> list:
        orders      = self._load_orders()
        open_orders = [
            o for o in orders
            if o.get("status") in ("filled", "filled_simulated")
            and o.get("side") == "buy"
        ]
        updated = []

        for o in open_orders:
            symbol = o["symbol"]
            try:
                price   = self._get_price(symbol)
                entry   = o["price"]
                pnl_pct = (price - entry) / entry * 100

                o["current_price"] = price
                o["pnl_pct"]       = round(pnl_pct, 2)
                o["pnl_usd"]       = round(o["amount"] * (price - entry), 2)

                if price <= o["sl_price"]:
                    o["status"] = "stopped"
                    log.warning("stop_loss_hit", symbol=symbol,
                                price=price, pnl_pct=pnl_pct)
                    if not self.dry_run and self.exchange:
                        self.exchange.create_market_order(
                            f"{symbol}/USDT", "sell", o["amount"]
                        )
                elif price >= o["tp1_price"] and not o.get("tp1_hit"):
                    o["tp1_hit"] = True
                    o["sl_price"] = o["price"]  # breakeven
                    log.info("tp1_hit", symbol=symbol,
                             price=price, pnl_pct=pnl_pct)
                    if not self.dry_run and self.exchange:
                        self.exchange.create_market_order(
                            f"{symbol}/USDT", "sell", o["amount"] * 0.33
                        )

                updated.append(o)
            except Exception as e:
                log.error("update_position_failed", symbol=symbol, error=str(e))

        non_open = [
            o for o in orders
            if o.get("status") not in ("filled", "filled_simulated")
        ]
        self._save_all_orders(non_open + updated)
        return updated

    def get_performance(self) -> dict:
        orders = self._load_orders()
        closed = [
            o for o in orders
            if o.get("status") in ("stopped", "tp2_hit", "closed")
        ]
        if not closed:
            return {"message": "Pas encore de trades fermés", "mode": "PAPER" if self.dry_run else "LIVE"}

        wins      = [o for o in closed if (o.get("pnl_pct") or 0) > 0]
        total_pnl = sum(o.get("pnl_usd") or 0 for o in closed)

        return {
            "total_trades":  len(closed),
            "win_rate":      round(len(wins) / len(closed) * 100, 1),
            "total_pnl_usd": round(total_pnl, 2),
            "avg_win_pct":   round(
                sum((o.get("pnl_pct") or 0) for o in wins) / max(1, len(wins)), 1
            ),
            "avg_loss_pct":  round(
                sum((o.get("pnl_pct") or 0) for o in closed
                    if (o.get("pnl_pct") or 0) <= 0) / max(1, len(closed) - len(wins)), 1
            ),
            "mode": "PAPER" if self.dry_run else "LIVE",
        }

    def _load_orders(self) -> list:
        if os.path.exists(self.ORDERS_FILE):
            with open(self.ORDERS_FILE) as f:
                return json.load(f)
        return []

    def _save_order(self, order: Order):
        orders = self._load_orders()
        orders.append(asdict(order))
        self._save_all_orders(orders)

    def _save_all_orders(self, orders: list):
        os.makedirs("data", exist_ok=True)
        with open(self.ORDERS_FILE, "w") as f:
            json.dump(orders, f, indent=2, default=str)
