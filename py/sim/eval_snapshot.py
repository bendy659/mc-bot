"""Проверка снимка мозга моста в симуляции — без учителей, без подсказок,
без обучения и случайных действий (train_tag_sim --eval).

Запуск (из корня проекта):
    python py/sim/eval_snapshot.py data/brains/bridge --game-minutes 8
    python py/sim/eval_snapshot.py snaps/bridge15/040.0m --port 6110 --seed 3

Снимок — папка задачки (legs.pt, head.pt, hands.pt) или папка с папкой
bridge внутри (так кладёт train_tag_sim --snapshots). Его копия идёт во
временную папку мозгов: проверка не трогает ни data/brains, ни мозги
обучения в data/sim/brains, ни их файл остановки — её можно гонять, пока
идёт обучение (несколько проверок разом — с разными --port).

Итог — строка сводки моста (переходов в минуту на бота, среднее время,
упал, застрял, "дошёл из попыток" по видам) и доли действий по каналам
(вживую сеть жмёт столб чаще, чем в симуляции — за этим и следим); с
--json — ещё и строка в файл, чтобы сравнивать снимки потом.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import train_tag_sim  # noqa: E402  (он же настраивает ai_loop на симуляцию)
import ai_loop  # noqa: E402
from training_modules.bridge import BridgeModule  # noqa: E402

TASK = "bridge"


def find_brain(path: Path) -> Path:
    """Папка с .pt мозга моста: сама path или path/bridge."""
    if (path / "legs.pt").exists():
        return path
    if (path / TASK / "legs.pt").exists():
        return path / TASK
    raise SystemExit(f"[eval] В {path} нет мозга моста (legs.pt или {TASK}/legs.pt).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка снимка мозга моста в симуляции")
    parser.add_argument("snapshot", type=Path, help="папка снимка (bridge/ внутри или сразу *.pt)")
    parser.add_argument("--game-minutes", type=float, default=8.0, help="сколько минут игры проверять")
    parser.add_argument("--minutes", type=float, default=120.0, help="предел настоящего времени, мин")
    parser.add_argument("--bots", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1, help="одинаковый у снимков — одни и те же раскладки на старте")
    # Две проверки разом не должны делить порты zmq (ai_loop их занимает,
    # даже когда сообщения идут не через сокеты).
    parser.add_argument("--port", type=int, default=6100, help="порт zmq (занимает его и следующий)")
    parser.add_argument("--json", type=Path, default=None, help="дописать итог строкой JSON в этот файл")
    args = parser.parse_args()

    brain = find_brain(args.snapshot)
    work = Path(tempfile.mkdtemp(prefix="mcbot_eval_"))
    shutil.copytree(brain, work / "brains" / TASK)

    train_tag_sim.CONFIG["zmq"] = {"node_to_py": args.port, "py_to_node": args.port + 1}
    train_tag_sim.SIM_DIR = work
    train_tag_sim.STOP_FILE = work / "stop"
    train_tag_sim.prepare_brains = lambda: None  # копии игровых мозгов других задачек не нужны
    ai_loop.DATA_DIR = work
    ai_loop.BRAINS_DIR = work / "brains"
    ai_loop.METRICS_CSV = work / "metrics.csv"

    # События окна сводки — те же, по которым пишется строка "как дела".
    collected: Counter = Counter()
    original_summarize = BridgeModule.summarize.__func__

    def summarize(cls, events, ticks, bot_minutes):
        collected.update({key: value for key, value in events.items() if isinstance(value, (int, float))})
        collected["bot_minutes"] += bot_minutes
        return original_summarize(cls, events, ticks, bot_minutes)

    BridgeModule.summarize = classmethod(summarize)

    # Доли действий по каналам — по ответам ai_loop ботам.
    actions: dict[str, Counter] = {}
    original_send = train_tag_sim.Outbox.send_json

    def send_json(self, message):
        if message.get("type") == "action":
            for channel, name in message.get("actions", {}).items():
                actions.setdefault(channel, Counter())[name] += 1
        original_send(self, message)

    train_tag_sim.Outbox.send_json = send_json

    # Сводка — одна, в конце: окно метрик — вся проверка.
    sys.argv = ["train_tag_sim.py", "--game", TASK, "--eval", "--bots", str(args.bots),
                "--minutes", str(args.minutes), "--game-minutes", str(args.game_minutes),
                "--report", str(args.game_minutes * 60 * 10), "--seed", str(args.seed)]
    try:
        train_tag_sim.main()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    bot_minutes = collected.pop("bot_minutes", 0.0)
    summary, rate = original_summarize(BridgeModule, dict(collected), 0, bot_minutes)
    shares = {channel: {name: round(count / sum(counter.values()), 3) for name, count in counter.most_common()}
              for channel, counter in actions.items()}
    print(f"[eval] {args.snapshot}: {summary}")
    for channel, share in shares.items():
        print(f"[eval]   {channel}: " + ", ".join(f"{name} {100 * value:.0f}%" for name, value in share.items()))
    if args.json:
        with open(args.json, "a", encoding="utf-8") as f:
            f.write(json.dumps({"snapshot": str(args.snapshot), "rate": rate, "bot_minutes": round(bot_minutes, 2),
                                "events": dict(collected), "actions": shares, "seed": args.seed},
                               ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
