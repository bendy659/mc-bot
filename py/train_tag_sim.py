"""Обучение салок и охоты ("Останови меня") в симуляции (py/sim/) — без
Minecraft.

Запуск:
    python py/train_tag_sim.py --minutes 30          # учить 30 минут
    python py/train_tag_sim.py --teachers 6          # половина ботов — учитель (примеры для мозгов)
    python py/train_tag_sim.py --eval --minutes 3     # проверка: без обучения и случайных действий
    python py/train_tag_sim.py --game hunt --teachers 4   # охота: цель — случайный бот, по кругу
    python py/train_tag_sim.py --game hunt --scripted-target   # охота на "человека" (его ведёт учитель)
    python py/train_tag_sim.py --install             # перенести мозги в игру
    python py/train_tag_sim.py --install chase flee  # только эти (охота ещё учится)

Учит тот же ai_loop, что и в игре, — с тем же судьёй салок, наградами,
памятью опыта и мозгами chase/flee, только состояния ему шлёт не Node, а
симуляция (py/sim/game.py), и время в ней идёт так быстро, как считает
компьютер. Мозги симуляции — в data/sim/brains/ (при первом запуске —
копия игровых data/brains/chase и flee), игровые не трогаются, так что
учить можно и при запущенном рое (только оба будут делить видеокарту).
Перенести обученное в игру — --install, при ОСТАНОВЛЕННОМ рое (иначе он
перезапишет файлы своими).

Что видит мозг в симуляции — то же, что в игре (сверка:
python py/sim/check_parity.py). Копать и строить можно (py/sim/world.py:
время копки как в игре; выкопанное — сразу в инвентарь, как у ботов плагина),
арена — заново в начале каждого раунда и охоты. Чего нет: воды, выпавших
предметов, еды и брони, задержек сети, людей (кроме "человека" по скрипту в
охоте) — это доучивается уже в игре.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "data" / "sim"
# Мягкая остановка: появился этот файл — сохранить мозги и выйти (снимать
# процесс силой — потерять обучение с последнего сохранения).
STOP_FILE = SIM_DIR / "stop"
TASKS = ("chase", "flee", "hunt", "walking", "bridge", "bedwars")  # walking — основа моста (train.warm_start_from)
TARGET_NAME = "Target"  # "человек" в охоте — ведёт учитель убегающего
HIDE_NAMES = {"pillar": "столб", "box": "коробка", "run": "бег"}

import config as config_module  # noqa: E402

CONFIG = config_module.CONFIG
# Свои порты (не мешать рою) и никакого RCON: судья шлёт команды симуляции,
# а не серверу-арене. Сама арена (размер, раскладка) — из того же конфига.
CONFIG["zmq"] = {"node_to_py": 5975, "py_to_node": 5976}
CONFIG["server"] = {key: value for key, value in CONFIG["server"].items() if key != "rcon"}

import ai_loop  # noqa: E402
import bedwars_game  # noqa: E402
from bridge_course import BridgeCourse  # noqa: E402
import hunt_game  # noqa: E402
import tag_game  # noqa: E402
from sim.game import SimArena  # noqa: E402
from sim.world import BLOCK_NAMES  # noqa: E402
from sim.teacher import bridge_rescue, teacher_actions  # noqa: E402



def use_sim_dir(folder: Path) -> None:
    """Где лежат мозги, память и метрики симуляции (и файл остановки).
    Своя папка — чтобы долгое обучение не перезаписывало data/sim в git
    каждые пару минут (в data/sim тогда кладёшь сам, когда нужно), а
    проверка снимка (sim/eval_snapshot.py) — не трогала мозги обучения."""
    global SIM_DIR, STOP_FILE
    SIM_DIR = folder
    STOP_FILE = folder / "stop"
    ai_loop.DATA_DIR = folder
    ai_loop.BRAINS_DIR = folder / "brains"
    ai_loop.METRICS_CSV = folder / "metrics.csv"


use_sim_dir(SIM_DIR)


class Outbox:
    """Вместо сокета к Node: сообщения ai_loop за тик — действия, цели, чат."""

    def __init__(self):
        self.actions: dict[int, dict] = {}
        self.targets: dict[int, dict] = {}
        self.chat: list[str] = []

    def send_json(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "action":
            self.actions[message["bot_id"]] = message
        elif kind == "set_target":
            self.targets[message["bot_id"]] = message
        elif kind == "chat":
            self.chat.append(message.get("text", ""))


class SimServer:
    """Вместо сервера-арены (TrainingServer): команды судьи — в симуляцию."""

    simulated = True  # судья моста строит трассу сам только настоящему серверу

    def __init__(self, arena: SimArena):
        self.arena = arena

    def send(self, command: str) -> None:
        self.arena.command(command)

    def add_reward(self, name: str, reward: float, now: float) -> None:
        pass  # табло "Мотивация" в симуляции некому смотреть

    def publish(self, now: float) -> None:
        pass

    def run_function(self, name: str) -> int:
        return 0  # наборы команд задач (server/functions) — только настоящему серверу

    def is_bed(self, world: str, cells: list) -> bool:
        """Бедварс: цела ли кровать (как TrainingServer.is_bed по RCON)."""
        return all(BLOCK_NAMES[self.arena.world.block(*cell)].endswith("_bed") for cell in cells)

    def close(self) -> None:
        pass


def prepare_brains() -> None:
    """Первый запуск — копия игровых мозгов: симуляция учит дальше их."""
    for task in TASKS:
        target = SIM_DIR / "brains" / task
        source = ROOT / "data" / "brains" / task
        if not target.exists() and source.exists():
            shutil.copytree(source, target)
            print(f"[sim] Мозг {task}: начинаю с копии игрового ({source}).")


def install(tasks: list[str]) -> int:
    """Мозги симуляции -> игра (старые игровые — в data/brains_old/). tasks —
    какие задачки: салки можно перенести, пока охота ещё учится."""
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        print(f"[sim] Таких задачек нет: {', '.join(unknown)} (есть {', '.join(TASKS)}).")
        return 1
    live = ROOT / "data" / "brains"
    recent = [p for task in TASKS for p in (live / task).glob("*.pt") if time.time() - p.stat().st_mtime < 120]
    if recent:
        print("[sim] Игровые мозги менялись меньше 2 минут назад — рой, похоже, запущен. Останови рой "
              "(он перезапишет файлы своими) и повтори --install.")
        return 1
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for task in tasks:
        source = SIM_DIR / "brains" / task
        if not source.exists():
            print(f"[sim] Мозга {task} в симуляции нет — пропускаю.")
            continue
        target = live / task
        if target.exists():
            backup = ROOT / "data" / "brains_old" / f"{task}_before_sim_{stamp}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), backup)
            print(f"[sim] Игровой мозг {task} отложен в {backup}.")
        shutil.copytree(source, target)
        print(f"[sim] Мозг {task} из симуляции — в игре.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Салки в симуляции: обучение мозгов chase/flee без Minecraft")
    parser.add_argument("--minutes", type=float, default=10.0, help="сколько учить (минут настоящего времени)")
    # Проверки снимков сравнивают по игровому времени: на машине без
    # видеокарты игра идёт медленнее, и те же --minutes — меньше попыток.
    parser.add_argument("--game-minutes", type=float, default=0.0,
                        help="остановиться после стольких минут ИГРОВОГО времени (0 — только по --minutes)")
    parser.add_argument("--bots", type=int, default=CONFIG["bot"].get("count", 12), help="ботов на арене")
    parser.add_argument("--report", type=float, default=300.0, help="сводка раз в столько секунд игрового времени")
    # Сколько шагов обучения на одно состояние. В игре — train.learn_steps_per_tick
    # (0.5), но шаг обучения стоит ~40 мс видеокарты, и симуляция упиралась бы
    # в него (медленнее реального времени). Реже учиться — больше разных
    # ситуаций на каждый шаг обучения.
    parser.add_argument("--learn-ratio", type=float, default=0.25, help="шагов обучения на одно состояние")
    # Учитель (py/sim/teacher.py) ведёт первых N ботов: их ходы уходят в память
    # мозгов как демонстрации (DQfD), а остальные боты играют с ним и учатся.
    parser.add_argument("--teachers", type=int, default=0, help="сколько ботов ведёт учитель")
    parser.add_argument("--eval", action="store_true", help="проверка: мозги без случайных действий, без обучения")
    parser.add_argument("--game", choices=("tag", "hunt", "bridge", "bedwars"), default="tag",
                        help="салки, охота (Останови меня), мост над пустотой (bridge: трасса bridge_course) "
                             "или бедварс (две команды на картах Hypixel, py/bedwars_game.py)")
    # Охота: цель — случайный бот роя (как !start auto, сам по кругу) или
    # "человек", которого ведёт учитель убегающего; он ходит шагом (поддаётся,
    # как автор собирался), --target-sprint — бегает.
    parser.add_argument("--scripted-target", action="store_true", help="охота на 'человека' вместо ботов")
    parser.add_argument("--target-sprint", action="store_true", help="'человек' бегает, а не ходит")
    parser.add_argument("--pillar-chance", type=float, default=0.5,
                        help="охота на 'человека': с этой вероятностью он строит столб, когда охотник рядом")
    parser.add_argument("--box-chance", type=float, default=0.0,
                        help="...а с этой — закрывается коробкой (стены в два блока вокруг себя)")
    # Решает прятаться, когда охотник ближе стольки блоков. Автор вживую
    # строится заранее — укрытие готово, когда боты подбегают; с 5 и даже 16
    # "человек" часто не достраивал, и охота на недостроенное проверяла
    # просто бег (у готового столба просто бег — 0.3 остановки в минуту,
    # учитель — 2.1).
    parser.add_argument("--hide-distance", type=float, default=48.0,
                        help="'человек' решает прятаться, когда охотник ближе стольки блоков (48 — сразу, укрытие готово)")
    # Охота на "человека" не дольше стольких секунд (0 — без предела): не
    # остановили — новая охота с целой ареной (застрявшая у коробки охота иначе
    # тянулась бы весь прогон, а проверке нужно много охот, а не одна).
    parser.add_argument("--hunt-seconds", type=float, default=0.0, help="предел охоты на 'человека', с")
    # DQfD: насколько Q действия учителя должен обгонять остальные (train.demo_margin,
    # в игре 1.0) и какая доля батча — его примеры (train.demo_fraction, 0.25).
    # Q мозга охоты — десятки, разрыв лучшего со вторым ~4.5: margin 1.0 там —
    # слабая подсказка, TD её легко перебивает.
    parser.add_argument("--demo-margin", type=float, default=None, help="margin DQfD для этого прогона")
    parser.add_argument("--demo-fraction", type=float, default=None, help="доля примеров учителя в батче")
    parser.add_argument("--rescue", action="store_true",
                        help="мост: учитель подсказывает сети в её же ошибке (замерла у края) — примеры там, где она ошибается")
    parser.add_argument("--fixed-kit", action="store_true",
                        help="охота: всем ровно start_blocks блоков и меч (по умолчанию — случайный набор у каждого)")
    parser.add_argument("--seed", type=int, default=None)
    # Снимки мозгов по ходу обучения: в охоте каждый круг дальше первого делал
    # хуже, и лучший мозг может оказаться посередине прогона — снимки потом
    # проверяются без учителей (--eval), в игру идёт лучший.
    parser.add_argument("--snapshots", type=Path, default=None,
                        help="папка: при каждой сводке класть туда копию мозгов (<минута игры>m/<задачка>)")
    parser.add_argument("--sim-dir", type=Path, default=None,
                        help="папка мозгов симуляции вместо data/sim (при первом запуске туда копируется data/sim/brains)")
    parser.add_argument("--install", nargs="*", metavar="TASK",
                        help="перенести мозги симуляции в игру и выйти (без имён — chase flee hunt bridge)")
    args = parser.parse_args()
    if args.install is not None:
        # walking в симуляции не учится (только основа моста) — без имён его не трогаем.
        return install(args.install or [task for task in TASKS if task != "walking"])

    if args.sim_dir is not None:
        default_brains = SIM_DIR / "brains"
        use_sim_dir(args.sim_dir.resolve())
        if not (SIM_DIR / "brains").exists() and default_brains.exists():
            shutil.copytree(default_brains, SIM_DIR / "brains")
            print(f"[sim] Мозги симуляции: начинаю с копии {default_brains}.")
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)  # старая просьба остановиться — не про этот запуск
    prepare_brains()
    CONFIG["train"]["learn_steps_per_tick"] = args.learn_ratio
    if args.demo_margin is not None:
        CONFIG["train"]["demo_margin"] = args.demo_margin
    if args.demo_fraction is not None:
        CONFIG["train"]["demo_fraction"] = args.demo_fraction
    rng = random.Random(args.seed)
    hunt = args.game == "hunt"
    bridge = args.game == "bridge"
    bedwars = args.game == "bedwars"
    scripted = hunt and args.scripted_target
    hunt_cfg = CONFIG["modules"].get("hunt", {})
    arena = SimArena(CONFIG, args.bots, rng, target_name=TARGET_NAME if scripted else None,
                     target_sprint=args.target_sprint, pillar_chance=args.pillar_chance if scripted else 0.0,
                     regen_seconds=hunt_cfg.get("regen_seconds", 1.0) if hunt else 1.0,
                     box_chance=args.box_chance if scripted else 0.0,
                     sword_sharpness=hunt_cfg.get("sword_sharpness", 0) if hunt else 0,
                     hide_distance=args.hide_distance, random_kit=hunt and not args.fixed_kit,
                     course=BridgeCourse(args.bots) if bridge else None, bedwars=bedwars)
    start_blocks = hunt_cfg.get("start_blocks", 0) if hunt else 0
    last_hunt = None  # (цель, начало) прошлой охоты — новая охота: арена целая, блоки выданы
    # Часы судей — игровое время симуляции, а не настоящее: "считает до
    # пяти", раунд, фора охоты — всё в секундах игры.
    clock = types.SimpleNamespace(time=lambda: arena.time)
    tag_game.time = clock
    hunt_game.time = clock
    bedwars_game.time = clock

    loop = ai_loop.AILoop(args.game)
    outbox = Outbox()
    loop.action_socket = outbox
    loop.server = SimServer(arena)
    loop.tag_game.direct_commands = True
    if args.eval:
        # Проверка ничего не пишет: автосохранение ai_loop (каждые
        # autosave_ticks состояний и на смерти бота) иначе перезаписывало бы
        # мозги — веса те же, но счётчик шагов (а с ним epsilon) уходил вперёд.
        loop.save = lambda: None
        loop._request_save = lambda tasks: None
    teacher_ids = set(range(1, args.teachers + 1))

    def teach(session, state: dict) -> dict | None:
        if session.id in teacher_ids:
            return teacher_actions(session, state, CONFIG, loop)
        if args.rescue and session.task_name == "bridge":
            return bridge_rescue(state)
        return None

    if teacher_ids or args.rescue:
        loop.teacher = teach

    def wants_route(bot_id: int) -> bool:
        session = loop.sessions.get(bot_id)
        return session is not None and session.task_name in ("chase", "hunt", "bedwars")

    def start_hunt() -> None:  # как !start (на "человека") или !start auto (на ботов)
        loop.handle_command({"type": "command", "cmd": "hunt_start", "target": TARGET_NAME if scripted else "auto",
                             "bot_id": 1, "via": "console"})

    restart_hunt_at = None
    hunt_started = False

    started = time.time()
    deadline = started + args.minutes * 60
    next_report = args.report
    ticks = learns = 0
    mode = "проверяю (без обучения)" if args.eval else "учу"
    game_name = (("Охота на человека" if scripted else "Охота на ботов") if hunt else "Мост" if bridge
                 else "Бедварс" if bedwars else "Салки")
    print(f"[sim] {game_name}: {args.bots} ботов (учитель ведёт {len(teacher_ids)}), {mode} {args.minutes:g} мин. "
          f"Мозги — {SIM_DIR / 'brains'}.")
    try:
        while time.time() < deadline:
            if ticks % 20 == 0 and STOP_FILE.exists():
                STOP_FILE.unlink(missing_ok=True)
                print("[sim] Попросили остановиться (data/sim/stop) — сохраняю и выхожу.")
                break
            if args.game_minutes > 0 and arena.time >= args.game_minutes * 60:
                break
            if hunt and loop.hunt.active_from is not None and (loop.hunt.target_name, loop.hunt.active_from) != last_hunt:
                last_hunt = (loop.hunt.target_name, loop.hunt.active_from)
                arena.new_hunt(start_blocks)
            states = arena.build_states(wants_route)
            if hunt and not hunt_started and len(loop.sessions) >= 2:
                hunt_started = True  # боты уже в игре — можно начинать
                start_hunt()
            if args.eval:
                for session in loop.sessions.values():
                    session.greedy = True  # без случайных действий — только выученное
            loop.handle_messages(states)
            arena.receive(outbox)
            arena.drive_scripted()
            arena.step()
            for name, killer in arena.take_scripted_deaths():
                # Как сообщение сервера о смерти: охота окончена, всем награда.
                text = f"{name} was slain by {killer}" if killer else f"{name} died"
                loop.handle_command({"type": "command", "cmd": "player_died", "name": name, "text": text,
                                     "bot_id": 1, "via": "console"})
                restart_hunt_at = arena.time + 4.0  # человек возродился — снова !start
            if restart_hunt_at is not None and arena.time >= restart_hunt_at:
                restart_hunt_at = None
                start_hunt()
            if (scripted and args.hunt_seconds > 0 and loop.hunt.active_from is not None
                    and arena.time - loop.hunt.active_from > args.hunt_seconds):
                start_hunt()  # время вышло — заново (арену вернёт new_hunt)
            ticks += 1
            # Кредит обучения копится с каждым состоянием (--learn-ratio),
            # выучиваем весь. В проверке не учимся и ничего не сохраняем.
            while not args.eval and loop.learn_one() is not None:
                learns += 1
            if arena.time >= next_report:
                next_report += args.report
                report(loop, arena, ticks, learns, started, outbox)
                if not args.eval:
                    loop.save()
                    loop._save_requested()  # потока обучения нет — сохраняем сами
                    if args.snapshots:
                        snapshot(loop, args.snapshots, arena.time)
    except KeyboardInterrupt:
        print("[sim] Остановлено.")
    report(loop, arena, ticks, learns, started, outbox)
    if not args.eval:
        loop.save()
        loop._save_requested()  # сохранить сразу: потока обучения здесь нет
        if args.snapshots:
            snapshot(loop, args.snapshots, arena.time)
    return 0


def snapshot(loop, folder: Path, game_seconds: float) -> None:
    """Копия только что сохранённых мозгов этого прогона — folder/<минута игры>m/<задачка>."""
    target = folder / f"{game_seconds / 60:05.1f}m"
    for task in loop.brains:
        shutil.copytree(SIM_DIR / "brains" / task, target / task, dirs_exist_ok=True)
    print(f"[sim] Снимок мозгов ({', '.join(loop.brains)}): {target}")


def report(loop, arena: SimArena, ticks: int, learns: int, started: float, outbox: Outbox) -> None:
    wall = max(time.time() - started, 1e-6)
    game = loop.tag_game
    minutes = max(arena.time / 60, 1e-6)
    print(f"[sim] игровое время {minutes:.1f} мин (быстрее настоящего в {arena.time / wall:.1f} раза), "
          f"тиков {ticks}, шагов обучения {learns} ({learns / wall:.1f}/с), раундов {game.rounds}, "
          f"осалили всего {game.tags} ({game.tags / minutes:.1f} в минуту игры); {game.describe(arena.time)}")
    hurrays = sum(1 for text in outbox.chat if text.startswith("Ура"))
    time_up = sum(1 for text in outbox.chat if text.startswith("Время вышло"))
    outbox.chat.clear()
    print(f"[sim] с прошлой сводки: поймали всех {hurrays} раз, время вышло {time_up} раз; "
          f"в охоте цель остановили всего {loop.hunt.kills} раз")
    if arena.stats:
        print("[sim] блоки с прошлой сводки: " + ", ".join(f"{name} {count}" for name, count in arena.stats.items()))
        arena.stats.clear()
    arena.count_hunt_time()
    if arena.hide_minutes:
        # Всего с начала прогона: по укрытию "человека" — сколько раз остановили
        # и за сколько минут охоты (столб и коробка — то, чему учим отдельно).
        print("[sim] охота по укрытиям (всего): " + ", ".join(
            f"{HIDE_NAMES.get(hide, hide)} {arena.hide_kills[hide]} за {minutes:.1f} мин"
            + (f" (не успели {arena.hide_timeouts[hide]})" if arena.hide_timeouts[hide] else "")
            for hide, minutes in sorted(arena.hide_minutes.items())))
    loop._report_metrics()


if __name__ == "__main__":
    sys.exit(main())
