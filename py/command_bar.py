"""Строка команд лаунчера: те же команды, что в чате игры (!task, !setTarget,
!greedy...), но без мусора в чате — ответы ботов приходят в лог лаунчера.

Что умеет:
  - выбор "кому": всем ботам или конкретному AI_N;
  - автодополнение: при наборе всплывает список подходящих команд и их
    аргументов (задачи, player/position/none, номера ботов, ники игроков,
    которые уже вводились). Tab — подставить, ↑↓ — выбрать, Esc — скрыть;
  - подсветка: известная команда — зелёным, неизвестная — красным; под
    строкой — подсказка, как пользоваться текущей командой;
  - история: ↑↓, когда список подсказок скрыт, листают прошлые команды.

Отправка — через колбэк on_send(to, text): лаунчер пишет её в stdin Node
(см. runConsoleCommand в js/bot.js). "!" в начале можно не набирать.
Команда с "/" в начале — серверу (колбэк on_server: лаунчер шлёт её по
RCON, ответ сервера — в лог), с подсказками для частых команд.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from training_modules import MIX, TAG, TASK_ORDER

VALID_COLOR = "#1e7b34"
UNKNOWN_COLOR = "#c0392b"
MAX_SUGGESTIONS = 8

# Команды: имя -> (как пользоваться, что делает). Список — тот же, что в
# runCommand в js/bot.js.
COMMANDS = {
    "task": ("!task <задача> | all <задача>", "сменить задачу боту (или всему рою)"),
    "setTarget": ("!setTarget player <ник> | position <x y z> | none", "поставить цель (none — вернуть модулю)"),
    "greedy": ("!greedy [номер]", "бот без случайных действий — показать, чему научился"),
    "start": ("!start [ник]", "\"Останови меня\" (задача hunt): охота на тебя или на ник"),
    "stop": ("!stop", "пауза ИИ у всех ботов"),
    "resume": ("!resume", "продолжить после паузы"),
    "debug": ("!debug", "позиция, углы, здоровье бота"),
    "entities": ("!entities", "кто рядом с ботом"),
    "whatYouSee": ("!whatYouSee", "что бот видит и какая у него цель"),
    "help": ("!help", "список команд"),
}

# Частые команды сервера — для подсказок; сервер знает и остальные.
SERVER_COMMANDS = {
    "gamemode": "режим игры: spectator — смотреть сверху, survival — обычный",
    "time": "время: set day — день",
    "weather": "погода: clear — ясно",
    "tp": "телепорт: /tp <кто> <куда>",
    "difficulty": "сложность: peaceful — без мобов",
    "kill": "убить: @e[type=item] — убрать валяющиеся предметы",
    "give": "дать предмет: /give <кто> dirt 64",
    "clear": "очистить инвентарь",
    "effect": "эффект: /effect give <кто> ...",
    "list": "кто на сервере",
    "say": "сказать в чат от сервера",
    "op": "дать права оператора",
    "team": "роли в салках: join runners <ник> — убегать, join it <ник> — водить, leave <ник> — выйти",
    "scoreboard": "табло",
    "gamerule": "правила игры",
    "stop": "остановить сервер (мир сохранится)",
}
SERVER_ARGUMENTS = {
    "gamemode": ["spectator", "survival", "creative", "adventure"],
    "time": ["set day", "set noon", "set night"],
    "weather": ["clear", "rain", "thunder"],
    "difficulty": ["peaceful", "easy", "normal", "hard"],
    "kill": ["@e[type=item]", "@e[type=!player]"],
    "team": ["join runners", "join it", "leave", "list"],
}

TASK_HINTS = {
    MIX: "разные задачки разным ботам",
    "walking": "ходить к точкам",
    "looking": "следить взглядом",
    "follow": "следовать за целью",
    "gathering": "добыть норму ресурса",
    "crafting": "заглушка",
    TAG: "салки-заражение: осаленный тоже водит",
    "chase": "водить (роль в салках)",
    "flee": "убегать (роль в салках)",
    "hunt": "Останови меня: догнать и бить цель (после !start)",
}


class CommandBar(ttk.Frame):
    def __init__(self, master, root: tk.Tk, bot_count: int, on_send, on_server=None):
        super().__init__(master, padding=(0, 4, 0, 0))
        self.root = root
        self.bot_count = bot_count
        self.on_send = on_send
        self.on_server = on_server
        # Цвета текста в строке — от темы лаунчера (apply_theme).
        self.neutral_color = "black"
        self.valid_color = VALID_COLOR
        self.unknown_color = UNKNOWN_COLOR
        self.history: list[str] = []
        self.history_index: int | None = None
        self.known_players: list[str] = []
        self.suggestions: list[tuple[str, str]] = []  # (что подставить целиком, пояснение)
        self.navigated = False  # выбирали подсказку стрелками — Enter её подставит

        ttk.Label(self, text="Кому:").grid(row=0, column=0, padx=(0, 4))
        targets = ["все"] + [f"AI_{i}" for i in range(1, bot_count + 1)]
        self.target_var = tk.StringVar(value="все")
        ttk.Combobox(self, textvariable=self.target_var, values=targets, state="readonly", width=8).grid(
            row=0, column=1, padx=(0, 6))

        self.text_var = tk.StringVar()
        self.entry = tk.Entry(self, textvariable=self.text_var, font=("Consolas", 10))
        self.entry.grid(row=0, column=2, sticky="ew")
        ttk.Button(self, text="Отправить", command=self.send).grid(row=0, column=3, padx=(6, 0))
        self.columnconfigure(2, weight=1)

        self.hint_var = tk.StringVar()
        self.hint_label = ttk.Label(self, textvariable=self.hint_var, foreground="#777777", font=("Consolas", 9))
        self.hint_label.grid(row=1, column=2, columnspan=2, sticky="w")

        # Всплывающий список подсказок — поверх окна, над строкой ввода.
        self.popup = tk.Listbox(root, font=("Consolas", 10), activestyle="none", exportselection=False,
                                highlightthickness=1, highlightbackground="#999999")
        self.popup.bind("<ButtonRelease-1>", lambda event: self.accept())

        self.entry.bind("<KeyRelease>", self.on_key_release)
        self.entry.bind("<Tab>", lambda event: self.accept() or "break")
        self.entry.bind("<Down>", lambda event: self.move(+1) or "break")
        self.entry.bind("<Up>", lambda event: self.move(-1) or "break")
        self.entry.bind("<Return>", self.on_return)
        self.entry.bind("<Escape>", lambda event: self.hide_popup())
        self.entry.bind("<FocusOut>", lambda event: self.after(150, self.hide_popup))
        self.update_state()

    # --- отправка и история --------------------------------------------------

    def target(self) -> str:
        value = self.target_var.get()
        return "all" if value == "все" else value.rsplit("_", 1)[1]  # "AI_3" -> "3"

    def send(self) -> None:
        text = self.text_var.get().strip()
        if not text:
            return
        if text.startswith("/") and self.on_server is not None:
            self.on_server(text[1:])  # серверу — "кому" тут не важно
        else:
            if not text.startswith("!"):
                text = "!" + text
            self.remember_player(text)
            self.on_send(self.target(), text)
        if not self.history or self.history[-1] != text:
            self.history.append(text)
        self.history_index = None
        self.text_var.set("")
        self.hide_popup()
        self.update_state()

    def remember_player(self, text: str) -> None:
        """Ник из "!setTarget player <ник>" запоминаем для автодополнения."""
        parts = text.lstrip("!").split()
        if len(parts) >= 3 and parts[0] == "setTarget" and parts[1] == "player" and parts[2] not in self.known_players:
            self.known_players.append(parts[2])

    def on_return(self, event):
        if self.popup.winfo_ismapped() and self.navigated:
            self.accept()
        else:
            self.send()
        return "break"

    # --- подсказки ----------------------------------------------------------

    def on_key_release(self, event) -> None:
        if event.keysym in ("Up", "Down", "Tab", "Return", "Escape"):
            return
        self.history_index = None
        self.navigated = False
        self.update_state()

    def update_state(self) -> None:
        text = self.text_var.get()
        tokens = text.split(" ")
        name = tokens[0].lstrip("!")

        if text.startswith("/"):
            # Команда серверу: знакомая — зелёным; незнакомая — не ошибка
            # (сервер знает куда больше команд), просто без подсказки.
            server_name = tokens[0][1:]
            known = server_name in SERVER_COMMANDS
            self.entry.configure(foreground=self.valid_color if known else self.neutral_color)
            self.hint_var.set(f"серверу: {SERVER_COMMANDS[server_name]}" if known
                              else "команда серверу (через RCON), ответ — в лог")
            self.suggestions = self.server_suggestions(tokens)
            self.show_popup()
            return

        # Подсветка и подсказка по использованию.
        if not name:
            self.entry.configure(foreground=self.neutral_color)
            self.hint_var.set("Tab — дополнить, ↑↓ — выбрать, Enter — отправить. «!» можно не писать, «/» — команда серверу.")
        elif name in COMMANDS:
            self.entry.configure(foreground=self.valid_color)
            usage, description = COMMANDS[name]
            self.hint_var.set(f"{usage} — {description}")
        elif len(tokens) == 1 and any(c.lower().startswith(name.lower()) for c in COMMANDS):
            self.entry.configure(foreground=self.neutral_color)  # ещё набирается
            self.hint_var.set("")
        else:
            self.entry.configure(foreground=self.unknown_color)
            self.hint_var.set(f"Нет такой команды: {name}")

        self.suggestions = self.compute_suggestions(tokens)
        self.show_popup()

    def compute_suggestions(self, tokens: list[str]) -> list[tuple[str, str]]:
        if len(tokens) == 1:
            prefix = tokens[0].lstrip("!").lower()
            if not prefix:
                return []
            return [(f"!{name} ", description) for name, (_, description) in COMMANDS.items()
                    if name.lower().startswith(prefix) and name.lower() != prefix]

        name = tokens[0].lstrip("!")
        done_args, prefix = tokens[1:-1], tokens[-1]
        head = " ".join(tokens[:-1]) + " "
        options = self.argument_options(name, done_args)
        return [(head + value + " ", hint) for value, hint in options
                if value.lower().startswith(prefix.lower()) and value != prefix]

    def apply_theme(self, palette: dict) -> None:
        """Цвета от темы лаунчера (py/launcher.py: THEMES)."""
        self.neutral_color = palette["field_fg"]
        self.valid_color = palette["valid"]
        self.unknown_color = palette["unknown"]
        self.entry.configure(background=palette["field"], insertbackground=palette["field_fg"],
                             selectbackground=palette["select"], selectforeground=palette["field_fg"],
                             highlightbackground=palette["border"], highlightcolor=palette["select"])
        self.popup.configure(background=palette["field"], foreground=palette["field_fg"],
                             selectbackground=palette["select"], selectforeground=palette["field_fg"],
                             highlightbackground=palette["border"])
        self.hint_label.configure(foreground=palette["muted"])
        self.update_state()

    def server_suggestions(self, tokens: list[str]) -> list[tuple[str, str]]:
        name = tokens[0][1:]
        if len(tokens) == 1:
            if not name:
                return [(f"/{command} ", hint) for command, hint in SERVER_COMMANDS.items()]
            return [(f"/{command} ", hint) for command, hint in SERVER_COMMANDS.items()
                    if command.startswith(name.lower()) and command != name]
        typed = " ".join(tokens[1:])
        return [(f"/{name} {value} ", "") for value in SERVER_ARGUMENTS.get(name, [])
                if value.startswith(typed) and value != typed.strip()]

    def argument_options(self, name: str, done_args: list[str]) -> list[tuple[str, str]]:
        """Варианты для следующего аргумента команды name."""
        tasks = [(task, TASK_HINTS.get(task, "")) for task in [MIX, TAG, *TASK_ORDER]]
        if name == "task":
            if not done_args:
                return [("all", "всему рою"), *tasks]
            if done_args == ["all"]:
                return tasks
        if name == "setTarget":
            if not done_args:
                return [("player", "следить за игроком"), ("position", "точка x y z"), ("none", "снять цель")]
            if done_args == ["player"]:
                return [(player, "игрок") for player in self.known_players]
        if name == "greedy" and not done_args:
            return [(str(i), f"бот AI_{i}") for i in range(1, self.bot_count + 1)]
        return []

    def show_popup(self) -> None:
        if not self.suggestions:
            self.hide_popup()
            return
        shown = self.suggestions[:MAX_SUGGESTIONS]
        self.popup.delete(0, tk.END)
        for full, hint in shown:
            last = full.strip().split(" ")[-1]
            self.popup.insert(tk.END, f"{last:<14} {hint}")
        self.popup.selection_clear(0, tk.END)
        self.popup.selection_set(0)
        self.popup.configure(height=len(shown))

        # Над строкой ввода, в координатах главного окна.
        self.root.update_idletasks()
        x = self.entry.winfo_rootx() - self.root.winfo_rootx()
        y = self.entry.winfo_rooty() - self.root.winfo_rooty()
        row_height = 18
        self.popup.place(x=x, y=y - row_height * len(shown) - 4, width=max(self.entry.winfo_width() // 2, 320))
        self.popup.lift()

    def hide_popup(self) -> None:
        self.popup.place_forget()
        self.navigated = False

    def move(self, step: int) -> None:
        if self.popup.winfo_ismapped():
            selected = self.popup.curselection()
            index = (selected[0] if selected else 0) + step
            index = max(0, min(index, self.popup.size() - 1))
            self.popup.selection_clear(0, tk.END)
            self.popup.selection_set(index)
            self.navigated = True
            return
        # Подсказок нет — листаем историю команд.
        if not self.history:
            return
        if self.history_index is None:
            self.history_index = len(self.history)
        self.history_index = max(0, min(self.history_index + step, len(self.history)))
        value = self.history[self.history_index] if self.history_index < len(self.history) else ""
        self.text_var.set(value)
        self.entry.icursor(tk.END)
        self.update_state()
        self.hide_popup()

    def accept(self) -> None:
        if not self.popup.winfo_ismapped() or not self.suggestions:
            return
        selected = self.popup.curselection()
        full, _ = self.suggestions[selected[0] if selected else 0]
        self.text_var.set(full)
        self.entry.icursor(tk.END)
        self.entry.focus_set()
        self.navigated = False
        self.update_state()
