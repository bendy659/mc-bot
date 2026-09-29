"""Консоль сервера по сети (RCON): Python шлёт серверу команды напрямую —
без op у бота и без флуда в чат (сервер кикает за флуд всех, кроме op).

Протокол — Source RCON (его понимает любой сервер Minecraft): пакет =
длина, id запроса, тип, текст в UTF-8 и два нулевых байта. Сначала вход
паролем (тип 3), потом команды (тип 2), ответ приходит с тем же id.
Включается в server.properties: enable-rcon, rcon.port, rcon.password.
"""

from __future__ import annotations

import socket
import struct
import threading

LOGIN, COMMAND = 3, 2


class RconError(Exception):
    pass


class Rcon:
    def __init__(self, host: str, port: int, password: str, timeout: float = 3.0):
        self.host = host
        self.port = port
        self.password = password
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.next_id = 1
        self.lock = threading.Lock()

    def connect(self) -> None:
        self.close()
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        request_id = self._send(LOGIN, self.password)
        reply_id, _ = self._receive()
        if reply_id != request_id:  # сервер отвечает id -1, если пароль не тот
            self.close()
            raise RconError("сервер не принял пароль RCON (rcon.password в server/server.properties и config.json)")

    def command(self, text: str) -> str:
        """Выполнить команду (без "/") и вернуть ответ сервера. Соединение
        упало — одна попытка переподключиться."""
        with self.lock:
            for attempt in (1, 2):
                try:
                    if self.sock is None:
                        self.connect()
                    request_id = self._send(COMMAND, text)
                    reply_id, body = self._receive()
                    while reply_id != request_id:  # хвост прошлого ответа — пропускаем
                        reply_id, body = self._receive()
                    return body
                except OSError:
                    self.close()
                    if attempt == 2:
                        raise
        return ""

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _send(self, kind: int, text: str) -> int:
        request_id = self.next_id
        self.next_id += 1
        body = text.encode("utf-8") + b"\x00\x00"
        self.sock.sendall(struct.pack("<iii", len(body) + 8, request_id, kind) + body)
        return request_id

    def _receive(self) -> tuple[int, str]:
        (length,) = struct.unpack("<i", self._read(4))
        data = self._read(length)
        request_id, _kind = struct.unpack("<ii", data[:8])
        return request_id, data[8:-2].decode("utf-8", errors="replace")

    def _read(self, size: int) -> bytes:
        data = b""
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("RCON: сервер закрыл соединение")
            data += chunk
        return data
