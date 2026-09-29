// ZeroMQ-мост между Mineflayer-ботом (Node) и ИИ-контроллером (Python).
//
// Схема: два однонаправленных канала PUSH/PULL.
//   Node -> Python (порт node_to_py): PUSH, JSON-состояние каждого тика.
//   Python -> Node (порт py_to_node): Node делает PULL и получает команды действий.
//
// PUSH/PULL выбран вместо PUB/SUB, потому что PUB/SUB может молча терять
// сообщения при медленном подписчике, а здесь терять ни тики, ни действия нельзя.

const zmq = require('zeromq');

class ZmqBridge {
    constructor(config) {
        this.nodeToPyUrl = `tcp://127.0.0.1:${config.zmq.node_to_py}`;
        this.pyToNodeUrl = `tcp://127.0.0.1:${config.zmq.py_to_node}`;

        // Node CONNECT-ится, Python BIND-ится. Так Python можно поднимать
        // и перезапускать независимо, а Node будет переподключаться сам.
        this.stateSocket = null;  // PUSH: отправка состояний
        this.actionSocket = null; // PULL: приём действий
        this.starting = null;     // Promise первого подключения (см. start)
    }

    // Идемпотентен: start() зовёт каждый бот роя при спавне (и при
    // переподключении), а сокеты нужны одни на весь процесс. Раньше каждый
    // вызов создавал НОВУЮ пару сокетов, старые оставались подключёнными,
    // и Python-PUSH раздавал действия по кругу всем PULL-сокетам — большая
    // часть действий уходила в никем не читаемые сокеты.
    start() {
        if (!this.starting) this.starting = this.connect();
        return this.starting;
    }

    async connect() {
        this.stateSocket = new zmq.Push();
        await this.stateSocket.connect(this.nodeToPyUrl);

        this.actionSocket = new zmq.Pull();
        // Небольшой таймаут приёма: receive() rejects с EAGAIN, если сообщений
        // нет — так тиковый цикл не блокируется на пустом канале.
        this.actionSocket.receiveTimeout = 1;
        await this.actionSocket.connect(this.pyToNodeUrl);
        console.log(`[zmq] Состояния -> ${this.nodeToPyUrl}`);
        console.log(`[zmq] Действия <- ${this.pyToNodeUrl}`);
    }

    // Отправить состояние в Python. PUSH блокируется при полном буфере,
    // поэтому await обязателен.
    async sendState(state) {
        await this.stateSocket.send(JSON.stringify(state));
    }

    // Команда для Python (например, смена задачи из чата) — тем же каналом,
    // что и состояния, Python различает их по полю type.
    async sendCommand(command) {
        await this.stateSocket.send(JSON.stringify({ type: 'command', ...command }));
    }

    // Прочитать одно действие, если оно пришло (неблокирующе).
    // Возвращает разобанный JSON или null.
    async receiveAction() {
        try {
            const [msg] = await this.actionSocket.receive();
            return JSON.parse(msg.toString());
        } catch (err) {
            if (err.code === 'EAGAIN') return null; // просто нет сообщения
            throw err;
        }
    }

    async stop() {
        if (this.stateSocket) await this.stateSocket.close();
        if (this.actionSocket) await this.actionSocket.close();
        console.log('[zmq] Сокеты закрыты.');
    }
}

module.exports = { ZmqBridge };
