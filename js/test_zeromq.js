// Smoke-тест ZMQ-канала без Minecraft-сервера: поднимает локальный PULL,
// отправляет через тот же класс ZmqBridge пару состояний и "действие"
// в обратную сторону, проверяет, что всё доходит. Запуск: npm run test-zmq

const { loadConfig } = require('./config');
const { ZmqBridge } = require('./zmq_bridge');

async function main() {
    const config = loadConfig();

    // Имитируем Python-сторону: bind на оба порта.
    const zmq = require('zeromq');
    const stateSink = new zmq.Pull();
    await stateSink.bind(`tcp://127.0.0.1:${config.zmq.node_to_py}`);
    const actionSource = new zmq.Push();
    await actionSource.bind(`tcp://127.0.0.1:${config.zmq.py_to_node}`);

    const bridge = new ZmqBridge(config);
    await bridge.start();
    await new Promise((r) => setTimeout(r, 300)); // даём connect-ам установиться

    const fakeState = {
        type: 'state', tick: 0,
        vision: { resolution: [2, 2], cells: [0.1, 0.2, 0.3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] },
        hearing: [0, 0, 0.5, 0, 0, 0, 0, 0],
        self: { x: 0, y: 64, z: 0, yaw: 0, pitch: 0, health: 20, on_ground: true },
        target: null, dead: false,
    };
    await bridge.sendState(fakeState);
    await actionSource.send(JSON.stringify({ type: 'action', name: 'walk_forward' }));

    const [stateMsg] = await stateSink.receive();
    const parsed = JSON.parse(stateMsg.toString());
    console.log('Состояние дошло: tick =', parsed.tick, 'cells =', parsed.vision.cells.length);

    const action = await bridge.receiveAction();
    console.log('Действие дошло:', action.name);

    await bridge.stop();
    stateSink.close();
    actionSource.close();
    console.log('OK: ZMQ-канал работает в обе стороны.');
}

main().catch((err) => { console.error(err); process.exit(1); });
