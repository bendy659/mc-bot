package mcbot;

import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

/**
 * Связь с Python (py/ai_loop.py) — как js/zmq_bridge.js: два канала
 * PUSH/PULL. Состояния и команды — на node_to_py (Python там BIND-ится),
 * действия, цели и ответы — с py_to_node. Плагин только CONNECT-ится:
 * Python можно запускать и перезапускать когда угодно.
 *
 * Всё неблокирующее и только из главного потока сервера: ждать Python
 * сервер не должен (тик сервера — 50 мс).
 */
final class Bridge implements AutoCloseable {
    private final ZContext context = new ZContext();
    private final ZMQ.Socket states;
    private final ZMQ.Socket actions;

    Bridge(int nodeToPy, int pyToNode) {
        states = context.createSocket(SocketType.PUSH);
        states.setLinger(0);
        // Python не слушает — состояния не копятся (иначе при его запуске
        // пришла бы лавина устаревших), а просто не отправляются.
        states.setImmediate(true);
        states.connect("tcp://127.0.0.1:" + nodeToPy);
        actions = context.createSocket(SocketType.PULL);
        actions.setLinger(0);
        actions.connect("tcp://127.0.0.1:" + pyToNode);
    }

    /** Отправить JSON (состояние или команду); false — Python не на связи. */
    boolean send(String json) {
        return states.send(json, ZMQ.DONTWAIT);
    }

    /** Следующее сообщение от Python или null. */
    String receive() {
        return actions.recvStr(ZMQ.DONTWAIT);
    }

    @Override
    public void close() {
        context.close();
    }
}
