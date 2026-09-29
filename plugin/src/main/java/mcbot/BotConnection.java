package mcbot;

import io.netty.channel.ChannelFutureListener;
import io.netty.channel.embedded.EmbeddedChannel;
import java.net.InetSocketAddress;
import java.util.function.Consumer;
import net.minecraft.network.Connection;
import net.minecraft.network.PacketListener;
import net.minecraft.network.ProtocolInfo;
import net.minecraft.network.protocol.Packet;
import net.minecraft.network.protocol.PacketFlow;
import org.jspecify.annotations.Nullable;

/**
 * Соединение бота-игрока: сети за ним нет. Всё, что сервер шлёт "клиенту"
 * бота (звуки, урон, сообщения), уходит в sink — так бот слышит и видит
 * ровно то, что услышал бы клиент mineflayer (js/hearing.js слушал те же
 * пакеты звуков).
 *
 * Канал — EmbeddedChannel: он "открыт", и сервер считает игрока подключённым;
 * настоящей записи в сокет нет.
 */
final class BotConnection extends Connection {
    private final Consumer<Packet<?>> sink;

    BotConnection(Consumer<Packet<?>> sink) {
        super(PacketFlow.SERVERBOUND);
        this.sink = sink;
        this.channel = new EmbeddedChannel();
        this.address = new InetSocketAddress("127.0.0.1", 0);
    }

    @Override
    public void send(Packet<?> packet, @Nullable ChannelFutureListener listener, boolean flush) {
        sink.accept(packet);
    }

    // Входящих пакетов нет: протоколы и обработчики канала не нужны.
    @Override
    public <T extends PacketListener> void setupInboundProtocol(ProtocolInfo<T> protocol, T packetListener) {
    }

    @Override
    public void setupOutboundProtocol(ProtocolInfo<?> protocol) {
    }

    @Override
    public void setReadOnly() {
    }

    @Override
    public void handleDisconnection() {
    }
}
