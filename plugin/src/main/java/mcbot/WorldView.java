package mcbot;

import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.LevelChunk;

/**
 * Блоки мира для лучей и маршрутов — только из УЖЕ загруженных чанков:
 * обычный getBlockState на незагруженном чанке грузит его синхронно (и
 * тормозит сервер). Незагруженное — null: у mineflayer это "чанк не пришёл"
 * (зрению — воздух, маршруту — нельзя).
 */
final class WorldView {
    final ServerLevel level;
    private final BlockPos.MutableBlockPos cursor = new BlockPos.MutableBlockPos();
    private int lastChunkX = Integer.MIN_VALUE;
    private int lastChunkZ = Integer.MIN_VALUE;
    private LevelChunk lastChunk;

    WorldView(ServerLevel level) {
        this.level = level;
    }

    /** Состояние блока или null, если чанк не загружен. */
    BlockState state(int x, int y, int z) {
        int chunkX = x >> 4;
        int chunkZ = z >> 4;
        if (chunkX != lastChunkX || chunkZ != lastChunkZ) {
            lastChunk = level.getChunkIfLoaded(chunkX, chunkZ);
            lastChunkX = chunkX;
            lastChunkZ = chunkZ;
        }
        if (lastChunk == null) return null;
        return lastChunk.getBlockState(cursor.set(x, y, z));
    }

    /** Сведения о блоке или null (чанк не загружен). */
    BlockInfo info(int x, int y, int z) {
        BlockState state = state(x, y, z);
        return state == null ? null : BlockInfo.of(state);
    }
}
