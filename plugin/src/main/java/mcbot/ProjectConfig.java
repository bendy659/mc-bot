package mcbot;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * config.json проекта — тот же файл, что читают js/config.js и
 * py/config.py (один конфиг на все стороны, AGENTS.md). Сервер запускается
 * из server/, проект — папкой выше; MCBOT_CONFIG — путь к другому конфигу
 * (как у Node и Python).
 */
final class ProjectConfig {
    final Path path;
    private final JsonObject root;

    private ProjectConfig(Path path, JsonObject root) {
        this.path = path;
        this.root = root;
    }

    static ProjectConfig load(Path serverDir) throws IOException {
        String override = System.getenv("MCBOT_CONFIG");
        Path path = override != null && !override.isBlank()
            ? Path.of(override)
            : serverDir.toAbsolutePath().normalize().resolveSibling("config.json");
        String text = Files.readString(path, StandardCharsets.UTF_8);
        return new ProjectConfig(path, JsonParser.parseString(text).getAsJsonObject());
    }

    /** Значение по пути "bot.count" (номер — элемент списка: "vision.resolution.0") или null. */
    JsonElement get(String dotted) {
        JsonElement node = root;
        for (String key : dotted.split("\\.")) {
            if (node == null) return null;
            if (node.isJsonObject()) {
                node = node.getAsJsonObject().get(key);
            } else if (node.isJsonArray() && key.chars().allMatch(Character::isDigit)) {
                int index = Integer.parseInt(key);
                node = index < node.getAsJsonArray().size() ? node.getAsJsonArray().get(index) : null;
            } else {
                return null;
            }
        }
        return node == null || node.isJsonNull() ? null : node;
    }

    String string(String dotted, String fallback) {
        JsonElement value = get(dotted);
        return value != null ? value.getAsString() : fallback;
    }

    int integer(String dotted, int fallback) {
        JsonElement value = get(dotted);
        return value != null ? value.getAsInt() : fallback;
    }

    double number(String dotted, double fallback) {
        JsonElement value = get(dotted);
        return value != null ? value.getAsDouble() : fallback;
    }

    boolean flag(String dotted, boolean fallback) {
        JsonElement value = get(dotted);
        return value != null ? value.getAsBoolean() : fallback;
    }

    String botUsername() {
        return string("bot.username", "AI");
    }

    String botSeparator() {
        return string("bot.separator", "_");
    }

    int botCount() {
        return Math.max(1, integer("bot.count", 1));
    }

    /**
     * bot.task_worlds — миры задачек: задачка -> имя мира ({"bridge": "mcbot_bridge"}).
     * Плагин создаёт их пустыми (TaskWorlds); обычный мир под задачки не трогается.
     */
    java.util.Map<String, String> taskWorlds() {
        java.util.Map<String, String> out = new java.util.LinkedHashMap<>();
        JsonElement worlds = get("bot.task_worlds");
        if (worlds != null && worlds.isJsonObject()) {
            for (var entry : worlds.getAsJsonObject().entrySet()) {
                out.put(entry.getKey(), entry.getValue().getAsString());
            }
        }
        return out;
    }

    /** bot.names — имена для показа ("Егор [AI_1]"), без повторов и пустых. */
    java.util.List<String> botNames() {
        java.util.List<String> out = new java.util.ArrayList<>();
        JsonElement names = get("bot.names");
        if (names != null && names.isJsonArray()) {
            for (JsonElement element : names.getAsJsonArray()) {
                String name = element.getAsString().trim();
                if (!name.isEmpty() && !out.contains(name)) out.add(name);
            }
        }
        return out;
    }
}
