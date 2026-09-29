// Сборка плагина: ./gradlew deploy — собрать и положить в server/plugins
// (сервер подхватит при следующем запуске).
//
// paperweight-userdev даёт внутренности сервера (net.minecraft.*) с именами
// Mojang: без них не создать ботов настоящими игроками сервера. С 26.1 Paper
// сам работает на именах Mojang — перекладывать имена при сборке не нужно.

plugins {
    java
    id("io.papermc.paperweight.userdev") version "2.0.0-beta.23"
}

group = "mcbot"
version = "0.1.0"

repositories {
    mavenCentral()
    maven("https://repo.papermc.io/repository/maven-public/")
}

dependencies {
    // Ровно та сборка Paper, что в server/ (paper-26.1.2-74.jar).
    paperweight.paperDevBundle("26.1.2.build.74-stable")
    // ZeroMQ на чистой Java — связь с Python (py/ai_loop.py) тем же
    // протоколом, что у Node. На сервер его скачивает сам Paper
    // (plugin.yml: libraries), в jar плагина он не входит.
    compileOnly("org.zeromq:jeromq:0.6.0")
}

java {
    toolchain.languageVersion = JavaLanguageVersion.of(25)
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
    options.release = 25
}

tasks.processResources {
    filesMatching("plugin.yml") {
        expand("version" to project.version)
    }
}

// Работающий сервер держит jar плагина открытым: перезапись на ходу либо не
// пройдёт (Windows), либо уронит плагин на первом же ещё не загруженном
// классе. Поэтому, если плагин уже стоит, новый кладём в plugins/update —
// Paper сам подменит его при следующем запуске сервера (bukkit.yml:
// update-folder, сверяет по имени плагина из plugin.yml).
tasks.register<Copy>("deploy") {
    description = "Собрать плагин и положить в server/plugins (или в plugins/update, если он уже стоит)"
    dependsOn(tasks.jar)
    val plugins = layout.projectDirectory.dir("../server/plugins")
    val installed = plugins.asFile.listFiles { file -> file.name.startsWith("mcbot-plugin") && file.name.endsWith(".jar") }
    from(tasks.jar)
    into(if (installed.isNullOrEmpty()) plugins else plugins.dir("update"))
}
