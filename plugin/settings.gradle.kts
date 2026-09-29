// Плагин сервера для mc-bot: боты — настоящие игроки сервера (вместо
// клиентов mineflayer из js/), ИИ по-прежнему в Python (py/ai_loop.py).
pluginManagement {
    repositories {
        gradlePluginPortal()
        maven("https://repo.papermc.io/repository/maven-public/")
    }
}

rootProject.name = "mcbot-plugin"
