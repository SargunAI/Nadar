plugins {
    id("java")
}

group = "ai.sargun"
version = project.findProperty("VERSION_NAME") ?: "0.1.0-SNAPSHOT"

tasks.test {
    useJUnitPlatform()
}