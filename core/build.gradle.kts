group = "ai.sargun"
version = project.findProperty("versionOverride") ?: project.property("VERSION_NAME") as String

plugins {
    id("java")
    id("me.champeau.jmh") version "0.7.3"
}

// * JMH CONFIG *
jmh {
    jvmArgs.addAll(listOf("--enable-preview", "--add-modules", "jdk.incubator.vector"))
    warmupIterations = 2
    iterations = 2
    fork = 2
}
// ***

tasks.test {
    useJUnitPlatform()
}

group = "com.hardik"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:6.0.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}
