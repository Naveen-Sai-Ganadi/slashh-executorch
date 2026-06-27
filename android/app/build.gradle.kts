plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.slashh"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.slashh"
        minSdk = 26                 // plan §6: min SDK ~26+
        targetSdk = 35              // Android 15 / One UI 7 (S25 Ultra)
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    // The model ships as an asset; don't let the build compress the .pte.
    androidResources {
        noCompress += "pte"
    }
    testOptions {
        unitTests {
            isIncludeAndroidResources = true
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.2")

    // ExecuTorch Android runtime (M7).
    // - Default: the Maven AAR (XNNPACK/CPU) — what runs today.
    // - NPU: drop a QNN-enabled AAR at app/libs/executorch-qnn.aar (built per
    //   docs/NPU-ON-DEVICE.md) + the QNN runtime .so's in
    //   src/main/jniLibs/arm64-v8a/, and it's used automatically (no code change).
    val qnnAar = file("libs/executorch-qnn.aar")
    if (qnnAar.exists()) {
        implementation(files(qnnAar))
        // A local files() AAR has no POM, so ExecuTorch's transitive native-loader
        // deps aren't pulled — declare them explicitly or NativeLoader is missing
        // and Module fails to load on-device (crash). Versions per the Maven POM.
        implementation("com.facebook.fbjni:fbjni:0.7.0")
        implementation("com.facebook.soloader:nativeloader:0.10.5")
    } else {
        implementation("org.pytorch:executorch-android:1.2.0")
    }

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")   // JSON in local JVM unit tests
}
