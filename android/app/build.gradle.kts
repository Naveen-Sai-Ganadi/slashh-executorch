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

    // ExecuTorch Android runtime (M7). Align this AAR with the pip `executorch`
    // version that produced the .pte (1.2.0) and bundle the QNN backend libs for
    // the Hexagon-NPU path. If no matching Maven artifact is available, build the
    // AAR from the ExecuTorch repo (`extension/android`) and drop it in libs/.
    implementation("org.pytorch:executorch-android:0.5.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")   // JSON in local JVM unit tests
}
