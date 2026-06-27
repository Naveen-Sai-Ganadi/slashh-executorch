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
    // The QNN HTP path needs the native libs on the *filesystem*, not zipped inside
    // the APK. The Hexagon skel (libQnnHtpV79Skel.so) is loaded onto the cDSP by the
    // fastrpc DSP loader, which can only open a real file path — it cannot read the
    // skel from `base.apk!/lib/...`. With the modern default (extractNativeLibs=false)
    // the skel never hits disk, so QNN fails with "Failed to load skel, error: 4000"
    // and the delegate init aborts. Legacy packaging extracts every .so to the app's
    // nativeLibraryDir (which fastrpc auto-adds to the DSP search path), letting the
    // skel load onto the Hexagon NPU.
    packaging {
        jniLibs {
            useLegacyPackaging = true
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
        // A local AAR (files(...)) carries NO transitive Maven deps, so the two
        // runtime libs the published executorch-android:1.2.0 POM declares must be
        // added by hand — without them org.pytorch.executorch.Module.<clinit>
        // throws NoClassDefFoundError (NativeLoader) and the app crashes on launch.
        implementation("com.facebook.fbjni:fbjni:0.7.0")
        implementation("com.facebook.soloader:nativeloader:0.10.5")
    } else {
        implementation("org.pytorch:executorch-android:1.2.0")
    }

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")   // JSON in local JVM unit tests
}
