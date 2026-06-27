package ai.slashh.relief

import android.content.Context
import java.security.MessageDigest

/**
 * Local, on-device preferences — NO server, NO network (preserves the
 * airplane-mode / privacy story). Stores a LOCAL account (credentials hashed,
 * kept only on this device), the onboarding flag, the user's enabled relief
 * set, the rotation index, and whether to also notify when relief triggers.
 */
class Prefs(context: Context) {

    private val sp = context.getSharedPreferences("slashh_prefs", Context.MODE_PRIVATE)

    // ---- Local account (on-device only) --------------------------------------
    val hasAccount: Boolean get() = sp.contains(KEY_EMAIL)
    val name: String get() = sp.getString(KEY_NAME, "") ?: ""

    fun createAccount(name: String, email: String, password: String) {
        sp.edit()
            .putString(KEY_NAME, name.trim())
            .putString(KEY_EMAIL, email.trim().lowercase())
            .putString(KEY_PWHASH, hash(password))
            .apply()
    }

    /** Validate a login against the locally stored credentials. */
    fun checkLogin(email: String, password: String): Boolean {
        val storedEmail = sp.getString(KEY_EMAIL, null) ?: return false
        val storedHash = sp.getString(KEY_PWHASH, null) ?: return false
        return storedEmail == email.trim().lowercase() && storedHash == hash(password)
    }

    private fun hash(s: String): String =
        MessageDigest.getInstance("SHA-256").digest(s.toByteArray())
            .joinToString("") { "%02x".format(it) }

    var onboarded: Boolean
        get() = sp.getBoolean(KEY_ONBOARDED, false)
        set(v) = sp.edit().putBoolean(KEY_ONBOARDED, v).apply()

    /** Enabled reliefs, in enum order. Defaults to all if nothing saved. */
    fun enabled(): List<ReliefType> {
        val keys = sp.getStringSet(KEY_ENABLED, null)
            ?: return ReliefType.entries.toList()
        val list = ReliefType.entries.filter { it.key in keys }
        return list.ifEmpty { ReliefType.entries.toList() }
    }

    fun setEnabled(types: Collection<ReliefType>) {
        sp.edit().putStringSet(KEY_ENABLED, types.map { it.key }.toSet()).apply()
    }

    var lastIndex: Int
        get() = sp.getInt(KEY_LAST_INDEX, -1)
        set(v) = sp.edit().putInt(KEY_LAST_INDEX, v).apply()

    /** Also post a system notification when relief triggers (in addition to in-app). */
    var notify: Boolean
        get() = sp.getBoolean(KEY_NOTIFY, true)
        set(v) = sp.edit().putBoolean(KEY_NOTIFY, v).apply()

    // ---- Per-user calibrated latch thresholds (null until set) ----------------
    val enterThreshold: Float? get() = sp.getFloat(KEY_ENTER, -1f).takeIf { it >= 0f }
    val releaseThreshold: Float? get() = sp.getFloat(KEY_RELEASE, -1f).takeIf { it >= 0f }

    fun saveThresholds(enter: Float, release: Float) {
        sp.edit().putFloat(KEY_ENTER, enter).putFloat(KEY_RELEASE, release).apply()
    }

    // ---- Per-user calibration (null until calibrated) -------------------------
    val calibrated: Boolean get() = sp.contains(KEY_CALM)
    val useModel: Boolean get() = sp.getBoolean(KEY_USE_MODEL, false)
    val calmAnchor: Float? get() = sp.getFloat(KEY_CALM, Float.NaN).takeIf { !it.isNaN() }
    val stressAnchor: Float? get() = sp.getFloat(KEY_STRESS, Float.NaN).takeIf { !it.isNaN() }

    fun saveCalibration(useModel: Boolean, calm: Float, stress: Float) {
        sp.edit()
            .putBoolean(KEY_USE_MODEL, useModel)
            .putFloat(KEY_CALM, calm).putFloat(KEY_STRESS, stress)
            .apply()
    }

    private companion object {
        const val KEY_ONBOARDED = "onboarded"
        const val KEY_ENABLED = "enabled_reliefs"
        const val KEY_LAST_INDEX = "last_relief_index"
        const val KEY_NOTIFY = "notify_on_relief"
        const val KEY_NAME = "acct_name"
        const val KEY_EMAIL = "acct_email"
        const val KEY_PWHASH = "acct_pwhash"
        const val KEY_CALM = "anchor_calm"
        const val KEY_STRESS = "anchor_stress"
        const val KEY_USE_MODEL = "use_model_signal"
        const val KEY_ENTER = "thr_enter"
        const val KEY_RELEASE = "thr_release"
    }
}
