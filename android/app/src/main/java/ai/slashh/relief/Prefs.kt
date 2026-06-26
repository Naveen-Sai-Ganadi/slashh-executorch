package ai.slashh.relief

import android.content.Context

/**
 * Local, on-device preferences — NO account, NO network (preserves the
 * airplane-mode / privacy story). Stores the onboarding flag, the user's
 * enabled relief set, the rotation index, and whether to also post a
 * notification when relief triggers.
 */
class Prefs(context: Context) {

    private val sp = context.getSharedPreferences("slashh_prefs", Context.MODE_PRIVATE)

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

    private companion object {
        const val KEY_ONBOARDED = "onboarded"
        const val KEY_ENABLED = "enabled_reliefs"
        const val KEY_LAST_INDEX = "last_relief_index"
        const val KEY_NOTIFY = "notify_on_relief"
    }
}
