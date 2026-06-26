package ai.slashh.relief

/**
 * The relief activities a user can opt into. Each is shown full-screen when
 * sustained stress is detected; which one fires is chosen by [ReliefSelector]
 * from the user's enabled set (stored locally in [Prefs]).
 *
 * Keep this android-free so the selection logic stays unit-testable.
 */
enum class ReliefType(val key: String, val title: String, val emoji: String) {
    BREATHING("breathing", "Breathe", "🫁"),
    TIC_TAC_TOE("tictactoe", "Tic-tac-toe", "⭕"),
    SOUNDS("sounds", "Calming sounds", "🎧"),
    JOKES("jokes", "A quick laugh", "😄"),
    COLOR("color", "Color tap", "🎨");

    companion object {
        fun fromKey(k: String): ReliefType? = entries.firstOrNull { it.key == k }
    }
}
