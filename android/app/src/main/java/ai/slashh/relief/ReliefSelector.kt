package ai.slashh.relief

/**
 * Picks the next relief to show from the user's enabled preferences.
 *
 * Rotates through the enabled set so the user gets variety across episodes
 * (rather than the same relief every time). Pure + deterministic so it's
 * unit-testable with no Android dependency.
 */
object ReliefSelector {

    /**
     * @param enabled   the user's opted-in reliefs (preference order)
     * @param lastIndex index of the relief shown last time (-1 if none yet)
     * @return the chosen relief and its index (persist the index for next time)
     */
    fun next(enabled: List<ReliefType>, lastIndex: Int): Pair<ReliefType, Int> {
        require(enabled.isNotEmpty()) { "no relief types enabled" }
        val idx = ((lastIndex + 1) % enabled.size + enabled.size) % enabled.size
        return enabled[idx] to idx
    }
}
