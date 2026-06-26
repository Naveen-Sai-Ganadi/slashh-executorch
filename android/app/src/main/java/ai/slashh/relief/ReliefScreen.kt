package ai.slashh.relief

/** Anything the router can surface as a relief (breathing overlay, games, …). */
interface ReliefScreen {
    fun show()
    fun hide()
}
