package ai.slashh.relief

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ReliefSelectorTest {

    private val all = ReliefType.entries.toList()

    @Test fun firstPickIsIndexZero() {
        val (type, idx) = ReliefSelector.next(all, lastIndex = -1)
        assertEquals(0, idx)
        assertEquals(all[0], type)
    }

    @Test fun rotatesThroughTheEnabledSet() {
        val enabled = listOf(ReliefType.BREATHING, ReliefType.JOKES, ReliefType.SOUNDS)
        var last = -1
        val seen = mutableListOf<ReliefType>()
        repeat(enabled.size) {
            val (t, i) = ReliefSelector.next(enabled, last)
            seen.add(t); last = i
        }
        assertEquals(enabled, seen)            // visits each once, in order
    }

    @Test fun wrapsAround() {
        val enabled = listOf(ReliefType.BREATHING, ReliefType.JOKES)
        val (t, i) = ReliefSelector.next(enabled, lastIndex = 1)
        assertEquals(0, i)
        assertEquals(ReliefType.BREATHING, t)
    }

    @Test fun singleEnabledAlwaysReturnsIt() {
        val (t, i) = ReliefSelector.next(listOf(ReliefType.COLOR), lastIndex = 0)
        assertEquals(0, i)
        assertEquals(ReliefType.COLOR, t)
    }

    @Test fun emptyThrows() {
        assertThrows(IllegalArgumentException::class.java) {
            ReliefSelector.next(emptyList(), -1)
        }
    }
}
