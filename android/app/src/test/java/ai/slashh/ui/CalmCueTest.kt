package ai.slashh.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM tests for the calming-cue trigger/rate-limit logic (M9). Clock is
 * injected, so every timing rule is pinned down deterministically with no
 * device and no sleeps.
 */
class CalmCueTest {

    @Test fun doesNotTriggerOnABriefSpike() {
        val cue = CalmCue(sustainMs = 4_000, cooldownMs = 60_000)
        assertFalse(cue.onState(stressed = true, nowMs = 0))       // just became stressed
        assertFalse(cue.onState(stressed = true, nowMs = 3_999))   // under sustain window
        assertFalse(cue.onState(stressed = false, nowMs = 4_500))  // cleared before sustain
    }

    @Test fun triggersOnceStressIsSustained() {
        val cue = CalmCue(sustainMs = 4_000, cooldownMs = 60_000)
        assertFalse(cue.onState(true, 0))
        assertTrue(cue.onState(true, 4_000))   // sustained -> show
        assertTrue(cue.justTriggered)          // rising edge -> haptic
        assertTrue(cue.onState(true, 5_000))   // stays up
        assertFalse(cue.justTriggered)         // not a fresh trigger
    }

    @Test fun staysQuietForTheEpisodeAfterDismiss() {
        val cue = CalmCue(sustainMs = 4_000, cooldownMs = 60_000)
        cue.onState(true, 0)
        assertTrue(cue.onState(true, 4_000))
        cue.dismiss(4_500)
        assertFalse(cue.isVisible)
        // still stressed, but dismissed this episode -> no re-show
        assertFalse(cue.onState(true, 10_000))
        assertFalse(cue.onState(true, 70_000))
    }

    @Test fun cooldownGatesTheNextEpisode() {
        val cue = CalmCue(sustainMs = 4_000, cooldownMs = 60_000)
        cue.onState(true, 0)
        assertTrue(cue.onState(true, 4_000))   // shown at t=4000
        cue.dismiss(4_500)
        // stress clears (episode ends) then returns soon — cooldown not elapsed
        assertFalse(cue.onState(false, 5_000))
        cue.onState(true, 6_000)
        assertFalse(cue.onState(true, 12_000)) // sustained again, but within cooldown of 4000
        // new episode well past cooldown -> shows again
        assertFalse(cue.onState(false, 70_000))
        cue.onState(true, 71_000)
        assertTrue(cue.onState(true, 75_000))
    }

    @Test fun resetClearsEverything() {
        val cue = CalmCue(sustainMs = 4_000, cooldownMs = 60_000)
        cue.onState(true, 0)
        cue.onState(true, 4_000)
        cue.reset()
        assertFalse(cue.isVisible)
        assertFalse(cue.onState(true, 4_100)) // sustain window restarts from reset
    }
}
