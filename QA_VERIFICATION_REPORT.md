# QA Verification Report - Slashh AI

## 1. Executive Summary
The Slashh AI app is in a **functional demo state** but is **not production-ready**. 
*   **Success:** Live NPU-based scoring via `npu_helper.sh` is confirmed and working on the Galaxy S25 Ultra.
*   **Success:** High stress simulation correctly triggers local notifications and automatically opens relief screens.
*   **Blocker:** **Relief Persistence Failure.** The relief screens (e.g., Affirmations, Color Toy) appear but are frequently interrupted or fail to remain active. The app sometimes resets to the dashboard or gets stuck in a semi-navigated state.
*   **Critical Issue:** **Calibration is Broken.** The "Calibrate" button often fails to navigate to the calibration screen, sometimes closing the app or doing nothing.
*   **Critical Issue:** **Navigation Instability.** Navigating to Settings or Technical Status can lead to stuck UI states where the back button fails to return to the dashboard.

## 2. Test Environment
*   **Device:** Samsung Galaxy S25 Ultra (Snapdragon 8 Elite)
*   **Android Version:** 15 (One UI 7.0)
*   **Branch:** `sk/demo-integration`
*   **Package Name:** `ai.slashh`
*   **Build Type:** Debug APK (Java 17)
*   **Offline Mode:** Verified working (No internet permission in Manifest).

## 3. What Works
*   **Install & Launch:** Debug APK installs and launches without startup crashes.
*   **Auth Bypass:** Successfully bypasses login to reach the dashboard.
*   **Permissions:** Requests and respects `RECORD_AUDIO` and `POST_NOTIFICATIONS`.
*   **Live Listening:** Background service (`StressMonitorService`) correctly holds the microphone.
*   **VAD:** Real-time Voice Activity Detection is functional (seen in logs and UI).
*   **NPU Helper:** `npu_helper.sh` out-of-process scoring is active and serving inference requests.
*   **Notifications:** High-stress events trigger local notifications immediately.
*   **Auto Relief Trigger:** High stress (simulated) correctly triggers the opening of a relief screen.

## 4. Critical Blunders / Must Fix

### Issue 1: Relief Screen Not Persistent (High Priority)
*   **Severity:** Blocker
*   **Steps:** Tap "Simulate" to trigger high stress.
*   **Expected:** Relief screen opens and stays until manually dismissed.
*   **Actual:** Screen opens but sometimes resets to dashboard or navigates away automatically when new scoring data arrives.
*   **Likely Cause:** `src/lib/slashh-store.tsx` or `SlashhApp.tsx` state updates from background scoring are triggering re-renders that reset the navigation state.

### Issue 2: Calibration Navigation Broken
*   **Severity:** Critical
*   **Steps:** Tap "Calibrate" on the Dashboard.
*   **Expected:** Navigate to the 12-sample calibration flow.
*   **Actual:** Sometimes does nothing; sometimes closes the app; sometimes UI remains on dashboard while logs show internal state change.
*   **Evidence:** Repeated tap attempts resulted in app closure or no visual transition.

### Issue 3: Settings/Technical Status Navigation Stuck
*   **Severity:** High
*   **Steps:** Open Settings -> Expand Technical Status -> Try to go back to Dashboard.
*   **Expected:** Back button returns to Dashboard.
*   **Actual:** User gets stuck in the expanded Settings view. System back and UI back both fail to return to dashboard without an app restart.

## 5. Important Improvements / Should Fix
*   **UI Mode Consistency:** The "Local mode" label is static; it should clearly indicate "Snapdragon NPU" or "CPU Fallback" based on real backend status.
*   **Permission Denial Handling:** Denying microphone permission currently leaves the dashboard in a "Listening" state but with 0% stress, which is confusing. It should show a "Mic Required" warning.
*   **Simulate Button Utility:** The "Simulate" button is great for testing but should be moved to a hidden debug menu for production.

## 6. Button-by-Button Test Matrix
| Screen | Button/Control | Expected | Actual | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dashboard | Pause | Pause listening | Pauses UI updates | PASS | |
| Dashboard | Simulate | Trigger high stress | Triggers relief | PASS | Reliability of relief opening is ~80% |
| Dashboard | Reset | Reset stress score | Resets to 0% | PASS | |
| Dashboard | Relief | Open relief options | Opens relief | PASS | |
| Dashboard | Calibrate | Start calibration | Fails to navigate | **FAIL** | Critical navigation bug |
| Settings | Back Icon | Exit to Dashboard | Fails/Stuck | **FAIL** | Navigation trap |
| Relief | Close (X) | Exit to Dashboard | Exits correctly | PASS | |
| Relief | Next card | Show new content | Updates content | PASS | Tested in Affirmations |

## 7. Flow-by-Flow Test Matrix
| Flow | Expected | Actual | Status |
| :--- | :--- | :--- | :--- |
| **Start/Stop Listening** | Service toggles mic | UI reflects state | PASS | |
| **Stress Detection** | Notification + Auto-Relief | Notification OK, Relief unstable | **PARTIAL** | |
| **Background Listening**| Mic held while Home | Notification fires | PASS | |
| **Calibration** | 12 sample capture | Screen unreachable | **FAIL** | |
| **Persistence** | Relief stays open | Auto-resets to Dash | **FAIL** | |

## 8. Logs / Evidence
*   **NPU Helper Active:**
    `Slashh: monitor scorer: WavLM via NPU helper (probe OK, channel /storage/emulated/0/Android/data/ai.slashh/files)`
*   **Stress Simulation Trigger:**
    `Slashh: simulate stress event triggered`
*   **Notification Sent:**
    `NotificationManager: ai.slashh: notify(1001, null, Notification(channel=slashh_relief ...))`

## 9. Recommended Fix Plan

### Fix Immediately (Before Demo)
1.  **Stabilize Navigation:** Fix the `Calibrate` and `Settings -> Back` navigation routes. Ensure `react-router` or the custom navigator isn't losing history.
2.  **Lock Relief State:** Modify `SlashhApp.tsx` to ignore incoming stress score updates while a Relief session is active, preventing auto-navigation back to Dashboard.

### Fix Before Real User Testing
1.  **Refactor Store:** Clean up `slashh-store.tsx` to handle "Background" vs "Foreground" state transitions more gracefully.
2.  **Model Mode Label:** Connect the "Local mode" UI badge to the real native backend status.

### Later Polish
1.  **Haptics:** Add haptic feedback for stress detection.
2.  **Calibration UI:** Improve the recording progress visualization.
