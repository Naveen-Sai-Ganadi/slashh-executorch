package ai.slashh.relief

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build

/**
 * Posts a LOCAL notification when relief triggers (no network — purely an
 * on-device nudge so the user is prompted even if the app is backgrounded).
 * Tapping it reopens the app, which shows the in-app relief.
 */
object ReliefNotifier {

    private const val CHANNEL_ID = "slashh_relief"
    private const val NOTIF_ID = 1001
    const val EXTRA_SHOW_RELIEF = "show_relief"

    fun ensureChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val mgr = context.getSystemService(NotificationManager::class.java)
            if (mgr.getNotificationChannel(CHANNEL_ID) == null) {
                val ch = NotificationChannel(
                    CHANNEL_ID, "Relief nudges", NotificationManager.IMPORTANCE_HIGH
                ).apply { description = "Gentle prompts when rising stress is detected" }
                mgr.createNotificationChannel(ch)
            }
        }
    }

    fun notify(context: Context, relief: ReliefType) {
        ensureChannel(context)
        val open = Intent(context, Class.forName("ai.slashh.MainActivity")).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra(EXTRA_SHOW_RELIEF, relief.key)
        }
        val pi = PendingIntent.getActivity(
            context, 0, open,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val n = androidx.core.app.NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_lock_idle_alarm)
            .setContentTitle("Slashh AI")
            .setContentText("Stress detected. Starting a calming session (${relief.title.lowercase()} ${relief.emoji}).")
            .setPriority(androidx.core.app.NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .setContentIntent(pi)
            .build()
        try {
            androidx.core.app.NotificationManagerCompat.from(context).notify(NOTIF_ID, n)
        } catch (_: SecurityException) {
            // POST_NOTIFICATIONS not granted (API 33+): in-app relief still fires.
        }
    }
}
