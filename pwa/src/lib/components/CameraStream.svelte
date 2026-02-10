<!--
  Live camera feed via MJPEG <img src="/stream"> + detection/threat overlay.
  Detections come from WebSocket (type: detections/scan_result).

  Mobile Chrome resilience:
  - Auto-reconnects MJPEG stream on error (exponential backoff, max 10s)
  - Shows loading spinner while stream is connecting
  - "LIVE" / "RECONNECTING" indicator in top-left
  - Camera feed always has visible height via aspect-video
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { detections, getApiUrl, threatData } from '$lib/stores/connection';

	let det = $derived($detections);
	let threat = $derived($threatData);
	let baseStreamUrl = $derived(getApiUrl('/stream'));
	let imgError = $state(false);
	let imgLoaded = $state(false);
	let reconnectCount = $state(0);
	let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

	// Append cache-busting param on reconnect to force a fresh MJPEG connection
	let streamUrl = $derived(
		reconnectCount === 0 ? baseStreamUrl : `${baseStreamUrl}?_r=${reconnectCount}`
	);

	let detCount = $derived(det.detections?.length ?? 0);
	let threatLevel = $derived(threat?.level ?? 'clear');

	const threatBorderColors: Record<string, string> = {
		clear: 'border-[var(--color-jarvis-cyan)]/20',
		low: 'border-[var(--color-jarvis-cyan)]/30',
		moderate: 'border-amber-500/50',
		high: 'border-[var(--color-jarvis-red)]/60',
		critical: 'border-[var(--color-jarvis-red)]',
	};

	function handleImgError() {
		imgError = true;
		imgLoaded = false;
		scheduleReconnect();
	}

	function handleImgLoad() {
		imgError = false;
		imgLoaded = true;
		// Reset reconnect backoff on success
		if (reconnectTimer) {
			clearTimeout(reconnectTimer);
			reconnectTimer = null;
		}
	}

	function scheduleReconnect() {
		if (reconnectTimer) return;
		// Exponential backoff: 2s, 4s, 8s, max 10s
		const delay = Math.min(2000 * Math.pow(2, Math.min(reconnectCount, 3)), 10_000);
		reconnectTimer = setTimeout(() => {
			reconnectTimer = null;
			reconnectCount++;
		}, delay);
	}

	// Periodic health check: if stream hasn't loaded after 8s, force reconnect
	let healthTimer: ReturnType<typeof setInterval> | null = null;
	let lastLoadTime = 0;

	onMount(() => {
		healthTimer = setInterval(() => {
			if (!imgLoaded && !imgError && !reconnectTimer) {
				// Stream never loaded and no error fired — force reconnect
				imgError = true;
				scheduleReconnect();
			}
		}, 8000);
	});

	onDestroy(() => {
		if (reconnectTimer) clearTimeout(reconnectTimer);
		if (healthTimer) clearInterval(healthTimer);
	});
</script>

<div class="relative glass overflow-hidden border-2 transition-colors duration-500 {threatBorderColors[threatLevel] ?? 'border-[var(--color-jarvis-cyan)]/20'}">
	<div class="relative aspect-video bg-[var(--color-jarvis-surface)]">
		{#if imgError}
			<!-- Offline / reconnecting state -->
			<div class="absolute inset-0 flex items-center justify-center">
				<div class="text-center">
					<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--color-jarvis-muted)" stroke-width="1.5" class="mx-auto mb-2">
						<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
						<circle cx="12" cy="13" r="4"></circle>
					</svg>
					<p class="text-xs text-[var(--color-jarvis-muted)]">Reconnecting camera…</p>
				</div>
			</div>
		{:else}
			<!-- Loading spinner overlay (shown until first frame arrives) -->
			{#if !imgLoaded}
				<div class="absolute inset-0 flex items-center justify-center z-10">
					<div class="text-center">
						<div class="w-8 h-8 border-2 border-[var(--color-jarvis-cyan)]/30 border-t-[var(--color-jarvis-cyan)] rounded-full mx-auto mb-2 cam-spinner"></div>
						<p class="text-[10px] text-[var(--color-jarvis-muted)]">Connecting to camera…</p>
					</div>
				</div>
			{/if}

			<!-- MJPEG stream: browser natively handles multipart/x-mixed-replace -->
			<img
				src={streamUrl}
				alt="Live camera feed with YOLOE detections"
				class="w-full h-full object-cover"
				onerror={handleImgError}
				onload={handleImgLoad}
			/>
		{/if}

		<!-- Top-left: LIVE / status indicator -->
		<div class="absolute top-2 left-2 flex items-center gap-1.5">
			{#if imgLoaded}
				<span class="text-[10px] px-1.5 py-0.5 rounded bg-black/60 backdrop-blur-sm font-mono font-bold flex items-center gap-1">
					<span class="w-1.5 h-1.5 rounded-full bg-[var(--color-jarvis-red)] cam-pulse"></span>
					<span class="text-[var(--color-jarvis-text)]">LIVE</span>
				</span>
			{:else if imgError}
				<span class="text-[10px] px-1.5 py-0.5 rounded bg-black/60 backdrop-blur-sm font-mono text-[var(--color-jarvis-muted)]">
					OFFLINE
				</span>
			{:else}
				<span class="text-[10px] px-1.5 py-0.5 rounded bg-black/60 backdrop-blur-sm font-mono text-[var(--color-jarvis-muted)]">
					CONNECTING
				</span>
			{/if}
		</div>

		<!-- Top-right: detection count + threat badge -->
		<div class="absolute top-2 right-2 flex items-center gap-1.5">
			{#if detCount > 0}
				<span class="text-[10px] px-1.5 py-0.5 rounded bg-black/60 text-[var(--color-jarvis-cyan)] font-mono backdrop-blur-sm">
					{detCount} det
				</span>
			{/if}
			{#if threatLevel !== 'clear'}
				<span class="text-[10px] px-1.5 py-0.5 rounded font-mono font-bold uppercase backdrop-blur-sm
					{threatLevel === 'high' || threatLevel === 'critical'
						? 'bg-[var(--color-jarvis-red)]/30 text-[var(--color-jarvis-red)]'
						: threatLevel === 'moderate'
							? 'bg-amber-500/30 text-amber-400'
							: 'bg-[var(--color-jarvis-cyan)]/20 text-[var(--color-jarvis-cyan)]'}">
					{threatLevel}
				</span>
			{/if}
		</div>

		<!-- Bottom: detection info overlay -->
		{#if det.description}
			<div class="absolute bottom-0 left-0 right-0 bg-black/70 backdrop-blur-sm px-3 py-2">
				<p class="text-xs text-[var(--color-jarvis-cyan)] truncate">{det.description}</p>
			</div>
		{/if}
	</div>
</div>

<style>
	@keyframes cam-spin {
		to { transform: rotate(360deg); }
	}
	.cam-spinner {
		animation: cam-spin 0.8s linear infinite;
	}
	@keyframes cam-blink {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.3; }
	}
	.cam-pulse {
		animation: cam-blink 1.5s ease-in-out infinite;
	}
</style>
