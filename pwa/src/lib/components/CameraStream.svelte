<!--
  Live camera feed via MJPEG <img src="/stream"> + detection/threat overlay.
  Detections come from WebSocket (type: detections/scan_result).
-->
<script lang="ts">
	import { detections, getApiUrl, threatData } from '$lib/stores/connection';

	let det = $derived($detections);
	let threat = $derived($threatData);
	let streamUrl = $derived(getApiUrl('/stream'));
	let imgError = $state(false);

	let detCount = $derived(det.detections?.length ?? 0);
	let threatLevel = $derived(threat?.level ?? 'clear');

	const threatBorderColors: Record<string, string> = {
		clear: 'border-transparent',
		low: 'border-[var(--color-jarvis-cyan)]/30',
		moderate: 'border-amber-500/50',
		high: 'border-[var(--color-jarvis-red)]/60',
		critical: 'border-[var(--color-jarvis-red)]',
	};

	function handleImgError() {
		imgError = true;
	}

	function handleImgLoad() {
		imgError = false;
	}
</script>

<div class="relative glass overflow-hidden border-2 transition-colors duration-500 {threatBorderColors[threatLevel] ?? 'border-transparent'}">
	<div class="relative aspect-video bg-[var(--color-jarvis-surface)]">
		{#if imgError}
			<div class="absolute inset-0 flex items-center justify-center">
				<div class="text-center">
					<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--color-jarvis-muted)" stroke-width="1.5" class="mx-auto mb-2">
						<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
						<circle cx="12" cy="13" r="4"></circle>
					</svg>
					<p class="text-xs text-[var(--color-jarvis-muted)]">Camera offline</p>
				</div>
			</div>
		{:else}
			<!-- MJPEG stream: browser natively handles multipart/x-mixed-replace -->
			<img
				src={streamUrl}
				alt="Live camera feed with YOLOE detections"
				class="w-full h-full object-cover"
				onerror={handleImgError}
				onload={handleImgLoad}
			/>
		{/if}

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
