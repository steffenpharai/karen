<!--
  Vitals HUD: fatigue gauge, posture indicator, heart rate display.
  Receives data via WebSocket (vitalsData store) or REST /api/vitals.
  MCU Jarvis aesthetic: cyan accents, clean gauges.
-->
<script lang="ts">
	import { vitalsData, connectionStatus } from '$lib/stores/connection';

	let currentVitals = $derived($vitalsData);
	let connStatus = $derived($connectionStatus);
	let isConnected = $derived(connStatus === 'connected');

	function fatigueColor(level: string): string {
		switch (level) {
			case 'alert': return 'var(--color-jarvis-green, #00ff88)';
			case 'mild': return 'var(--color-jarvis-cyan)';
			case 'moderate': return 'var(--color-jarvis-amber, #ffaa00)';
			case 'severe': return 'var(--color-jarvis-red)';
			default: return 'var(--color-jarvis-muted)';
		}
	}

	function postureColor(label: string): string {
		switch (label) {
			case 'good': return 'var(--color-jarvis-green, #00ff88)';
			case 'fair': return 'var(--color-jarvis-cyan)';
			case 'poor': return 'var(--color-jarvis-red)';
			default: return 'var(--color-jarvis-muted)';
		}
	}

	function fatigueWidth(level: string): number {
		switch (level) {
			case 'alert': return 100;
			case 'mild': return 70;
			case 'moderate': return 40;
			case 'severe': return 15;
			default: return 0;
		}
	}
</script>

<div class="glass p-4 space-y-3">
	<div class="flex items-center justify-between">
		<h2 class="font-[var(--font-heading)] text-[var(--color-jarvis-cyan)] text-xs font-bold uppercase tracking-widest">
			Vitals
		</h2>
		{#if !isConnected}
			<span class="text-[10px] text-[var(--color-jarvis-muted)]">offline</span>
		{/if}
	</div>

	{#if currentVitals}
		<!-- Fatigue -->
		<div class="space-y-1">
			<div class="flex justify-between items-baseline">
				<span class="text-[10px] text-[var(--color-jarvis-muted)] uppercase tracking-wider">Fatigue</span>
				<span class="text-xs font-mono" style="color: {fatigueColor(currentVitals.fatigue)}">
					{currentVitals.fatigue}
				</span>
			</div>
			<div class="h-1.5 rounded-full bg-[var(--color-jarvis-border)] overflow-hidden">
				<div
					class="h-full rounded-full transition-all duration-500"
					style="width: {fatigueWidth(currentVitals.fatigue)}%; background: {fatigueColor(currentVitals.fatigue)}"
				></div>
			</div>
		</div>

		<!-- Posture -->
		<div class="flex justify-between items-center">
			<span class="text-[10px] text-[var(--color-jarvis-muted)] uppercase tracking-wider">Posture</span>
			<div class="flex items-center gap-1.5">
				<div
					class="w-2 h-2 rounded-full"
					style="background: {postureColor(currentVitals.posture)}"
				></div>
				<span class="text-xs font-mono" style="color: {postureColor(currentVitals.posture)}">
					{currentVitals.posture}
				</span>
			</div>
		</div>

		<!-- Heart Rate -->
		<div class="flex justify-between items-center">
			<span class="text-[10px] text-[var(--color-jarvis-muted)] uppercase tracking-wider">Heart Rate</span>
			{#if currentVitals.heart_rate !== null}
				<div class="flex items-center gap-1">
					<span class="text-xs font-mono text-[var(--color-jarvis-cyan)]">
						{Math.round(currentVitals.heart_rate)} BPM
					</span>
					{#if currentVitals.hr_confidence < 0.5}
						<span
							class="text-[8px] px-1 py-0.5 rounded border border-[var(--color-jarvis-amber,#ffaa00)]/40 text-[var(--color-jarvis-amber,#ffaa00)]"
							title="Low confidence rPPG measurement"
						>
							est.
						</span>
					{/if}
				</div>
			{:else}
				<span class="text-xs text-[var(--color-jarvis-muted)]">—</span>
			{/if}
		</div>

		<!-- Alerts -->
		{#if currentVitals.alerts && currentVitals.alerts.length > 0}
			<div class="space-y-1 pt-1 border-t border-[var(--color-jarvis-border)]">
				{#each currentVitals.alerts as alert}
					<div class="flex items-center gap-1.5 text-[10px] text-[var(--color-jarvis-amber,#ffaa00)]">
						<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
							<line x1="12" y1="9" x2="12" y2="13"></line>
							<line x1="12" y1="17" x2="12.01" y2="17"></line>
						</svg>
						<span>{alert}</span>
					</div>
				{/each}
			</div>
		{/if}
	{:else}
		<p class="text-xs text-[var(--color-jarvis-muted)]">No vitals data</p>
	{/if}
</div>
