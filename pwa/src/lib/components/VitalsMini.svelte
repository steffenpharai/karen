<!--
  Compact vitals + threat summary for folded (cover) layout.
  Single row: fatigue dot, posture dot, threat badge.
-->
<script lang="ts">
	import { vitalsData, threatData } from '$lib/stores/connection';

	let vitals = $derived($vitalsData);
	let threat = $derived($threatData);

	let threatLevel = $derived(threat?.level ?? 'clear');
	let showThreat = $derived(threatLevel !== 'clear');

	const fatigueIcons: Record<string, string> = {
		alert: '#00ff88',
		mild: '#00ffff',
		moderate: '#ffaa00',
		severe: '#ff3355',
		unknown: '#6b6b80',
	};

	const postureIcons: Record<string, string> = {
		good: '#00ff88',
		fair: '#00ffff',
		poor: '#ff3355',
		unknown: '#6b6b80',
	};

	const threatColorMap: Record<string, string> = {
		low: '#00ffff',
		moderate: '#ffaa00',
		high: '#ff3355',
		critical: '#ff3355',
	};
</script>

{#if vitals || showThreat}
	<div class="flex items-center justify-center gap-3 py-1.5 px-3">
		{#if vitals}
			<!-- Fatigue -->
			<div class="flex items-center gap-1" title="Fatigue: {vitals.fatigue}">
				<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24"
					fill="none" stroke={fatigueIcons[vitals.fatigue] ?? '#6b6b80'} stroke-width="2">
					<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
					<circle cx="12" cy="12" r="3"/>
				</svg>
				<span class="text-[9px] font-mono uppercase" style="color: {fatigueIcons[vitals.fatigue] ?? '#6b6b80'}">
					{vitals.fatigue}
				</span>
			</div>

			<span class="text-[var(--color-jarvis-border)]">|</span>

			<!-- Posture -->
			<div class="flex items-center gap-1" title="Posture: {vitals.posture}">
				<div class="w-1.5 h-1.5 rounded-full" style="background: {postureIcons[vitals.posture] ?? '#6b6b80'}"></div>
				<span class="text-[9px] font-mono uppercase" style="color: {postureIcons[vitals.posture] ?? '#6b6b80'}">
					{vitals.posture}
				</span>
			</div>

			{#if vitals.heart_rate !== null}
				<span class="text-[var(--color-jarvis-border)]">|</span>
				<span class="text-[9px] font-mono text-[var(--color-jarvis-cyan)]" title="Heart rate (estimated)">
					{Math.round(vitals.heart_rate)} bpm
				</span>
			{/if}
		{/if}

		{#if showThreat}
			{#if vitals}<span class="text-[var(--color-jarvis-border)]">|</span>{/if}
			<span
				class="text-[9px] font-mono font-bold uppercase"
				style="color: {threatColorMap[threatLevel] ?? '#6b6b80'}"
				title={threat?.summary ?? ''}
			>
				{#if threatLevel === 'high' || threatLevel === 'critical'}
					<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24"
						fill="none" stroke="currentColor" stroke-width="2.5" class="inline-block -mt-0.5 mr-0.5">
						<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
						<line x1="12" y1="9" x2="12" y2="13"/>
						<line x1="12" y1="17" x2="12.01" y2="17"/>
					</svg>
				{/if}
				{threatLevel}
			</span>
		{/if}
	</div>
{/if}
