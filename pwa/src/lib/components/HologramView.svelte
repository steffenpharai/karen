<!--
  Hologram viewer: 3D (Three.js/WebGL) when available, 2D canvas fallback otherwise.

  Data flows AUTONOMOUSLY via WebSocket — the server's background vision loop
  pushes hologram data (type: "hologram") every few seconds.
  Refresh button sends a WS request for an immediate update.
  No REST dependency — works as long as the WebSocket is connected.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { hologramData, sendHologramRequest, connectionStatus, type HologramData } from '$lib/stores/connection';

	let { compact = false }: { compact?: boolean } = $props();

	let container = $state<HTMLDivElement>(undefined!);
	let canvas2d = $state<HTMLCanvasElement>(undefined!);

	let error = $state('');
	let sceneReady = $state(false);
	let webglAvailable = $state(false);

	// Three.js refs (only used if WebGL works)
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let THREE: any = null;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let renderer: any = null;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let scene: any = null;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let camera: any = null;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let controls: any = null;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let pointsMesh: any = null;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let boxMeshes: any[] = [];
	let animId = 0;
	let ro: ResizeObserver | null = null;

	let currentData = $derived($hologramData);
	let connStatus = $derived($connectionStatus);
	let objectCount = $derived(currentData?.tracked_objects?.length ?? 0);
	let pointCount = $derived(currentData?.point_cloud?.length ?? 0);
	let isConnected = $derived(connStatus === 'connected');
	let lastUpdate = $state(0);
	let timeSinceUpdate = $state('');

	// Update "ago" timer
	let agoTimer: ReturnType<typeof setInterval> | null = null;

	function requestRefresh() {
		// Use WebSocket, not REST
		sendHologramRequest();
	}

	// ── WebGL detection ───────────────────────────────────────────────

	function detectWebGL(): boolean {
		try {
			const c = document.createElement('canvas');
			const gl = c.getContext('webgl2') || c.getContext('webgl') || c.getContext('experimental-webgl');
			if (!gl) return false;
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const debugInfo = (gl as any).getExtension('WEBGL_debug_renderer_info');
			if (debugInfo) {
				// eslint-disable-next-line @typescript-eslint/no-explicit-any
				const glRenderer = (gl as any).getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
				if (glRenderer && /disabled/i.test(glRenderer)) return false;
			}
			return true;
		} catch {
			return false;
		}
	}

	// ── 3D scene (Three.js) ───────────────────────────────────────────

	async function init3D() {
		try {
			const three = await import('three');
			THREE = three;
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			let OrbitControlsCtor: any = null;
			try {
				const mod = await import('three/addons/controls/OrbitControls.js');
				OrbitControlsCtor = mod.OrbitControls;
			} catch {
				try {
					const mod2 = await import('three/examples/jsm/controls/OrbitControls.js');
					OrbitControlsCtor = mod2.OrbitControls;
				} catch { /* continue without controls */ }
			}

			if (!container) return;
			const width = container.clientWidth || 400;
			const height = container.clientHeight || 300;

			renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
			renderer.setSize(width, height);
			renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
			renderer.setClearColor(0x0a0a0f, 1);
			container.appendChild(renderer.domElement);

			scene = new THREE.Scene();
			scene.fog = new THREE.FogExp2(0x0a0a0f, 0.06);

			camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 100);
			camera.position.set(0, 2, 8);

			if (OrbitControlsCtor) {
				controls = new OrbitControlsCtor(camera, renderer.domElement);
				controls.enableDamping = true;
				controls.dampingFactor = 0.05;
				controls.autoRotate = true;
				controls.autoRotateSpeed = 0.5;
			}

			const grid = new THREE.GridHelper(20, 20, 0x00ffff, 0x001a1a);
			grid.material.opacity = 0.3;
			grid.material.transparent = true;
			scene.add(grid);
			scene.add(new THREE.AmbientLight(0x00ffff, 0.3));

			sceneReady = true;

			function animate() {
				animId = requestAnimationFrame(animate);
				controls?.update();
				renderer.render(scene, camera);
			}
			animate();

			ro = new ResizeObserver(() => {
				if (!container || !camera || !renderer) return;
				const w = container.clientWidth || 400;
				const h = container.clientHeight || 300;
				camera.aspect = w / h;
				camera.updateProjectionMatrix();
				renderer.setSize(w, h);
			});
			ro.observe(container);
		} catch {
			// WebGL failed at runtime — fall back to 2D
			webglAvailable = false;
			sceneReady = true;
		}
	}

	// ── 2D canvas fallback ────────────────────────────────────────────

	function draw2D(data: HologramData) {
		if (!canvas2d) return;
		const ctx = canvas2d.getContext('2d');
		if (!ctx) return;

		const W = canvas2d.width;
		const H = canvas2d.height;

		// Clear
		ctx.fillStyle = '#0a0a0f';
		ctx.fillRect(0, 0, W, H);

		// Grid
		ctx.strokeStyle = 'rgba(0, 255, 255, 0.08)';
		ctx.lineWidth = 1;
		for (let x = 0; x < W; x += 30) {
			ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
		}
		for (let y = 0; y < H; y += 30) {
			ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
		}

		// Point cloud (top-down scatter)
		const pc = data.point_cloud;
		if (pc && pc.length > 0) {
			const step = Math.max(1, Math.floor(pc.length / 2000));
			for (let i = 0; i < pc.length; i += step) {
				const p = pc[i];
				const sx = W / 2 + p.x * 25;
				const sy = H / 2 - p.z * 25;
				const alpha = Math.max(0.2, 1.0 - Math.abs(p.y) * 0.3);
				ctx.fillStyle = `rgba(${p.r}, ${p.g}, ${p.b}, ${alpha.toFixed(2)})`;
				ctx.fillRect(sx, sy, 2, 2);
			}
		}

		// Tracked objects
		const tracked = data.tracked_objects;
		if (tracked) {
			for (const obj of tracked) {
				const [x1, y1, x2, y2] = obj.xyxy;
				const scaleX = W / 640;
				const scaleY = H / 480;
				const rx = x1 * scaleX;
				const ry = y1 * scaleY;
				const rw = (x2 - x1) * scaleX;
				const rh = (y2 - y1) * scaleY;

				const color = obj.class_name === 'person' ? '#00ffff' : '#00ff88';
				ctx.strokeStyle = color;
				ctx.lineWidth = 1.5;
				ctx.strokeRect(rx, ry, rw, rh);

				const label = `#${obj.track_id} ${obj.class_name}`;
				const textW = ctx.measureText(label).width + 6;
				ctx.fillStyle = 'rgba(0,0,0,0.7)';
				ctx.fillRect(rx, ry - 14, textW, 14);
				ctx.fillStyle = color;
				ctx.font = '10px monospace';
				ctx.fillText(label, rx + 3, ry - 3);

				if (obj.depth != null) {
					const depthLabel = `${(obj.depth * 10).toFixed(1)}m`;
					ctx.fillStyle = 'rgba(0,0,0,0.7)';
					ctx.fillRect(rx, ry + rh, ctx.measureText(depthLabel).width + 6, 14);
					ctx.fillStyle = '#ffaa00';
					ctx.fillText(depthLabel, rx + 3, ry + rh + 10);
				}
			}
		}

		// Watermark
		ctx.fillStyle = 'rgba(0, 255, 255, 0.15)';
		ctx.font = 'bold 10px monospace';
		ctx.fillText('HOLOGRAM  //  2D MODE', 8, H - 8);
	}

	// ── 3D scene update ───────────────────────────────────────────────

	function update3D(data: HologramData) {
		if (!scene || !THREE) return;

		if (pointsMesh) {
			scene.remove(pointsMesh);
			pointsMesh.geometry.dispose();
			pointsMesh.material.dispose();
			pointsMesh = null;
		}

		for (const m of boxMeshes) {
			scene.remove(m);
			if (m.geometry) m.geometry.dispose();
			if (m.material) {
				if (m.material.map) m.material.map.dispose();
				m.material.dispose();
			}
		}
		boxMeshes = [];

		const pc = data.point_cloud;
		if (pc && pc.length > 0) {
			const geometry = new THREE.BufferGeometry();
			const positions = new Float32Array(pc.length * 3);
			const colors = new Float32Array(pc.length * 3);
			for (let i = 0; i < pc.length; i++) {
				positions[i * 3] = pc[i].x;
				positions[i * 3 + 1] = pc[i].y;
				positions[i * 3 + 2] = pc[i].z;
				colors[i * 3] = pc[i].r / 255;
				colors[i * 3 + 1] = pc[i].g / 255;
				colors[i * 3 + 2] = pc[i].b / 255;
			}
			geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
			geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
			const material = new THREE.PointsMaterial({
				size: 0.05, vertexColors: true, transparent: true, opacity: 0.8, sizeAttenuation: true
			});
			pointsMesh = new THREE.Points(geometry, material);
			scene.add(pointsMesh);
		}

		const tracked = data.tracked_objects;
		if (tracked) {
			for (const obj of tracked) {
				const [x1, y1, x2, y2] = obj.xyxy;
				const cx = ((x1 + x2) / 2 - 320) / 100;
				const cy = -((y1 + y2) / 2 - 240) / 100;
				const z = (obj.depth ?? 0.5) * 10;
				const w = Math.max((x2 - x1) / 100, 0.1);
				const h = Math.max((y2 - y1) / 100, 0.1);

				const boxGeo = new THREE.BoxGeometry(w, h, 0.5);
				const boxMat = new THREE.MeshBasicMaterial({
					color: obj.class_name === 'person' ? 0x00ffff : 0x00ff88,
					wireframe: true, transparent: true, opacity: 0.7
				});
				const mesh = new THREE.Mesh(boxGeo, boxMat);
				mesh.position.set(cx, cy, z);
				scene.add(mesh);
				boxMeshes.push(mesh);

				if (obj.class_name) {
					const labelCanvas = document.createElement('canvas');
					labelCanvas.width = 128; labelCanvas.height = 32;
					const lctx = labelCanvas.getContext('2d');
					if (lctx) {
						lctx.fillStyle = 'rgba(0,0,0,0.6)';
						lctx.fillRect(0, 0, 128, 32);
						lctx.font = '14px monospace';
						lctx.fillStyle = obj.class_name === 'person' ? '#00ffff' : '#00ff88';
						lctx.fillText(`#${obj.track_id} ${obj.class_name}`, 4, 20);
						const texture = new THREE.CanvasTexture(labelCanvas);
						const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
						const sprite = new THREE.Sprite(spriteMat);
						sprite.position.set(cx, cy + h / 2 + 0.3, z);
						sprite.scale.set(1.5, 0.4, 1);
						scene.add(sprite);
						boxMeshes.push(sprite);
					}
				}
			}
		}
	}

	// ── Lifecycle ─────────────────────────────────────────────────────

	onMount(() => {
		webglAvailable = detectWebGL();
		if (webglAvailable) {
			init3D();
		} else {
			sceneReady = true;
		}

		// "time ago" ticker
		agoTimer = setInterval(() => {
			if (lastUpdate === 0) { timeSinceUpdate = ''; return; }
			const sec = Math.floor((Date.now() - lastUpdate) / 1000);
			timeSinceUpdate = sec < 5 ? 'live' : sec < 60 ? `${sec}s ago` : `${Math.floor(sec / 60)}m ago`;
		}, 1000);
	});

	onDestroy(() => {
		if (animId) cancelAnimationFrame(animId);
		if (agoTimer) clearInterval(agoTimer);
		ro?.disconnect();
		controls?.dispose();
		renderer?.dispose();
	});

	// React to data changes (from WS push — autonomous)
	$effect(() => {
		if (!currentData || !sceneReady) return;
		lastUpdate = Date.now();
		error = '';
		if (webglAvailable && scene && THREE) {
			update3D(currentData);
		} else if (!webglAvailable) {
			draw2D(currentData);
		}
	});
</script>

<div class="glass p-3 space-y-2">
	<div class="flex items-center justify-between">
		<h2 class="font-[var(--font-heading)] text-[var(--color-jarvis-cyan)] text-xs font-bold uppercase tracking-widest">
			Hologram
		</h2>
		<div class="flex items-center gap-2">
			{#if !webglAvailable && sceneReady}
				<span class="text-[8px] text-[var(--color-jarvis-muted)] font-mono">2D</span>
			{/if}
			{#if timeSinceUpdate}
				<span class="text-[8px] font-mono {timeSinceUpdate === 'live' ? 'text-[var(--color-jarvis-green)]' : 'text-[var(--color-jarvis-muted)]'}">
					{timeSinceUpdate}
				</span>
			{/if}
			{#if pointCount > 0}
				<span class="text-[9px] text-[var(--color-jarvis-muted)] font-mono">
					{pointCount.toLocaleString()} pts
				</span>
			{/if}
			{#if objectCount > 0}
				<span class="text-[9px] text-[var(--color-jarvis-cyan)] font-mono">
					{objectCount} obj
				</span>
			{/if}
			<button
				onclick={requestRefresh}
				disabled={!isConnected}
				class="text-[10px] px-2 py-0.5 rounded border border-[var(--color-jarvis-cyan)]/30 text-[var(--color-jarvis-cyan)] hover:bg-[var(--color-jarvis-cyan)]/10 transition-colors disabled:opacity-30"
				title={isConnected ? 'Request immediate scan' : 'WebSocket disconnected'}
			>
				Refresh
			</button>
		</div>
	</div>

	{#if error}
		<div class="flex items-start gap-1.5 px-2 py-1.5 rounded bg-[var(--color-jarvis-red)]/10 border border-[var(--color-jarvis-red)]/20">
			<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--color-jarvis-red)" stroke-width="2" class="flex-shrink-0 mt-0.5">
				<circle cx="12" cy="12" r="10"/>
				<line x1="12" y1="8" x2="12" y2="12"/>
				<line x1="12" y1="16" x2="12.01" y2="16"/>
			</svg>
			<p class="text-[10px] text-[var(--color-jarvis-red)] leading-snug">{error}</p>
		</div>
	{/if}

	{#if !isConnected && !currentData}
		<div class="py-4 text-center">
			<p class="text-[10px] text-[var(--color-jarvis-muted)]">Waiting for WebSocket connection…</p>
		</div>
	{:else if !currentData && sceneReady}
		<div class="py-2 text-center">
			<p class="text-[10px] text-[var(--color-jarvis-muted)] opacity-60">Waiting for vision data…</p>
		</div>
	{/if}

	<!-- 3D container (WebGL) -->
	{#if webglAvailable}
		<div
			bind:this={container}
			class="w-full rounded-lg overflow-hidden"
			style="height: {compact ? '200px' : '300px'}; min-height: {compact ? '150px' : '200px'};"
		></div>
	{:else}
		<!-- 2D canvas fallback -->
		<canvas
			bind:this={canvas2d}
			width="640"
			height="480"
			class="w-full rounded-lg"
			style="height: {compact ? '200px' : '300px'}; min-height: {compact ? '150px' : '200px'}; image-rendering: pixelated;"
		></canvas>
	{/if}

	{#if currentData?.description}
		<p class="text-[10px] text-[var(--color-jarvis-muted)] leading-snug">
			{currentData.description}
		</p>
	{/if}
</div>
