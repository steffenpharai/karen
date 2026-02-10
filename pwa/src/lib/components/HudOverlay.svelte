<!--
  Iron Man-style HUD overlay: live camera feed + canvas AR overlay.

  Replaces the separate CameraStream + HologramView with a unified view.
  The MJPEG raw feed (no server-side annotations) serves as the base layer.
  A canvas overlay draws:
    - Corner bracket tracking boxes (color-coded by class, depth-aware alpha)
    - Object labels (class, distance, speed, track ID)
    - Velocity trajectory prediction lines
    - Circular radar minimap (top-down view)
    - Threat level indicator bar
    - Vitals mini-display (CPU temp, fan, RAM, uptime)
    - HUD border frame with scan-line animation
    - Stats readout (object count, update timing)

  Data flow:
    - trackedObjects store (2s via scan_result, filtered server-side to age<=1)
    - vitalsData store (via WS push)
    - threatData store (via WS push)

  Staleness handling:
    - Server filters tracked objects by age<=1 (no zombie tracks)
    - Client detects scene changes (<40% track ID overlap) and clears immediately
    - Velocity extrapolation predicts positions between 2s scan updates
    - Overlays fade 2.5s–6s after last data, then disappear
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		trackedObjects,
		vitalsData,
		threatData,
		detections,
		connectionStatus,
		getApiUrl,
		type TrackedObjectData,
		type VitalsData,
		type ThreatData,
	} from '$lib/stores/connection';

	let { compact = false }: { compact?: boolean } = $props();

	// ── Refs ──────────────────────────────────────────────────────────
	let wrapper = $state<HTMLDivElement>(undefined!);
	let canvas = $state<HTMLCanvasElement>(undefined!);
	let imgEl = $state<HTMLImageElement>(undefined!);

	// ── MJPEG stream state ────────────────────────────────────────────
	let baseStreamUrl = $derived(getApiUrl('/stream/raw'));
	let imgError = $state(false);
	let imgLoaded = $state(false);
	let reconnectCount = $state(0);
	let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	let streamUrl = $derived(
		reconnectCount === 0 ? baseStreamUrl : `${baseStreamUrl}?_r=${reconnectCount}`
	);

	function handleImgError() {
		imgError = true;
		imgLoaded = false;
		scheduleReconnect();
	}
	function handleImgLoad() {
		imgError = false;
		imgLoaded = true;
		if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
	}
	function scheduleReconnect() {
		if (reconnectTimer) return;
		const delay = Math.min(2000 * Math.pow(2, Math.min(reconnectCount, 3)), 10_000);
		reconnectTimer = setTimeout(() => { reconnectTimer = null; reconnectCount++; }, delay);
	}

	// ── Store subscriptions ───────────────────────────────────────────
	let tracked = $derived($trackedObjects);
	let vitals = $derived($vitalsData);
	let threat = $derived($threatData);
	let det = $derived($detections);
	let connStatus = $derived($connectionStatus);
	let isConnected = $derived(connStatus === 'connected');

	let threatBorderClass = $derived(
		threat?.level === 'high' || threat?.level === 'critical'
			? 'border-red-500/60'
			: threat?.level === 'moderate'
				? 'border-amber-500/50'
				: 'border-cyan-500/20'
	);

	// ── Animation state ───────────────────────────────────────────────
	let animId = 0;
	let ro: ResizeObserver | null = null;
	let canvasW = 0;
	let canvasH = 0;
	let dpr = 1;
	let lastDataTime = 0;
	let scanPhase = -1; // -1 = inactive, 0..1 = sweeping
	let prevTrackIds = new Set<number>();
	let newTrackIds = new Set<number>();
	let newTrackFlashEnd = 0;

	// Interpolated bracket positions: trackId -> {x1,y1,x2,y2}
	let interpPositions: Map<number, { x1: number; y1: number; x2: number; y2: number }> = new Map();

	// ── Constants ─────────────────────────────────────────────────────
	const CAMERA_W = 1280;
	const CAMERA_H = 720;

	const COCO_NAMES: Record<number, string> = {
		0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',
		6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
		11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',
		16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',21:'bear',
		22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',
		27:'tie',28:'suitcase',29:'frisbee',30:'skis',31:'snowboard',
		32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',
		36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',
		40:'wine glass',41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',
		46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
		51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',
		57:'couch',58:'potted plant',59:'bed',60:'dining table',61:'toilet',
		62:'tv',63:'laptop',64:'mouse',65:'remote',66:'keyboard',
		67:'cell phone',68:'microwave',69:'oven',70:'toaster',71:'sink',
		72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',
		77:'teddy bear',78:'hair drier',79:'toothbrush',
	};

	const VEHICLE_CLASSES = new Set(['car','truck','bus','motorcycle','bicycle','train','boat','airplane']);
	const WEAPON_CLASSES = new Set(['knife','scissors','baseball bat']);

	function objColor(name: string): string {
		if (name === 'person') return '#00ffff';
		if (VEHICLE_CLASSES.has(name)) return '#ffaa00';
		if (WEAPON_CLASSES.has(name)) return '#ff3355';
		return '#00ff88';
	}

	function resolveName(obj: TrackedObjectData): string {
		return obj.class_name || COCO_NAMES[obj.cls] || `obj_${obj.cls}`;
	}

	// ── Canvas sizing ─────────────────────────────────────────────────
	function resizeCanvas() {
		if (!canvas || !wrapper) return;
		const rect = wrapper.getBoundingClientRect();
		dpr = Math.min(window.devicePixelRatio || 1, 2);
		canvasW = rect.width;
		canvasH = rect.height;
		canvas.width = Math.round(canvasW * dpr);
		canvas.height = Math.round(canvasH * dpr);
		canvas.style.width = `${canvasW}px`;
		canvas.style.height = `${canvasH}px`;
	}

	// ── Coordinate mapping ────────────────────────────────────────────
	function camToCanvas(cx: number, cy: number): [number, number] {
		return [(cx / CAMERA_W) * canvasW * dpr, (cy / CAMERA_H) * canvasH * dpr];
	}

	/**
	 * Extrapolate object positions using velocity vectors.
	 * Predicts where each object IS NOW based on where it was at scan time
	 * plus velocity * elapsed seconds.  Clamps to camera bounds.
	 */
	function extrapolateObjects(objects: TrackedObjectData[], elapsedSec: number): TrackedObjectData[] {
		const extrap = Math.min(Math.max(elapsedSec, 0), 4); // cap at 4s prediction
		return objects.map((obj) => {
			const vx = obj.velocity[0] * extrap;
			const vy = obj.velocity[1] * extrap;
			return {
				...obj,
				xyxy: [
					Math.max(0, Math.min(CAMERA_W, obj.xyxy[0] + vx)),
					Math.max(0, Math.min(CAMERA_H, obj.xyxy[1] + vy)),
					Math.max(0, Math.min(CAMERA_W, obj.xyxy[2] + vx)),
					Math.max(0, Math.min(CAMERA_H, obj.xyxy[3] + vy)),
				],
			};
		});
	}

	// ── Interpolation ─────────────────────────────────────────────────
	function lerpPositions(objects: TrackedObjectData[], dt: number) {
		const speed = Math.min(dt * 6, 1); // lerp factor (faster convergence)
		const currentIds = new Set<number>();
		for (const obj of objects) {
			currentIds.add(obj.track_id);
			const [tx1, ty1] = camToCanvas(obj.xyxy[0], obj.xyxy[1]);
			const [tx2, ty2] = camToCanvas(obj.xyxy[2], obj.xyxy[3]);
			const prev = interpPositions.get(obj.track_id);
			if (prev) {
				prev.x1 += (tx1 - prev.x1) * speed;
				prev.y1 += (ty1 - prev.y1) * speed;
				prev.x2 += (tx2 - prev.x2) * speed;
				prev.y2 += (ty2 - prev.y2) * speed;
			} else {
				interpPositions.set(obj.track_id, { x1: tx1, y1: ty1, x2: tx2, y2: ty2 });
			}
		}
		// Remove tracks no longer present
		for (const id of interpPositions.keys()) {
			if (!currentIds.has(id)) interpPositions.delete(id);
		}
	}

	// ── Drawing functions ─────────────────────────────────────────────

	function drawHudFrame(ctx: CanvasRenderingContext2D, W: number, H: number, t: number) {
		const pad = 6 * dpr;
		const cornerSize = 20 * dpr;
		const lineW = 1.0 * dpr;

		ctx.save();
		ctx.strokeStyle = 'rgba(0, 255, 255, 0.25)';
		ctx.lineWidth = lineW;
		ctx.shadowColor = 'rgba(0, 255, 255, 0.15)';
		ctx.shadowBlur = 4 * dpr;

		// Corner decorations
		const corners: [number, number, number, number][] = [
			[pad, pad, 1, 1], [W - pad, pad, -1, 1],
			[pad, H - pad, 1, -1], [W - pad, H - pad, -1, -1],
		];
		for (const [x, y, dx, dy] of corners) {
			ctx.beginPath();
			ctx.moveTo(x, y + cornerSize * dy);
			ctx.lineTo(x, y);
			ctx.lineTo(x + cornerSize * dx, y);
			ctx.stroke();
		}

		// Top center label
		ctx.shadowBlur = 0;
		ctx.font = `bold ${9 * dpr}px monospace`;
		ctx.fillStyle = 'rgba(0, 255, 255, 0.35)';
		ctx.textAlign = 'center';
		ctx.fillText('J.A.R.V.I.S.  //  HUD', W / 2, pad + 10 * dpr);

		// Scan line animation
		if (scanPhase >= 0 && scanPhase <= 1) {
			const sy = pad + scanPhase * (H - 2 * pad);
			const grad = ctx.createLinearGradient(0, sy, W, sy);
			grad.addColorStop(0, 'rgba(0, 255, 255, 0)');
			grad.addColorStop(0.3, 'rgba(0, 255, 255, 0.3)');
			grad.addColorStop(0.5, 'rgba(0, 255, 255, 0.5)');
			grad.addColorStop(0.7, 'rgba(0, 255, 255, 0.3)');
			grad.addColorStop(1, 'rgba(0, 255, 255, 0)');
			ctx.strokeStyle = grad;
			ctx.lineWidth = 1.5 * dpr;
			ctx.beginPath();
			ctx.moveTo(pad, sy);
			ctx.lineTo(W - pad, sy);
			ctx.stroke();
		}

		ctx.restore();
	}

	function drawCornerBrackets(
		ctx: CanvasRenderingContext2D,
		x1: number, y1: number, x2: number, y2: number,
		color: string, depthAlpha: number, isNew: boolean, t: number,
	) {
		const bw = x2 - x1;
		const bh = y2 - y1;
		const cornerLen = Math.max(10 * dpr, Math.min(bw, bh) * 0.2);
		const thickness = Math.max(1.5, (2 + depthAlpha) * dpr);
		const pulse = Math.sin(t * 2.5) * 0.1 + 0.9;
		const alpha = depthAlpha * pulse * (isNew ? (Math.sin(t * 12) * 0.3 + 0.7) : 1);

		ctx.save();
		ctx.strokeStyle = color;
		ctx.lineWidth = thickness;
		ctx.globalAlpha = Math.min(1, alpha);
		ctx.shadowColor = color;
		ctx.shadowBlur = (isNew ? 12 : 6) * dpr;
		ctx.lineCap = 'round';

		// Top-left
		ctx.beginPath();
		ctx.moveTo(x1, y1 + cornerLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cornerLen, y1);
		ctx.stroke();
		// Top-right
		ctx.beginPath();
		ctx.moveTo(x2 - cornerLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cornerLen);
		ctx.stroke();
		// Bottom-left
		ctx.beginPath();
		ctx.moveTo(x1, y2 - cornerLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cornerLen, y2);
		ctx.stroke();
		// Bottom-right
		ctx.beginPath();
		ctx.moveTo(x2 - cornerLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cornerLen);
		ctx.stroke();

		// Thin crosshair at center
		const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
		ctx.globalAlpha = Math.min(1, alpha * 0.4);
		ctx.lineWidth = 0.8 * dpr;
		ctx.shadowBlur = 0;
		const chSize = 6 * dpr;
		ctx.beginPath();
		ctx.moveTo(cx - chSize, cy); ctx.lineTo(cx + chSize, cy);
		ctx.moveTo(cx, cy - chSize); ctx.lineTo(cx, cy + chSize);
		ctx.stroke();

		ctx.restore();
	}

	function drawObjectLabel(
		ctx: CanvasRenderingContext2D,
		obj: TrackedObjectData,
		x1: number, y1: number, x2: number, y2: number,
		color: string,
	) {
		const name = resolveName(obj).toUpperCase();
		const fontSize = 9 * dpr;
		const smallFontSize = 7.5 * dpr;
		const padX = 5 * dpr;
		const padY = 3 * dpr;
		const lineH = fontSize + 2 * dpr;

		// Position: upper-right of box, offset
		const labelX = x2 + 6 * dpr;
		const labelY = y1;

		// Build label lines
		const lines: string[] = [name];
		if (obj.depth != null) {
			const dist = (obj.depth * 10).toFixed(1);
			const speed = Math.sqrt(obj.velocity[0] ** 2 + obj.velocity[1] ** 2);
			const speedMs = (speed / 100).toFixed(1); // rough px/s to m/s
			lines.push(`${dist}m  ${speedMs}m/s`);
		} else {
			const speed = Math.sqrt(obj.velocity[0] ** 2 + obj.velocity[1] ** 2);
			if (speed > 5) lines.push(`${(speed / 100).toFixed(1)}m/s`);
		}
		lines.push(`#${obj.track_id}  ${Math.round(obj.conf * 100)}%`);

		ctx.save();
		ctx.font = `bold ${fontSize}px monospace`;

		// Measure max width
		let maxW = 0;
		for (const line of lines) {
			maxW = Math.max(maxW, ctx.measureText(line).width);
		}
		// Use smaller font for data lines
		ctx.font = `${smallFontSize}px monospace`;
		for (let i = 1; i < lines.length; i++) {
			maxW = Math.max(maxW, ctx.measureText(lines[i]).width);
		}

		const boxW = maxW + padX * 2;
		const boxH = lineH * lines.length + padY * 2;

		// Clamp to canvas
		let lx = labelX;
		let ly = labelY;
		if (lx + boxW > canvasW * dpr - 4 * dpr) lx = x1 - boxW - 6 * dpr;
		if (ly + boxH > canvasH * dpr - 4 * dpr) ly = canvasH * dpr - boxH - 4 * dpr;
		if (ly < 4 * dpr) ly = 4 * dpr;

		// Background
		ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
		ctx.beginPath();
		const r = 3 * dpr;
		ctx.roundRect(lx, ly, boxW, boxH, r);
		ctx.fill();

		// Border
		ctx.strokeStyle = color;
		ctx.lineWidth = 0.8 * dpr;
		ctx.globalAlpha = 0.6;
		ctx.stroke();
		ctx.globalAlpha = 1;

		// Connecting line from label to object
		const connFromX = lx < x2 ? lx : lx + boxW;
		const connFromY = ly + boxH / 2;
		const connToX = (x1 + x2) / 2;
		const connToY = (y1 + y2) / 2;
		ctx.strokeStyle = color;
		ctx.globalAlpha = 0.3;
		ctx.lineWidth = 0.7 * dpr;
		ctx.setLineDash([3 * dpr, 3 * dpr]);
		ctx.beginPath();
		ctx.moveTo(connFromX, connFromY);
		ctx.lineTo(connToX, connToY);
		ctx.stroke();
		ctx.setLineDash([]);
		ctx.globalAlpha = 1;

		// Text
		let ty = ly + padY + fontSize;
		ctx.font = `bold ${fontSize}px monospace`;
		ctx.fillStyle = color;
		ctx.textAlign = 'left';
		ctx.fillText(lines[0], lx + padX, ty);
		ctx.font = `${smallFontSize}px monospace`;
		ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
		for (let i = 1; i < lines.length; i++) {
			ty += lineH;
			ctx.fillText(lines[i], lx + padX, ty);
		}

		ctx.restore();
	}

	function drawTrajectory(
		ctx: CanvasRenderingContext2D,
		obj: TrackedObjectData,
		x1: number, y1: number, x2: number, y2: number,
		color: string,
	) {
		const speed = Math.sqrt(obj.velocity[0] ** 2 + obj.velocity[1] ** 2);
		if (speed < 10) return; // skip slow/stationary

		const cx = (x1 + x2) / 2;
		const cy = (y1 + y2) / 2;
		const scale = (canvasW * dpr) / CAMERA_W;
		const vx = obj.velocity[0] * scale;
		const vy = obj.velocity[1] * scale;
		const len = Math.min(Math.sqrt(vx * vx + vy * vy) * 0.8, 80 * dpr);
		const angle = Math.atan2(vy, vx);
		const ex = cx + Math.cos(angle) * len;
		const ey = cy + Math.sin(angle) * len;

		ctx.save();
		ctx.strokeStyle = color;
		ctx.globalAlpha = 0.5;
		ctx.lineWidth = 1.2 * dpr;
		ctx.setLineDash([4 * dpr, 4 * dpr]);
		ctx.beginPath();
		ctx.moveTo(cx, cy);
		ctx.lineTo(ex, ey);
		ctx.stroke();
		ctx.setLineDash([]);

		// Arrowhead
		const aSize = 5 * dpr;
		ctx.fillStyle = color;
		ctx.globalAlpha = 0.6;
		ctx.beginPath();
		ctx.moveTo(ex, ey);
		ctx.lineTo(ex - aSize * Math.cos(angle - 0.5), ey - aSize * Math.sin(angle - 0.5));
		ctx.lineTo(ex - aSize * Math.cos(angle + 0.5), ey - aSize * Math.sin(angle + 0.5));
		ctx.closePath();
		ctx.fill();

		ctx.restore();
	}

	function drawRadar(
		ctx: CanvasRenderingContext2D,
		W: number, H: number,
		objects: TrackedObjectData[],
		t: number,
	) {
		const radius = (compact ? 35 : 42) * dpr;
		const cx = W - radius - 10 * dpr;
		const cy = H - radius - 10 * dpr;

		ctx.save();

		// Background circle
		ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
		ctx.beginPath();
		ctx.arc(cx, cy, radius, 0, Math.PI * 2);
		ctx.fill();

		// Border
		ctx.strokeStyle = 'rgba(0, 255, 255, 0.3)';
		ctx.lineWidth = 1 * dpr;
		ctx.stroke();

		// Distance rings
		ctx.globalAlpha = 0.15;
		for (const r of [0.33, 0.66]) {
			ctx.beginPath();
			ctx.arc(cx, cy, radius * r, 0, Math.PI * 2);
			ctx.stroke();
		}

		// Crosshairs
		ctx.globalAlpha = 0.12;
		ctx.beginPath();
		ctx.moveTo(cx - radius, cy); ctx.lineTo(cx + radius, cy);
		ctx.moveTo(cx, cy - radius); ctx.lineTo(cx, cy + radius);
		ctx.stroke();

		// Sweep line
		ctx.globalAlpha = 0.3;
		ctx.strokeStyle = '#00ffff';
		ctx.lineWidth = 1 * dpr;
		const sweepAngle = (t * 0.8) % (Math.PI * 2);
		ctx.beginPath();
		ctx.moveTo(cx, cy);
		ctx.lineTo(cx + Math.cos(sweepAngle) * radius, cy + Math.sin(sweepAngle) * radius);
		ctx.stroke();

		// Sweep trail (gradient arc)
		ctx.globalAlpha = 0.08;
		ctx.beginPath();
		ctx.moveTo(cx, cy);
		ctx.arc(cx, cy, radius * 0.95, sweepAngle - 0.8, sweepAngle);
		ctx.closePath();
		ctx.fillStyle = '#00ffff';
		ctx.fill();

		// Camera icon at center
		ctx.globalAlpha = 0.6;
		ctx.fillStyle = '#00ffff';
		const triSize = 4 * dpr;
		ctx.beginPath();
		ctx.moveTo(cx, cy - triSize);
		ctx.lineTo(cx - triSize * 0.7, cy + triSize * 0.5);
		ctx.lineTo(cx + triSize * 0.7, cy + triSize * 0.5);
		ctx.closePath();
		ctx.fill();

		// Objects as dots
		for (const obj of objects) {
			const name = resolveName(obj);
			const color = objColor(name);
			// Map x position to radar x (horizontal spread)
			const objCx = (obj.xyxy[0] + obj.xyxy[2]) / 2;
			const rx = ((objCx / CAMERA_W) - 0.5) * 2 * radius * 0.85;
			// Map depth to radar y (distance from camera)
			const depth = obj.depth ?? 0.5;
			const ry = -(depth * radius * 0.85); // negative = upward = further

			const dotX = cx + rx;
			const dotY = cy + ry;

			// Only draw if within radar circle
			const distFromCenter = Math.sqrt(rx * rx + ry * ry);
			if (distFromCenter > radius * 0.9) continue;

			ctx.globalAlpha = 0.8;
			ctx.fillStyle = color;
			ctx.beginPath();
			ctx.arc(dotX, dotY, 3 * dpr, 0, Math.PI * 2);
			ctx.fill();

			// Glow
			ctx.globalAlpha = 0.2;
			ctx.shadowColor = color;
			ctx.shadowBlur = 4 * dpr;
			ctx.beginPath();
			ctx.arc(dotX, dotY, 4 * dpr, 0, Math.PI * 2);
			ctx.fill();
			ctx.shadowBlur = 0;
		}

		// Label
		ctx.globalAlpha = 0.4;
		ctx.fillStyle = '#00ffff';
		ctx.font = `bold ${7 * dpr}px monospace`;
		ctx.textAlign = 'center';
		ctx.fillText('RADAR', cx, cy + radius + 9 * dpr);

		ctx.restore();
	}

	function drawThreatBar(
		ctx: CanvasRenderingContext2D,
		W: number, H: number,
		threatInfo: ThreatData | null,
		t: number,
	) {
		if (!threatInfo) return;
		const barW = 6 * dpr;
		const barH = (compact ? 80 : 110) * dpr;
		const x = W - barW - 10 * dpr;
		const y = 30 * dpr;

		ctx.save();

		// Background
		ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
		ctx.beginPath();
		ctx.roundRect(x - 2 * dpr, y - 2 * dpr, barW + 4 * dpr, barH + 4 * dpr, 3 * dpr);
		ctx.fill();

		// Gradient bar
		const grad = ctx.createLinearGradient(0, y + barH, 0, y);
		grad.addColorStop(0, '#00ff88');
		grad.addColorStop(0.4, '#ffaa00');
		grad.addColorStop(1, '#ff3355');
		ctx.fillStyle = grad;
		ctx.globalAlpha = 0.35;
		ctx.beginPath();
		ctx.roundRect(x, y, barW, barH, 2 * dpr);
		ctx.fill();

		// Level marker
		const level = threatInfo.score ?? 0;
		const markerY = y + barH - level * barH;
		const markerColor = level < 0.3 ? '#00ff88' : level < 0.6 ? '#ffaa00' : '#ff3355';
		ctx.globalAlpha = 0.9;
		ctx.fillStyle = markerColor;
		ctx.beginPath();
		ctx.roundRect(x - 1 * dpr, markerY - 2 * dpr, barW + 2 * dpr, 4 * dpr, 1 * dpr);
		ctx.fill();

		// Pulse at high threat
		if (level > 0.5) {
			ctx.shadowColor = markerColor;
			ctx.shadowBlur = (8 + Math.sin(t * 4) * 4) * dpr;
			ctx.beginPath();
			ctx.roundRect(x - 1 * dpr, markerY - 2 * dpr, barW + 2 * dpr, 4 * dpr, 1 * dpr);
			ctx.fill();
			ctx.shadowBlur = 0;
		}

		// Label
		ctx.globalAlpha = 0.5;
		ctx.fillStyle = markerColor;
		ctx.font = `bold ${7 * dpr}px monospace`;
		ctx.textAlign = 'center';
		ctx.fillText('THREAT', x + barW / 2, y - 5 * dpr);

		ctx.restore();
	}

	function drawVitalsHud(
		ctx: CanvasRenderingContext2D,
		W: number, H: number,
		vitalsInfo: VitalsData | null,
		t: number,
	) {
		if (!vitalsInfo) return;
		const x = 10 * dpr;
		const y = H - (compact ? 50 : 60) * dpr;
		const fontSize = 8 * dpr;

		ctx.save();

		// Background panel
		const panelW = (compact ? 110 : 130) * dpr;
		const panelH = (compact ? 42 : 50) * dpr;
		ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
		ctx.beginPath();
		ctx.roundRect(x, y, panelW, panelH, 4 * dpr);
		ctx.fill();
		ctx.strokeStyle = 'rgba(0, 255, 255, 0.2)';
		ctx.lineWidth = 0.7 * dpr;
		ctx.stroke();

		// Title
		ctx.fillStyle = 'rgba(0, 255, 255, 0.5)';
		ctx.font = `bold ${7 * dpr}px monospace`;
		ctx.textAlign = 'left';
		ctx.fillText('VITALS', x + 5 * dpr, y + 10 * dpr);

		// Heart rate
		let ty = y + 22 * dpr;
		if (vitalsInfo.heart_rate != null) {
			const heartPulse = Math.sin(t * 5) > 0.7 ? '#ff3355' : 'rgba(255, 51, 85, 0.5)';
			ctx.fillStyle = heartPulse;
			ctx.font = `${fontSize}px monospace`;
			ctx.fillText(`♥ ${Math.round(vitalsInfo.heart_rate)} BPM`, x + 5 * dpr, ty);
			ty += 12 * dpr;
		}

		// Fatigue
		const fColor = vitalsInfo.fatigue === 'alert' ? '#00ff88' :
			vitalsInfo.fatigue === 'mild' ? '#00ffff' :
			vitalsInfo.fatigue === 'moderate' ? '#ffaa00' : '#ff3355';
		ctx.fillStyle = fColor;
		ctx.font = `${fontSize}px monospace`;
		ctx.fillText(`FAT: ${vitalsInfo.fatigue?.toUpperCase() ?? '—'}`, x + 5 * dpr, ty);

		// Posture (right column)
		const pColor = vitalsInfo.posture === 'good' ? '#00ff88' :
			vitalsInfo.posture === 'fair' ? '#00ffff' : '#ff3355';
		ctx.fillStyle = pColor;
		ctx.fillText(`POS: ${vitalsInfo.posture?.toUpperCase() ?? '—'}`, x + 68 * dpr, ty);

		ctx.restore();
	}

	function drawStatsReadout(
		ctx: CanvasRenderingContext2D,
		W: number, H: number,
		objectCount: number,
		timeSince: number,
	) {
		const x = 10 * dpr;
		const y = 20 * dpr;
		const fontSize = 8 * dpr;

		ctx.save();
		ctx.font = `${fontSize}px monospace`;
		ctx.textAlign = 'left';

		// Object count
		ctx.fillStyle = 'rgba(0, 255, 255, 0.55)';
		ctx.fillText(`${objectCount} TRACKED`, x, y);

		// Time since update
		const timeStr = timeSince < 2 ? 'LIVE' : timeSince < 60 ? `${Math.round(timeSince)}s AGO` : `${Math.round(timeSince / 60)}m AGO`;
		const timeColor = timeSince < 5 ? 'rgba(0, 255, 136, 0.6)' : 'rgba(255, 170, 0, 0.5)';
		ctx.fillStyle = timeColor;
		ctx.fillText(timeStr, x, y + 11 * dpr);

		// Connection status
		if (!isConnected) {
			ctx.fillStyle = 'rgba(255, 51, 85, 0.7)';
			ctx.fillText('DISCONNECTED', x, y + 22 * dpr);
		}

		ctx.restore();
	}

	// ── Main draw loop ────────────────────────────────────────────────

	let lastFrameTime = 0;
	/** Max seconds before overlays start fading */
	const STALE_FADE_START = 2.5;
	/** Seconds after which overlays fully disappear */
	const STALE_FADE_END = 6;

	function drawFrame(now: number) {
		animId = requestAnimationFrame(drawFrame);

		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const dt = (now - lastFrameTime) / 1000;
		lastFrameTime = now;
		const t = now / 1000; // time in seconds

		const W = canvas.width;
		const H = canvas.height;

		// Clear
		ctx.clearRect(0, 0, W, H);

		// Time since last tracking data arrived
		const timeSinceData = lastDataTime > 0 ? (now - lastDataTime) / 1000 : 999;

		// Stale factor: 1.0 = fresh, fades to 0.0 when data is old
		const staleFactor = timeSinceData < STALE_FADE_START
			? 1.0
			: Math.max(0, 1 - (timeSinceData - STALE_FADE_START) / (STALE_FADE_END - STALE_FADE_START));

		// Only use tracked objects (with stable IDs) — no raw detection fallback
		// Raw detections have no persistent IDs and cause bracket jumping
		const objects: TrackedObjectData[] = tracked;

		// Extrapolate positions forward using velocity (predicts where objects are NOW)
		const predicted = staleFactor > 0 ? extrapolateObjects(objects, timeSinceData) : [];

		// Update scan animation
		if (scanPhase >= 0) {
			scanPhase += dt * 1.5;
			if (scanPhase > 1) scanPhase = -1;
		}

		// Lerp interpolated positions toward extrapolated targets
		lerpPositions(predicted, dt);

		// 1. HUD frame (always drawn)
		drawHudFrame(ctx, W, H, t);

		// 2. Tracked objects (with stale fading)
		if (staleFactor > 0.05) {
			for (const obj of predicted) {
				const pos = interpPositions.get(obj.track_id);
				if (!pos) continue;

				const name = resolveName(obj);
				const color = objColor(name);
				const depthAlpha = obj.depth != null ? Math.max(0.4, 1 - obj.depth * 0.6) : 0.8;
				const isNew = newTrackIds.has(obj.track_id) && now < newTrackFlashEnd;
				const alpha = depthAlpha * staleFactor;

				drawCornerBrackets(ctx, pos.x1, pos.y1, pos.x2, pos.y2, color, alpha, isNew, t);
				if (staleFactor > 0.3) {
					drawObjectLabel(ctx, obj, pos.x1, pos.y1, pos.x2, pos.y2, color);
				}
				drawTrajectory(ctx, obj, pos.x1, pos.y1, pos.x2, pos.y2, color);
			}
		}

		// 3. Radar (uses original non-extrapolated positions for stability)
		if (objects.length > 0 || isConnected) {
			drawRadar(ctx, W, H, objects, t);
		}

		// 4. Threat bar
		drawThreatBar(ctx, W, H, threat, t);

		// 5. Vitals
		drawVitalsHud(ctx, W, H, vitals, t);

		// 6. Stats
		drawStatsReadout(ctx, W, H, predicted.length, timeSinceData);
	}

	// ── Lifecycle ─────────────────────────────────────────────────────

	let healthTimer: ReturnType<typeof setInterval> | null = null;

	onMount(() => {
		resizeCanvas();
		lastFrameTime = performance.now();
		animId = requestAnimationFrame(drawFrame);

		ro = new ResizeObserver(() => resizeCanvas());
		if (wrapper) ro.observe(wrapper);

		// MJPEG health check
		healthTimer = setInterval(() => {
			if (!imgLoaded && !imgError && !reconnectTimer) {
				imgError = true;
				scheduleReconnect();
			}
		}, 8000);
	});

	onDestroy(() => {
		if (animId) cancelAnimationFrame(animId);
		ro?.disconnect();
		if (reconnectTimer) clearTimeout(reconnectTimer);
		if (healthTimer) clearInterval(healthTimer);
	});

	// React to new tracking data
	$effect(() => {
		// Trigger on tracked objects change
		const objs = tracked;
		const now = performance.now();

		// ALWAYS update lastDataTime — even when empty (camera moved to blank scene).
		// This lets the stale-fade kick in for any remaining brackets.
		lastDataTime = now;

		const currentIds = new Set(objs.map((o) => o.track_id));

		// ── Scene change detection ───────────────────────────────────
		// If most previous tracks disappeared, the camera moved to a new
		// scene.  Clear interpolated positions immediately so stale
		// brackets don't linger over a completely different image.
		if (prevTrackIds.size > 0) {
			if (currentIds.size === 0) {
				// All objects gone — clear everything instantly
				interpPositions.clear();
			} else {
				const overlap = [...prevTrackIds].filter((id) => currentIds.has(id)).length;
				const overlapRatio = overlap / prevTrackIds.size;
				if (overlapRatio < 0.4) {
					// Fewer than 40% of old tracks survived → scene changed
					// Remove positions for tracks that no longer exist
					for (const id of interpPositions.keys()) {
						if (!currentIds.has(id)) interpPositions.delete(id);
					}
				}
			}
		}

		if (objs.length > 0) {
			scanPhase = 0; // trigger scan animation
		}

		// Detect new tracks for flash animation
		const freshIds = new Set<number>();
		for (const id of currentIds) {
			if (!prevTrackIds.has(id)) freshIds.add(id);
		}
		if (freshIds.size > 0) {
			newTrackIds = freshIds;
			newTrackFlashEnd = now + 1500;
		}
		prevTrackIds = currentIds;
	});
</script>

<div class="relative glass overflow-hidden border-2 transition-colors duration-500 {threatBorderClass}">
	<div bind:this={wrapper} class="relative aspect-video bg-[var(--color-jarvis-surface)]">
		{#if imgError}
			<div class="absolute inset-0 flex items-center justify-center z-0">
				<div class="text-center">
					<div class="w-8 h-8 border-2 border-[var(--color-jarvis-cyan)]/30 border-t-[var(--color-jarvis-cyan)] rounded-full mx-auto mb-2 hud-spinner"></div>
					<p class="text-xs text-[var(--color-jarvis-muted)]">Reconnecting feed…</p>
				</div>
			</div>
		{:else}
			{#if !imgLoaded}
				<div class="absolute inset-0 flex items-center justify-center z-10">
					<div class="text-center">
						<div class="w-8 h-8 border-2 border-[var(--color-jarvis-cyan)]/30 border-t-[var(--color-jarvis-cyan)] rounded-full mx-auto mb-2 hud-spinner"></div>
						<p class="text-[10px] text-[var(--color-jarvis-muted)]">Connecting to camera…</p>
					</div>
				</div>
			{/if}
			<img
				bind:this={imgEl}
				src={streamUrl}
				alt="Live camera feed"
				class="w-full h-full object-cover"
				onerror={handleImgError}
				onload={handleImgLoad}
			/>
		{/if}

		<!-- Canvas HUD overlay -->
		<canvas
			bind:this={canvas}
			class="absolute inset-0 w-full h-full"
			style="pointer-events: none;"
		></canvas>

		<!-- HTML: LIVE badge (top-left) -->
		<div class="absolute top-2 left-2 z-20">
			{#if imgLoaded}
				<span class="text-[10px] px-1.5 py-0.5 rounded bg-black/60 backdrop-blur-sm font-mono font-bold flex items-center gap-1">
					<span class="w-1.5 h-1.5 rounded-full bg-[var(--color-jarvis-red)] hud-pulse"></span>
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
	</div>
</div>

<style>
	@keyframes hud-spin {
		to { transform: rotate(360deg); }
	}
	.hud-spinner {
		animation: hud-spin 0.8s linear infinite;
	}
	@keyframes hud-blink {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.3; }
	}
	.hud-pulse {
		animation: hud-blink 1.5s ease-in-out infinite;
	}
</style>
