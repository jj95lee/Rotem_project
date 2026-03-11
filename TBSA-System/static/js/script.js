/* script.js - 서버 생성 실시간 경로 이미지 방식 */
let currentSeq = 1;
let lastSeq = null;
let lastLogMsg = "";
let lastImageUpdate = 0;
let lastPathImageUpdate = 0;

let staticPathTimestamp = Date.now();
let lastPathNodeCount = 0;

let isImageLoading = false;

// SEQ 2 상태 표시용 변수
let lastActionStatus = "";

// 실시간 경로 이미지 업데이트 간격 (ms)
const PATH_IMAGE_INTERVAL = 200;  // 0.5초마다 업데이트

// ========================================
// SEQ4 전용 1인칭 전방뷰 렌더링 모듈
// ========================================
const SEQ4 = {
    canvas: null,
    ctx: null,
    miniCanvas: null,
    miniCtx: null,
    discovered: new Map(),
    currentNearby: [],
    trail: [],
    maxTrailLength: 100,
    sweepAngle: 0,
    animationFrame: null,

    tankX: 60,
    tankZ: 200,
    heading: 0,
    cameraHeading: 180,
    arrived: false,
    path: [],
    destination: null,

    config: {
        lidarRange: 50.0,
        canvasWidth: 1000,
        canvasHeight: 600,
        focalLength: 500,
        cameraHeight: 3.0,
        horizon: 230,
    },

    init() {
        console.log('[SEQ4] v3 init - cameraHeading default:', this.cameraHeading);
        this.canvas = document.getElementById('seq4Canvas');
        if (!this.canvas) return false;
        this.ctx = this.canvas.getContext('2d');
        this.resizeCanvas();
        // 컨테이너 크기 변경 감지
        this._resizeObserver = new ResizeObserver(() => this.resizeCanvas());
        this._resizeObserver.observe(this.canvas.parentElement);
        // 미니맵 캔버스
        this.miniCanvas = document.getElementById('seq4Minimap');
        if (this.miniCanvas) {
            this.miniCtx = this.miniCanvas.getContext('2d');
            this._miniResizeObserver = new ResizeObserver(() => this.resizeMinimap());
            this._miniResizeObserver.observe(this.miniCanvas.parentElement);
            this.resizeMinimap();
        }
        this.startAnimation();
        return true;
    },

    resizeCanvas() {
        const container = this.canvas.parentElement;
        if (!container) return;
        const w = container.clientWidth;
        const h = container.clientHeight;
        if (w < 10 || h < 10) return;
        this.canvas.width = w;
        this.canvas.height = h;
        this.config.canvasWidth = w;
        this.config.canvasHeight = h;
        this.config.focalLength = w * 0.5;
        this.config.horizon = Math.floor(h * 0.38);
    },

    resizeMinimap() {
        if (!this.miniCanvas) return;
        const container = this.miniCanvas.parentElement;
        if (!container) return;
        const size = Math.min(container.clientWidth, container.clientHeight);
        if (size < 10) return;
        this.miniCanvas.width = size;
        this.miniCanvas.height = size;
    },

    startAnimation() {
        if (this.animationFrame) cancelAnimationFrame(this.animationFrame);
        this.animate();
    },

    stopAnimation() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    },

    updateData(data) {
        const seq4 = data.seq4 || {};
        const { nearby = [], heading = 0, path = [] } = seq4;
        const { tank_pose, destination, log } = data;

        if (tank_pose) {
            this.tankX = tank_pose[0];
            this.tankZ = tank_pose.length === 3 ? tank_pose[2] : tank_pose[1];
            const now = Date.now();
            if (this.trail.length === 0 ||
                Math.hypot(this.tankX - this.trail[this.trail.length - 1].x,
                          this.tankZ - this.trail[this.trail.length - 1].z) > 0.5) {
                this.trail.push({ x: this.tankX, z: this.tankZ, time: now });
                if (this.trail.length > this.maxTrailLength) this.trail.shift();
            }
        }

        this.heading = heading || 0;
        this.path = path || [];
        this.destination = destination;

        // 카메라 방향: 실제 이동 방향에서 계산 (heading 값 의존 X)
        if (this.trail.length >= 3) {
            const last = this.trail[this.trail.length - 1];
            const prev = this.trail[Math.max(0, this.trail.length - 4)];
            const mdx = last.x - prev.x;
            const mdz = last.z - prev.z;
            if (Math.hypot(mdx, mdz) > 0.3) {
                this.cameraHeading = Math.atan2(mdx, mdz) * 180 / Math.PI;
            }
        } else if (destination) {
            const ddx = destination[0] - this.tankX;
            const ddz = destination[1] - this.tankZ;
            this.cameraHeading = Math.atan2(ddx, ddz) * 180 / Math.PI;
        }

        // 현재 LiDAR 범위 안의 장애물만 저장 (실시간 표시용)
        if (nearby) {
            const now = Date.now();
            this.currentNearby = nearby.map(obs => {
                const key = `${obs.x.toFixed(1)}_${obs.z.toFixed(1)}`;
                // 최초 발견 시각 기록 (글로우 효과용)
                if (!this.discovered.has(key)) {
                    this.discovered.set(key, now);
                }
                return { ...obs, discoveredAt: this.discovered.get(key) };
            });
        } else {
            this.currentNearby = [];
        }

        // 우측 패널 UI 업데이트
        const el = id => document.getElementById(id);
        const d = el('seq4-discovered');
        const l = el('seq4-log');
        const cp = el('seq4-current-pos');
        const dp = el('seq4-dest-pos');
        const rm = el('seq4-remaining');
        const arrRow = el('seq4-arrival-row');

        if (d) d.textContent = `${this.currentNearby.length}개`;
        if (l && log) l.textContent = log;
        if (cp) cp.textContent = `(${this.tankX.toFixed(1)}, ${this.tankZ.toFixed(1)})`;
        if (dp && destination) dp.textContent = `(${destination[0].toFixed(1)}, ${destination[1].toFixed(1)})`;
        else if (dp) dp.textContent = '-';
        if (rm && destination) {
            const dist = Math.hypot(destination[0] - this.tankX, destination[1] - this.tankZ);
            this.arrived = dist < 5;
            rm.textContent = this.arrived ? '도착!' : `${dist.toFixed(1)}m`;
            rm.style.color = dist < 20 ? '#4CAF50' : '#ff6666';
            if (arrRow) arrRow.style.display = this.arrived ? 'flex' : 'none';
        } else if (rm) { rm.textContent = '-'; }
    },

    animate() {
        this.render();
        if (this.miniCtx) this.renderMinimap(this.miniCtx);
        this.sweepAngle += 2;
        if (this.sweepAngle >= 360) this.sweepAngle = 0;
        this.animationFrame = requestAnimationFrame(() => this.animate());
    },

    // 월드 좌표 → 카메라 좌표 (이동 방향 기준)
    worldToCamera(wx, wz) {
        const dx = wx - this.tankX;
        const dz = wz - this.tankZ;
        const rad = this.cameraHeading * Math.PI / 180;
        return {
            x: dx * Math.cos(rad) - dz * Math.sin(rad),
            z: dx * Math.sin(rad) + dz * Math.cos(rad)
        };
    },

    // ========== 메인 렌더 ==========
    render() {
        if (!this.ctx) return;
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.config.canvasWidth, this.config.canvasHeight);

        this.renderSky(ctx);
        this.renderGround(ctx);
        this.renderPathFP(ctx);
        this.renderObstaclesFP(ctx);
        // this.renderLidarSweepFP(ctx);  // 스캔라인 제거
        this.renderDestinationFP(ctx);
        this.renderCrosshair(ctx);
        this.renderHUD(ctx);
    },

    // ========== 하늘 ==========
    renderSky(ctx) {
        const w = this.config.canvasWidth;
        const hz = this.config.horizon;
        const grad = ctx.createLinearGradient(0, 0, 0, hz);
        grad.addColorStop(0, '#050510');
        grad.addColorStop(0.7, '#0a1a0a');
        grad.addColorStop(1, '#0f2f0f');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, hz);

        // 지평선 글로우
        const glow = ctx.createLinearGradient(0, hz - 25, 0, hz + 10);
        glow.addColorStop(0, 'rgba(0,255,0,0)');
        glow.addColorStop(0.5, 'rgba(0,255,0,0.06)');
        glow.addColorStop(1, 'rgba(0,255,0,0)');
        ctx.fillStyle = glow;
        ctx.fillRect(0, hz - 25, w, 35);
    },

    // ========== 지면 + 그리드 ==========
    renderGround(ctx) {
        const w = this.config.canvasWidth;
        const h = this.config.canvasHeight;
        const hz = this.config.horizon;
        const fl = this.config.focalLength;
        const camH = this.config.cameraHeight;

        // 지면 그라데이션
        const gg = ctx.createLinearGradient(0, hz, 0, h);
        gg.addColorStop(0, '#0a1a0a');
        gg.addColorStop(1, '#152015');
        ctx.fillStyle = gg;
        ctx.fillRect(0, hz, w, h - hz);

        // 수평 그리드 (거리별)
        for (let z = 5; z <= 60; z += 5) {
            const sy = hz + (camH / z) * fl;
            if (sy > h || sy < hz) continue;
            const a = Math.max(0.03, 0.2 * (1 - z / 60));
            ctx.strokeStyle = `rgba(0,255,0,${a})`;
            ctx.lineWidth = z < 15 ? 1 : 0.5;
            ctx.beginPath();
            ctx.moveTo(0, sy); ctx.lineTo(w, sy);
            ctx.stroke();
        }

        // 수직 그리드 (소실점 수렴)
        const vx = w / 2;
        for (let x = -60; x <= 60; x += 10) {
            if (x === 0) continue;
            const sx = (x / 3) * fl + w / 2;
            const a = Math.max(0.03, 0.1 * (1 - Math.abs(x) / 60));
            ctx.strokeStyle = `rgba(0,255,0,${a})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(vx, hz); ctx.lineTo(sx, h);
            ctx.stroke();
        }

        // LiDAR 범위선
        const rangeSY = hz + (camH / this.config.lidarRange) * fl;
        if (rangeSY > hz && rangeSY < h) {
            ctx.strokeStyle = 'rgba(0,255,0,0.25)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([8, 4]);
            ctx.beginPath();
            ctx.moveTo(0, rangeSY); ctx.lineTo(w, rangeSY);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.font = '10px monospace';
            ctx.fillStyle = 'rgba(0,255,0,0.4)';
            ctx.textAlign = 'center';
            ctx.fillText(`── ${this.config.lidarRange}m LIDAR RANGE ──`, w / 2, rangeSY - 4);
            ctx.textAlign = 'left';
        }
    },

    // ========== 장애물 (원근 투영) ==========
    renderObstaclesFP(ctx) {
        const w = this.config.canvasWidth;
        const h = this.config.canvasHeight;
        const hz = this.config.horizon;
        const fl = this.config.focalLength;
        const camH = this.config.cameraHeight;
        const now = Date.now();

        // 거리순 정렬 (먼 것부터 → 가까운 것이 위에 그려짐)
        const sorted = this.currentNearby.map(obs => {
            const cam = this.worldToCamera(obs.x, obs.z);
            return { ...obs, camX: cam.x, camZ: cam.z };
        }).filter(o => o.camZ > 2).sort((a, b) => b.camZ - a.camZ);

        sorted.forEach(obs => {
            const { camX, camZ, size, discoveredAt } = obs;
            const isNew = (now - discoveredAt) < 2000;
            const glowFade = isNew ? Math.max(0, 1 - (now - discoveredAt) / 2000) : 0;

            const sx = (camX / camZ) * fl + w / 2;
            const groundY = hz + (camH / camZ) * fl;
            const appW = Math.max(2, (size / camZ) * fl);
            const appH = Math.max(2, (size * 1.8 / camZ) * fl);

            if (sx + appW / 2 < 0 || sx - appW / 2 > w) return;
            if (groundY < hz || groundY > h) return;

            const fog = Math.max(0.1, 1 - camZ / this.config.lidarRange);
            const topY = groundY - appH;

            // 바닥 그림자
            ctx.fillStyle = `rgba(0,0,0,${fog * 0.3})`;
            ctx.fillRect(sx - appW * 0.6, groundY - 2, appW * 1.2, 4);

            // 블록 본체
            if (isNew) {
                const br = Math.floor(220 * fog + 35 * glowFade);
                ctx.fillStyle = `rgba(${br},${Math.floor(br * 0.7)},0,${fog})`;
            } else {
                const g = Math.floor(70 * fog);
                ctx.fillStyle = `rgb(${g + 30},${g + 30},${g})`;
            }
            ctx.fillRect(sx - appW / 2, topY, appW, appH);

            // 윗면 (밝게)
            if (appH > 8) {
                const topH = Math.max(2, appH * 0.15);
                ctx.fillStyle = isNew
                    ? `rgba(255,220,80,${fog * 0.7})`
                    : `rgba(${Math.floor(110 * fog)},${Math.floor(110 * fog)},${Math.floor(100 * fog)},1)`;
                ctx.fillRect(sx - appW / 2, topY, appW, topH);
            }

            // 테두리
            if (isNew) {
                ctx.strokeStyle = `rgba(255,200,0,${fog * (0.5 + glowFade * 0.5)})`;
                ctx.lineWidth = 2;
                ctx.shadowBlur = 8 * glowFade;
                ctx.shadowColor = 'rgba(255,200,0,0.8)';
            } else {
                ctx.strokeStyle = `rgba(0,255,0,${fog * 0.35})`;
                ctx.lineWidth = 1;
            }
            ctx.strokeRect(sx - appW / 2, topY, appW, appH);
            ctx.shadowBlur = 0;

            // 거리 라벨
            if (camZ < 35 && appW > 12) {
                ctx.font = '10px monospace';
                ctx.fillStyle = `rgba(0,255,0,${fog * 0.8})`;
                ctx.textAlign = 'center';
                ctx.fillText(`${camZ.toFixed(0)}m`, sx, topY - 5);
                ctx.textAlign = 'left';
            }
        });
    },

    // ========== A* 경로 (바닥 도트) ==========
    renderPathFP(ctx) {
        if (!this.path || this.path.length < 2) return;
        const w = this.config.canvasWidth;
        const h = this.config.canvasHeight;
        const hz = this.config.horizon;
        const fl = this.config.focalLength;
        const camH = this.config.cameraHeight;

        this.path.forEach(p => {
            const cam = this.worldToCamera(p[0], p[1]);
            if (cam.z < 3) return;
            const sx = (cam.x / cam.z) * fl + w / 2;
            const sy = hz + (camH / cam.z) * fl;
            if (sx < 0 || sx > w || sy < hz || sy > h) return;

            const fog = Math.max(0.1, 1 - cam.z / this.config.lidarRange);
            const r = Math.max(1.5, 4 * (1 - cam.z / 60));
            ctx.fillStyle = `rgba(255,255,0,${fog * 0.5})`;
            ctx.beginPath();
            ctx.arc(sx, sy, r, 0, Math.PI * 2);
            ctx.fill();
        });
    },

    // ========== LiDAR 스캔라인 (1인칭) ==========
    renderLidarSweepFP(ctx) {
        const w = this.config.canvasWidth;
        const h = this.config.canvasHeight;
        const hz = this.config.horizon;
        const fl = this.config.focalLength;

        // 스윕 각도 → 화면상 위치
        const relAngle = ((this.sweepAngle - this.cameraHeading + 540) % 360) - 180;
        if (Math.abs(relAngle) > 50) return;

        const sx = Math.tan(relAngle * Math.PI / 180) * fl + w / 2;

        // 녹색 스캔 밴드
        const grad = ctx.createLinearGradient(sx - 35, 0, sx + 35, 0);
        grad.addColorStop(0, 'rgba(0,255,0,0)');
        grad.addColorStop(0.5, 'rgba(0,255,0,0.12)');
        grad.addColorStop(1, 'rgba(0,255,0,0)');
        ctx.fillStyle = grad;
        ctx.fillRect(sx - 35, hz, 70, h - hz);

        // 중심선
        ctx.strokeStyle = 'rgba(0,255,0,0.25)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sx, hz); ctx.lineTo(sx, h);
        ctx.stroke();
    },

    // ========== 목적지 표시 ==========
    renderDestinationFP(ctx) {
        if (!this.destination) return;
        const w = this.config.canvasWidth;
        const h = this.config.canvasHeight;
        const hz = this.config.horizon;
        const fl = this.config.focalLength;
        const camH = this.config.cameraHeight;

        const cam = this.worldToCamera(this.destination[0], this.destination[1]);
        const dist = Math.hypot(this.destination[0] - this.tankX, this.destination[1] - this.tankZ);

        // 전방에 있으면 지면에 마커
        if (cam.z > 3) {
            const sx = (cam.x / cam.z) * fl + w / 2;
            const sy = hz + (camH / cam.z) * fl;
            if (sx > 0 && sx < w && sy > hz && sy < h) {
                const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 300);
                const ms = Math.max(4, (6 / cam.z) * fl);

                ctx.fillStyle = `rgba(255,68,68,${0.5 + pulse * 0.3})`;
                ctx.beginPath();
                ctx.moveTo(sx, sy - ms);
                ctx.lineTo(sx + ms * 0.6, sy);
                ctx.lineTo(sx, sy + ms * 0.4);
                ctx.lineTo(sx - ms * 0.6, sy);
                ctx.closePath();
                ctx.fill();
                ctx.strokeStyle = `rgba(255,100,100,${0.6 + pulse * 0.3})`;
                ctx.lineWidth = 1.5;
                ctx.stroke();

                ctx.font = 'bold 12px monospace';
                ctx.fillStyle = '#ff4444';
                ctx.textAlign = 'center';
                ctx.fillText(`TGT ${dist.toFixed(0)}m`, sx, sy - ms - 8);
                ctx.textAlign = 'left';
            }
        }

        // 화면 상단에 방향 화살표 (항상 표시)
        const worldAngle = Math.atan2(
            this.destination[0] - this.tankX,
            this.destination[1] - this.tankZ
        ) * 180 / Math.PI;
        const relA = ((worldAngle - this.cameraHeading + 540) % 360) - 180;
        const arrowX = Math.max(30, Math.min(w - 30, w / 2 + relA * 3));
        const pulse2 = 0.6 + 0.4 * Math.sin(Date.now() / 400);

        ctx.save();
        ctx.translate(arrowX, 25);
        ctx.fillStyle = `rgba(255,68,68,${pulse2})`;
        ctx.beginPath();
        ctx.moveTo(0, -10); ctx.lineTo(10, 5); ctx.lineTo(-10, 5);
        ctx.closePath();
        ctx.fill();
        ctx.font = '10px monospace';
        ctx.fillStyle = '#ff4444';
        ctx.textAlign = 'center';
        ctx.fillText(`${dist.toFixed(0)}m`, 0, 20);
        ctx.textAlign = 'left';
        ctx.restore();
    },

    // ========== 조준점 ==========
    renderCrosshair(ctx) {
        const cx = this.config.canvasWidth / 2;
        const cy = this.config.horizon + 30;

        ctx.strokeStyle = 'rgba(0,255,0,0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx - 20, cy); ctx.lineTo(cx - 6, cy);
        ctx.moveTo(cx + 6, cy); ctx.lineTo(cx + 20, cy);
        ctx.moveTo(cx, cy - 15); ctx.lineTo(cx, cy - 6);
        ctx.moveTo(cx, cy + 6); ctx.lineTo(cx, cy + 15);
        ctx.stroke();

        ctx.fillStyle = 'rgba(0,255,0,0.5)';
        ctx.beginPath();
        ctx.arc(cx, cy, 2, 0, Math.PI * 2);
        ctx.fill();
    },

    // ========== HUD 오버레이 ==========
    renderHUD(ctx) {
        const w = this.config.canvasWidth;
        const h = this.config.canvasHeight;

        // 상단 나침반
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(w / 2 - 100, 0, 200, 18);
        ctx.strokeStyle = 'rgba(0,255,0,0.3)';
        ctx.lineWidth = 1;
        ctx.strokeRect(w / 2 - 100, 0, 200, 18);

        const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
        const ch = ((this.cameraHeading % 360) + 360) % 360;
        const di = Math.round(ch / 45) % 8;
        ctx.font = '11px monospace';
        ctx.fillStyle = '#0f0';
        ctx.textAlign = 'center';
        ctx.fillText(`${dirs[di]}  ${ch.toFixed(0)}°`, w / 2, 13);
        ctx.textAlign = 'left';

        // 좌하단: 위치 + 장애물
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(5, h - 45, 190, 40);
        ctx.strokeStyle = 'rgba(0,255,0,0.2)';
        ctx.strokeRect(5, h - 45, 190, 40);
        ctx.font = '11px monospace';
        ctx.fillStyle = '#0f0';
        ctx.fillText(`POS (${this.tankX.toFixed(1)}, ${this.tankZ.toFixed(1)})`, 12, h - 28);
        ctx.fillText(`OBS ${this.currentNearby.length} nearby`, 12, h - 13);

        // 우하단: 목적지 + 거리
        if (this.destination) {
            const dist = Math.hypot(this.destination[0] - this.tankX, this.destination[1] - this.tankZ);
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillRect(w - 195, h - 45, 190, 40);
            ctx.strokeStyle = this.arrived ? 'rgba(76,175,80,0.5)' : 'rgba(255,68,68,0.3)';
            ctx.strokeRect(w - 195, h - 45, 190, 40);
            ctx.font = '11px monospace';
            ctx.fillStyle = this.arrived ? '#4CAF50' : '#ff6666';
            ctx.fillText(`TGT (${this.destination[0].toFixed(0)}, ${this.destination[1].toFixed(0)})`, w - 188, h - 28);
            ctx.fillStyle = dist < 20 ? '#4CAF50' : '#ff6666';
            ctx.fillText(this.arrived ? 'ARRIVED' : `DST ${dist.toFixed(1)}m`, w - 188, h - 13);
        }

        // 도착 오버레이
        if (this.arrived) {
            const pulse = 0.6 + 0.4 * Math.sin(Date.now() / 500);
            ctx.fillStyle = `rgba(0,0,0,${0.3 * pulse})`;
            ctx.fillRect(w / 2 - 160, h / 2 - 30, 320, 60);
            ctx.strokeStyle = `rgba(76,175,80,${pulse})`;
            ctx.lineWidth = 2;
            ctx.strokeRect(w / 2 - 160, h / 2 - 30, 320, 60);
            ctx.font = 'bold 24px monospace';
            ctx.fillStyle = `rgba(76,175,80,${pulse})`;
            ctx.textAlign = 'center';
            ctx.fillText('MISSION COMPLETE', w / 2, h / 2 + 8);
            ctx.textAlign = 'left';
        }
    },

    // ========== 미니맵 (탑다운 레이더) - 우측 패널 캔버스 ==========
    renderMinimap(ctx) {
        if (!this.miniCanvas) return;
        const size = this.miniCanvas.width;
        if (size < 10) return;

        const cx = size / 2;
        const cy = size / 2;
        const radius = size / 2 - 8;
        const scale = radius / this.config.lidarRange;

        ctx.clearRect(0, 0, size, size);
        ctx.save();

        // 배경 원
        ctx.beginPath();
        ctx.arc(cx, cy, radius + 2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.85)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(0,255,0,0.4)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // 클리핑
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.clip();

        // LiDAR 범위 원 (점선)
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = 'rgba(0,255,0,0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // 십자선
        ctx.strokeStyle = 'rgba(0,255,0,0.1)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(cx - radius, cy); ctx.lineTo(cx + radius, cy);
        ctx.moveTo(cx, cy - radius); ctx.lineTo(cx, cy + radius);
        ctx.stroke();

        // 레이더 스윕 부채꼴
        const sweepRad = this.sweepAngle * Math.PI / 180;
        const headRad = this.cameraHeading * Math.PI / 180;
        const mapSweep = sweepRad - headRad;
        const grad = ctx.createConicGradient(mapSweep - 0.5, cx, cy);
        grad.addColorStop(0, 'rgba(0,255,0,0)');
        grad.addColorStop(0.12, 'rgba(0,255,0,0.15)');
        grad.addColorStop(0.15, 'rgba(0,255,0,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, radius, mapSweep - 0.5, mapSweep);
        ctx.closePath();
        ctx.fill();

        // 월드→미니맵 좌표 변환 (전차 중심, 진행 방향이 위쪽)
        const toMap = (wx, wz) => {
            const dx = wx - this.tankX;
            const dz = wz - this.tankZ;
            const r = -headRad;
            const rx = dx * Math.cos(r) - dz * Math.sin(r);
            const rz = dx * Math.sin(r) + dz * Math.cos(r);
            return { x: cx + rx * scale, y: cy - rz * scale };
        };

        // 장애물 표시
        this.currentNearby.forEach(obs => {
            const p = toMap(obs.x, obs.z);
            const s = Math.max(3, obs.size * scale * 0.8);
            ctx.fillStyle = 'rgba(255,220,80,0.8)';
            ctx.fillRect(p.x - s / 2, p.y - s / 2, s, s);
            ctx.strokeStyle = 'rgba(255,200,0,0.4)';
            ctx.lineWidth = 0.5;
            ctx.strokeRect(p.x - s / 2, p.y - s / 2, s, s);
        });

        // 목적지 표시
        if (this.destination) {
            const dp = toMap(this.destination[0], this.destination[1]);
            const dist = Math.hypot(dp.x - cx, dp.y - cy);
            if (dist > radius - 5) {
                const angle = Math.atan2(dp.y - cy, dp.x - cx);
                const ex = cx + (radius - 10) * Math.cos(angle);
                const ey = cy + (radius - 10) * Math.sin(angle);
                ctx.fillStyle = '#ff4444';
                ctx.beginPath();
                ctx.arc(ex, ey, 5, 0, Math.PI * 2);
                ctx.fill();
                ctx.font = '10px monospace';
                ctx.fillStyle = '#ff4444';
                ctx.textAlign = 'center';
                const dstDist = Math.hypot(this.destination[0] - this.tankX, this.destination[1] - this.tankZ);
                ctx.fillText(`${dstDist.toFixed(0)}m`, ex, ey - 9);
                ctx.textAlign = 'left';
            } else {
                ctx.fillStyle = '#ff4444';
                ctx.beginPath();
                ctx.moveTo(dp.x, dp.y - 6);
                ctx.lineTo(dp.x + 5, dp.y + 4);
                ctx.lineTo(dp.x - 5, dp.y + 4);
                ctx.closePath();
                ctx.fill();
            }
        }

        // 전차 (중앙, 삼각형)
        ctx.fillStyle = '#00ff00';
        ctx.beginPath();
        ctx.moveTo(cx, cy - 7);
        ctx.lineTo(cx + 5, cy + 5);
        ctx.lineTo(cx - 5, cy + 5);
        ctx.closePath();
        ctx.fill();

        // 범위 라벨
        ctx.font = '10px monospace';
        ctx.fillStyle = 'rgba(0,255,0,0.5)';
        ctx.textAlign = 'center';
        ctx.fillText(`${this.config.lidarRange}m`, cx, cy + radius - 3);
        ctx.textAlign = 'left';

        ctx.restore();
    },
};


window.addEventListener('load', () => {
    console.log('페이지 로드 완료 - 서버 생성 실시간 경로 이미지 모드');
    gameLoop();
});

function selectSeq(seq) {
    fetch('/change_seq', { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ seq: seq }) 
    })
    .then(r => r.json())
    .then(data => { 
        if (data.status === 'OK') { 
            currentSeq = seq; 
            refresh(); 
        } 
    });
}

function setQuickDest(x, z) {
    document.getElementById('dest-input').value = `${x}, ${z}`;
    setDestination();
}

function setDestination() {
    const input = document.getElementById('dest-input').value.trim();
    const status = document.getElementById('dest-status');
    const coords = input.replace(/[()]/g, '').split(',').map(s => parseFloat(s.trim()));
    
    if (coords.length !== 2 || coords.some(isNaN)) { 
        status.textContent = '❌ 형식 오류'; 
        return; 
    }
    
    fetch('/set_destination', {
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ destination: `${coords[0]},0,${coords[1]}` })
    })
    .then(r => r.json())
    .then(data => { 
        status.textContent = data.status === 'OK' ? `✅ (${coords[0]}, ${coords[1]})` : '❌ 실패'; 
        // 목적지 설정 후 즉시 이미지 갱신
        if (data.status === 'OK') {
            updateRealtimePathImage();
        }
    });
}

// 전투 액션 전송 (FIRE, RESCAN, RETREAT)
function sendCombatAction(action) {
    fetch('/combat_action', { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ action: action }) 
    });
}

// 타겟 재탐색
function handleRescan() {
    sendCombatAction('RESCAN');
}

// SCAN 탐색 방향 설정 함수
function setScanDir(dir) {
    fetch('/set_scan_direction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ direction: dir })
    })
    .then(res => res.json())
    .then(data => { if (data.status === 'OK') refresh(); })
    .catch(err => console.error("방향 설정 실패:", err));
}

// 전투 모드 변경 요청 함수
async function setSeq2Mode(mode) {
    try {
        const res = await fetch("/set_seq2_mode", {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify({ mode })
        });
        const j = await res.json();
        console.log("서버 모드 전환 응답:", j);
    } catch (e) {
        console.error("모드 전환 실패:", e);
    }
}

// 실시간 경로 이미지 업데이트 함수
function updateRealtimePathImage() {
    if (isImageLoading) return;

    const imgElement = document.getElementById('realtimePathImage');
    if (!imgElement) return;

    const now = Date.now();
    
    // 업데이트 간격 체크
    if (now - lastPathImageUpdate < 100) return;

    // 컨테이너 크기 계산
    const container = imgElement.parentElement;
    let width = container ? container.clientWidth : 640;
    let height = container ? container.clientHeight : 640;
    width = Math.max(400, Math.min(1000, width));
    height = Math.max(400, Math.min(1000, height));

    // 3. 로딩 시작 표시
    isImageLoading = true; 

    // 4. 새 이미지 객체를 미리 만들어서 로딩함 (깜빡임 방지 테크닉)
    const newImg = new Image();
    
    newImg.onload = () => {
        // 로딩이 성공하면 실제 화면에 반영
        imgElement.src = newImg.src;
        lastPathImageUpdate = Date.now();
        isImageLoading = false; // 락 해제
        
        // 다음 프레임을 위해 즉시 재호출 (애니메이션처럼 부드럽게)
        // 상황에 따라 requestAnimationFrame을 써도 됨
        setTimeout(updateRealtimePathImage, 50); 
    };

    newImg.onerror = () => {
        console.error("이미지 로드 실패, 재시도");
        isImageLoading = false; // 실패해도 락 해제
        setTimeout(updateRealtimePathImage, 500); // 0.5초 뒤 재시도
    };

    // 요청 시작
    newImg.src = `/realtime_path_image?width=${width}&height=${height}&t=${now}`;
}

function refresh() {
    const t = new Date().getTime();
    
    return fetch('/debug_status')
    .then(r => r.json())
    .then(j => {
        // UI 상태 업데이트
        const banner = document.getElementById('msg-banner');
        banner.textContent = j.msg || "CONNECTED";
        const serverSeq = j.seq;

        // [추가/수정] SEQ가 변경되었을 때 로그 초기화
        if (lastSeq !== null && lastSeq !== serverSeq) {
            console.log(`🔄 SEQ 변경 감지 (${lastSeq} -> ${serverSeq}): 로그 초기화`);
            const logArea = document.getElementById('driving-log');
            if (logArea) logArea.innerHTML = ''; // 로그 내용 비우기
            lastLogMsg = ""; // 중복 방지 메시지도 초기화
        }
        lastSeq = serverSeq; // 현재 SEQ 업데이트
        
        document.querySelectorAll('.layout-content').forEach(l => l.classList.remove('active'));
        
        // SEQ 상태 표시 업데이트
        const seqDisplay = document.getElementById('current-seq-display');
        if (seqDisplay) {
            if (serverSeq === 1){
                seqDisplay.textContent = '정찰지 이동';
            } else if (serverSeq === 2){
                seqDisplay.textContent = '전장 상황 인식';
            } else if (serverSeq === 3){
                seqDisplay.textContent = '경유지 이동';
            } else {
                seqDisplay.textContent = '베이스캠프로 복귀';
            }
            seqDisplay.className = `seq-display seq-${serverSeq}`;
        }
        
        const combatModeDisplay = document.getElementById('combat-mode-display');
        const combatModeBadge = document.getElementById('combat-mode-badge');

        if (serverSeq === 2) {
            combatModeDisplay.style.display = 'flex';
            const mode = j.combat_mode || 'SCAN';
            combatModeBadge.textContent = mode;
            combatModeBadge.className = 'mode-badge mode-' + mode.toLowerCase();
        } else {
            combatModeDisplay.style.display = 'none';
        }
        
        if (j.tank_pose) {
            const posText = `(${j.tank_pose[0].toFixed(1)}, ${j.tank_pose[1].toFixed(1)})`;
            const el = document.getElementById('header-current-pos');
            if (el) el.textContent = posText;
        }
        
        if (j.destination) {
            const destText = `(${j.destination[0].toFixed(1)}, ${j.destination[1].toFixed(1)})`;
            const el = document.getElementById('header-dest-pos');
            if (el) el.textContent = destText;
        } else {
            const el = document.getElementById('header-dest-pos');
            if (el) el.textContent = '-';
        }
        document.getElementById('destination-input').classList.toggle('active', serverSeq !== 2);

        // ═══════════════════════════════════════════════════════════════
        // SEQ 1, 3: 서버 생성 실시간 경로 이미지
        // ═══════════════════════════════════════════════════════════════
        if (serverSeq === 1 || serverSeq === 3) {
            document.getElementById('navigation-layout').classList.add('active');

            // 실시간 경로 이미지 업데이트
            updateRealtimePathImage();

            const currentNodes = j.path_nodes || 0;
            if (currentNodes !== lastPathNodeCount && currentNodes > 0) {
                console.log(`📡 경로 변경 감지! (${lastPathNodeCount} -> ${currentNodes}) 이미지 새로고침`);
                staticPathTimestamp = new Date().getTime(); // 타임스탬프 갱신으로 강제 새로고침 트리거
                lastPathNodeCount = currentNodes;
            }
            
            const pipImg = document.getElementById('staticPathPip');
            if (pipImg) {
                // 현재 SEQ에 맞는 정적 경로 이미지 URL 설정
                // app.py의 /get_static_path/<seq> 엔드포인트 활용
                const targetSrc = `/get_static_path/${serverSeq}?t=${staticPathTimestamp}`;
                
                // src가 바뀌었을 때만 업데이트 (깜빡임 방지)
                if (!pipImg.src.endsWith(targetSrc) && pipImg.getAttribute('src') !== targetSrc) {
                    pipImg.src = targetSrc;
                }
                pipImg.style.display = 'block';
            }

            // 경로 노드 정보 업데이트
            const currentNodeIdx = j.current_node !== undefined ? j.current_node : '-';
            const totalNodes = j.path_nodes !== undefined ? j.path_nodes : '-';
            document.getElementById('path-node-info').textContent = `${currentNodeIdx}/${totalNodes}`;

            if (j.path_nodes && j.current_node) {
                document.getElementById('path-node-info').textContent = `${j.current_node}/${j.path_nodes}`;
            } else if (j.path_nodes) {
                document.getElementById('path-node-info').textContent = `-/${j.path_nodes}`;
            } else {
                document.getElementById('path-node-info').textContent = '-/-';
            }

            // 로그 업데이트
            if (j.log && j.log !== lastLogMsg) {
                const logArea = document.getElementById('driving-log');
                if (logArea) {
                    logArea.innerHTML = `[${new Date().toLocaleTimeString()}] ${j.log}\n` + logArea.innerHTML;
                    lastLogMsg = j.log;
                }
            }
        } 
        // ═══════════════════════════════════════════════════════════════
        // SEQ 2: 전투
        // ═══════════════════════════════════════════════════════════════
        else if (serverSeq === 2) {
            document.getElementById('combat-layout').classList.add('active');
            document.getElementById('combat-overlay').src = '/overlay/left?t=' + t;
            
            const combatMode = j.combat_mode || 'SCAN';
            
            // SCAN 방향 선택 버튼 제어 로직 추가
            const scanQBtn = document.getElementById('scan-q-btn');
            const scanEBtn = document.getElementById('scan-e-btn');
            const scanDirCard = document.getElementById('scan-direction-card');
            const fireReady = j.fire_ready || false;
            const lockedTarget = j.locked_target;
            const hasTarget = lockedTarget && lockedTarget.bbox;
            const autoAttack = j.auto_attack_active || false;
            
            // 서버에서 받은 타겟 목록 (SCAN 결과 + is_locked 플래그 포함)
            const targets = j.detected_targets || [];
            
            // 버튼 상태 업데이트
            const standbyBtn = document.getElementById('standby-btn');
            const rescanBtn = document.getElementById('rescan-btn');
            const retreatBtn = document.getElementById('retreat-btn');
            const fireBtn = document.getElementById('fire-btn');
            const actionStatus = document.getElementById('action-status-text');
            
            // SCAN 또는 RESCAN(서버에서는 결국 SCAN 모드)일 때만 활성화
            // if (combatMode === 'SCAN') {
            //     scanQBtn.disabled = false;
            //     scanEBtn.disabled = false;
            //     scanDirCard.style.opacity = "1.0"; // 시각적으로 활성화 표시
            // } else {
            //     scanQBtn.disabled = true;
            //     scanEBtn.disabled = true;
            //     scanDirCard.style.opacity = "0.5"; // 비활성화 시 흐리게 처리
            // }
            
            // / 상황인식판 요소 가져오기
            const situationCard = document.getElementById('situation-awareness-card');
            const tankCountDisplay = document.getElementById('tank-count-display');
            const redCountDisplay = document.getElementById('red-count-display');

            // SCAN 방향이 선택되었는지 확인
            const scanDirectionSet = j.scan_direction !== null && j.scan_direction !== undefined;

            if (combatMode === 'SCAN') {
                if (!scanDirectionSet) {
                    // 방향 선택 전: SCAN 방향 카드 표시, 상황인식판 숨김
                    scanDirCard.style.display = "block";
                    situationCard.style.display = "none";
                    scanQBtn.disabled = false;
                    scanEBtn.disabled = false;
                } else {
                    // 방향 선택 후: SCAN 방향 카드 숨김, 상황인식판 표시
                    scanDirCard.style.display = "none";
                    situationCard.style.display = "block";
                    
                    // Tank, RED 개수 계산
                    let tankCount = 0;
                    let redCount = 0;
                    targets.forEach(t => {
                        const className = (t.className || t.category || '').toLowerCase();
                        if (className === 'tank') tankCount++;
                        if (className === 'red') redCount++;
                    });
                    
                    tankCountDisplay.textContent = tankCount;
                    redCountDisplay.textContent = redCount;
                }
            } else {
                // SCAN 모드가 아닐 때 (STANDBY 등)
                scanDirCard.style.display = "none";
                situationCard.style.display = "block";
                scanQBtn.disabled = true;
                scanEBtn.disabled = true;
                
                // Tank, RED 개수 계산
                let tankCount = 0;
                let redCount = 0;
                targets.forEach(t => {
                    const className = (t.className || t.category || '').toLowerCase();
                    if (className === 'tank') tankCount++;
                    if (className === 'red') redCount++;
                });
                
                tankCountDisplay.textContent = tankCount;
                redCountDisplay.textContent = redCount;
            }

            // STANDBY 버튼: SCAN 모드일 때만 활성화
            if (combatMode === 'SCAN') {
                standbyBtn.disabled = false;
                standbyBtn.classList.remove('active-mode');
            } else {
                standbyBtn.disabled = true;
                standbyBtn.classList.add('active-mode');
            }
            
            // RESCAN, RETREAT 버튼: SCAN (적군 확인 되었을때)과 STANDBY 모드일 때 활성화
            const hasEnemies = targets.length > 0;
            const isCombatReady = ((combatMode === 'SCAN' && hasEnemies) || combatMode === 'STANDBY');
            rescanBtn.disabled = !isCombatReady;
            retreatBtn.disabled = !isCombatReady;         
       
            // 공격 버튼 활성화 로직 (단순화) (0130 추가)
            if (serverSeq === 2 && j.combat_mode === 'STANDBY') {
                fireBtn.disabled = false;
                fireBtn.classList.add('ready');
                fireBtn.textContent = "🔥 포격";  // 항상 고정
            } else {
                fireBtn.disabled = true;
                fireBtn.classList.remove('ready');
                fireBtn.textContent = "🔥 포격";  // 항상 고정
            }

            // 상태 텍스트 업데이트 로직 수정
            let newStatusText = "";
            let newStatusColor = "";

            if (combatMode === 'SCAN') {
                if (!j.scan_direction) {
                    newStatusText = '📡 방향(Q/E)을 선택하세요';
                    newStatusColor = '#2196F3';
                } else {
                    newStatusText = '🔍 객체 식별 중...';
                    newStatusColor = '#2196F3';
                }
            } else if (combatMode === 'STANDBY') {
                if (fireReady) {
                    newStatusText = '🎯 타겟 락온 완료 - FIRE 가능!';
                    newStatusColor = '#f44336';
                } else if (hasTarget) {
                    newStatusText = '⏳ 타겟 조준 중...';
                    newStatusColor = '#FF9800';
                } else {
                    newStatusText = '🔒 STANDBY 모드 - 타겟 대기 중...';
                    newStatusColor = '#4CAF50';
                }
            }

            // 텍스트가 변경되었을 때만 DOM 업데이트 실행
            if (newStatusText !== lastActionStatus) {
                actionStatus.textContent = newStatusText;
                actionStatus.style.color = newStatusColor;
                lastActionStatus = newStatusText; // 현재 상태 저장
            }

            // 타겟 카운트 업데이트
            document.getElementById('target-count').textContent = `(${targets.length})`;
            
            // 락된 타겟 정보 업데이트
            if (lockedTarget) {
                document.getElementById('lock-distance').textContent = 
                    lockedTarget.distance_m ? `${lockedTarget.distance_m.toFixed(1)}m` : '-';
                    // 전달받은 정밀 좌표(XYZ) 출력
                if (lockedTarget.position) {
                    const {x, y, z} = lockedTarget.position;
                    document.getElementById('lock-pos').textContent = `X:${x}, Y:${y}, Z:${z}`;
                } else {
                    document.getElementById('lock-pos').textContent = '-';
                }

                document.getElementById('lock-yaw').textContent = 
                    lockedTarget.yaw_error_deg !== undefined ? `${lockedTarget.yaw_error_deg.toFixed(1)}°` : '-';
                
                // 락된 타겟 카드 하이라이트
                document.getElementById('locked-target-card').style.borderColor = '#d16666';
            } else {
                document.getElementById('lock-distance').textContent = '-';
                document.getElementById('lock-yaw').textContent = '-';
                document.getElementById('locked-target-card').style.borderColor = '#333';
            }
            
            // ✅ 탐지된 타겟 리스트 업데이트 (서버에서 is_locked 플래그 사용)
            const targetList = document.getElementById('target-list');
            targetList.innerHTML = targets.slice(0, 10).map((t, i) => {
                const isLocked = t.is_locked || false;  // 서버에서 계산된 값 사용
                const dist = t.distance_m ? `${t.distance_m.toFixed(1)}m` : '';
                const className = t.className || t.category || 'Unknown';
                
                // ✅ locked 타겟은 'target-locked' 클래스 (빨간색)
                const itemClass = isLocked ? 'target-item target-locked' : 'target-item';
                const icon = isLocked ? '🔴' : '🔘';
                
                return `<div class="${itemClass}">
                    ${icon} ${className} ${dist}
                </div>`;
            }).join('');
        } 
        // ═══════════════════════════════════════════════════════════════
        // SEQ 4: 자율주행 - Canvas 렌더링
        // ═══════════════════════════════════════════════════════════════
        else if (serverSeq === 4) {
            const layout = document.getElementById('autonomous-layout');
            layout.classList.add('active');

            // Canvas 렌더링 초기화 (최초 1회)
            if (!SEQ4.canvas) {
                SEQ4.init();
            }

            // 데이터 업데이트
            SEQ4.updateData(j);
        }
    })
    .catch(err => {
        console.error('디버그 상태 오류:', err);
    });
}

function gameLoop() {
    refresh().finally(() => {
        setTimeout(gameLoop, 150);
    });
}