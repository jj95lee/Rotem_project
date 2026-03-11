/* script.js - 완전 수정 (이미지 풀스크린, 화살표 정확 위치) */
let currentSeq = 1;
let lastLogMsg = "";
let lastImageUpdate = 0;
let pathImageData = null;
let pathImageVersion = 0;

let canvas = null;
let ctx = null;
let canvasWidth = 0;
let canvasHeight = 0;

// ✅ 서버에서 동적으로 받을 좌표 범위
let pathImageBounds = { 
    x_min: 75, x_max: 200,
    z_min: 0, z_max: 300
};

// 이미지 정보 (렌더링용)
let imgWidth = 0;
let imgHeight = 0;
let offsetX = 0;
let offsetY = 0;

window.addEventListener('load', () => {
    console.log('페이지 로드 완료');
    canvas = document.getElementById('pathCanvas');
    ctx = canvas.getContext('2d');
    initCanvas();
    refresh();
});

function initCanvas() {
    const container = canvas.parentElement;
    canvasWidth = container.clientWidth - 20;
    canvasHeight = container.clientHeight - 20;
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    console.log(`캔버스 초기화: ${canvasWidth}x${canvasHeight}`);
}

window.addEventListener('resize', () => {
    if (currentSeq === 1 || currentSeq === 3) {
        initCanvas();
        renderPathCanvas();
    }
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
            if ((seq === 1 || seq === 3) && !canvas) {
                setTimeout(() => {
                    canvas = document.getElementById('pathCanvas');
                    ctx = canvas.getContext('2d');
                    initCanvas();
                }, 100);
            }
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
    });
}

function sendCombatAction(action) {
    fetch('/combat_action', { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ action: action }) 
    });
}

function handleRescan() {
    sendCombatAction('RESCAN');
}

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

// ✅ 캔버스 렌더링: 이미지를 전체 캔버스에 풀스크린으로 표시
function renderPathCanvas() {
    if (!ctx || !canvas) return;

    // 1. 캔버스 클리어
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // 2. 경로 이미지를 전체 캔버스에 꽉 채워서 표시
    if (pathImageData) {
        try {
            const img = new Image();
            img.onload = () => {
                // ✅ 이미지를 캔버스 전체에 꽉 채우기 (aspect ratio 무시)
                ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight);
                
                // 이미지 정보 저장 (화살표 위치 계산용)
                imgWidth = img.width;
                imgHeight = img.height;
                offsetX = 0;
                offsetY = 0;
                
                // ✅ 3. 갈매기 화살표 그리기
                drawRobotSeagull();
            };
            img.src = 'data:image/png;base64,' + pathImageData;
        } catch (e) {
            console.error('경로 이미지 그리기 오류:', e);
            ctx.fillStyle = '#ccc';
            ctx.font = '16px Arial';
            ctx.fillText('경로 이미지 로딩 중...', 20, 30);
        }
    } else {
        ctx.fillStyle = '#ccc';
        ctx.font = '16px Arial';
        ctx.fillText('경로를 기다리는 중...', 20, 30);
    }
}

// ✅ 갈매기 화살표 그리기 (정확한 위치)
function drawRobotSeagull() {
    if (!ctx) return;

    // 현재 로봇 위치
    let robotX = 130;
    let robotZ = 30;
    
    const posText = document.getElementById('current-pos').textContent;
    if (posText && posText !== '-') {
        const coords = posText.replace(/[()]/g, '').split(',');
        if (coords.length === 2) {
            robotX = parseFloat(coords[0]) || 130;
            robotZ = parseFloat(coords[1]) || 30;
        }
    }

    // ✅ 핵심: 좌표 범위를 캔버스 크기로 직접 매핑
    // 이미지가 캔버스 전체를 채우므로, 비율로 직접 계산
    const xRatio = (robotX - pathImageBounds.x_min) / (pathImageBounds.x_max - pathImageBounds.x_min);
    const zRatio = (robotZ - pathImageBounds.z_min) / (pathImageBounds.z_max - pathImageBounds.z_min);
    
    // ✅ 캔버스의 픽셀 좌표로 변환
    const canvasX = xRatio * canvasWidth;
    const canvasY = zRatio * canvasHeight;

    console.log(`🚗 로봇: (${robotX}, ${robotZ}) → 캔버스 (${canvasX.toFixed(1)}, ${canvasY.toFixed(1)})`);

    // ✅ 갈매기 화살표 (아래를 가리킴)
    ctx.save();
    ctx.translate(canvasX, canvasY);

    // 갈매기 모양 (크기 25px)
    const size = 25;
    ctx.fillStyle = 'red';
    ctx.beginPath();
    ctx.moveTo(0, size);       // ✅ 아래쪽이 뾰족함 (아래로 향함)
    ctx.lineTo(-size/2, -size/2);  // 좌측 날개
    ctx.lineTo(0, -size/3);    // 중앙
    ctx.lineTo(size/2, -size/2);   // 우측 날개
    ctx.closePath();
    ctx.fill();

    // 테두리
    ctx.strokeStyle = 'darkred';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.restore();
}

function refresh() {
    const t = new Date().getTime();
    fetch('/debug_status')
    .then(r => r.json())
    .then(j => {
        const banner = document.getElementById('msg-banner');
        banner.textContent = j.msg || "CONNECTED";
        const serverSeq = j.seq;
        
        document.querySelectorAll('.layout-content').forEach(l => l.classList.remove('active'));
        document.querySelectorAll('.seq-btn').forEach(b => b.classList.toggle('active', parseInt(b.dataset.seq) === serverSeq));
        
        const combatModeDisplay = document.getElementById('combat-mode-display');
        if (serverSeq === 2) {
            combatModeDisplay.style.display = 'flex';
            const mode = j.combat_mode || 'SCAN';
            document.getElementById('combat-mode-badge').textContent = mode;
            document.getElementById('combat-mode-badge').className = 'mode-badge mode-' + mode.toLowerCase();
        } else {
            combatModeDisplay.style.display = 'none';
        }
        
        document.getElementById('position-panel').classList.toggle('hidden', serverSeq === 2);
        document.getElementById('destination-input').classList.toggle('active', serverSeq !== 2);

        // SEQ 1, 3: 경로 표시
        if (serverSeq === 1 || serverSeq === 3) {
            document.getElementById('navigation-layout').classList.add('active');

            // ✅ 서버에서 받은 좌표 범위 적용
            if (j.path_bounds) {
                pathImageBounds = j.path_bounds;
                console.log(`📊 SEQ ${serverSeq} 좌표 범위: X(${j.path_bounds.x_min}~${j.path_bounds.x_max}), Z(${j.path_bounds.z_min}~${j.path_bounds.z_max})`);
            }

            // ✅ 경로 이미지 다운로드
            if (j.global_path_version !== pathImageVersion) {
                fetch(`/get_static_path/${serverSeq}?t=${t}`)
                    .then(res => res.arrayBuffer())
                    .then(buffer => {
                        const bytes = new Uint8Array(buffer);
                        let binary = '';
                        for (let i = 0; i < bytes.byteLength; i++) {
                            binary += String.fromCharCode(bytes[i]);
                        }
                        pathImageData = btoa(binary);
                        pathImageVersion = j.global_path_version;
                        renderPathCanvas();
                    })
                    .catch(err => console.error('경로 이미지 로딩 오류:', err));
            } else {
                renderPathCanvas();
            }

            // 경로 노드 정보
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
                logArea.innerHTML = `[${new Date().toLocaleTimeString()}] ${j.log}\n` + logArea.innerHTML;
                lastLogMsg = j.log;
            }
        } 
        // SEQ 2: 전투
        else if (serverSeq === 2) {
            document.getElementById('combat-layout').classList.add('active');
            
            const now = Date.now();
            if (now - lastImageUpdate > 100) {
                document.getElementById('combat-overlay').src = '/overlay/left?t=' + t + '&nc=' + Math.random();
                lastImageUpdate = now;
            }
            
            const combatMode = j.combat_mode || 'SCAN';
            const targets = j.detected_targets || [];
            const fireReady = j.fire_ready || false;
            const lockedTarget = j.locked_target;
            const hasTarget = lockedTarget && lockedTarget.bbox;
            const autoAttack = j.auto_attack_active || false;
            
            const scanQBtn = document.getElementById('scan-q-btn');
            const scanEBtn = document.getElementById('scan-e-btn');
            const scanDirCard = document.getElementById('scan-direction-card');
            
            if (combatMode === 'SCAN') {
                scanQBtn.disabled = false;
                scanEBtn.disabled = false;
                scanDirCard.style.opacity = "1.0";
            } else {
                scanQBtn.disabled = true;
                scanEBtn.disabled = true;
                scanDirCard.style.opacity = "0.5";
            }

            const standbyBtn = document.getElementById('standby-btn');
            const rescanBtn = document.getElementById('rescan-btn');
            const retreatBtn = document.getElementById('retreat-btn');
            const fireBtn = document.getElementById('fire-btn');
            
            if (combatMode === 'SCAN') {
                standbyBtn.disabled = false;
                standbyBtn.classList.remove('active-mode');
            } else {
                standbyBtn.disabled = true;
                standbyBtn.classList.add('active-mode');
            }
            
            const hasEnemies = targets.length > 0;
            const isCombatReady = ((combatMode === 'SCAN' && hasEnemies) || combatMode === 'STANDBY');
            rescanBtn.disabled = !isCombatReady;
            retreatBtn.disabled = !isCombatReady;
            
            if (serverSeq === 2 && j.combat_mode === 'STANDBY') {
                fireBtn.disabled = false; 
                if (autoAttack) {
                    fireBtn.classList.add('active-attack');
                    fireBtn.classList.remove('ready');
                    fireBtn.textContent = "🎯 조준 정렬 대기 중...";
                } else if (fireReady) {
                    fireBtn.classList.add('ready');
                    fireBtn.classList.remove('active-attack');
                    fireBtn.textContent = "🔥 즉시 발사 가능";
                } else {
                    fireBtn.classList.remove('ready', 'active-attack');
                    fireBtn.textContent = "🚀 자동 포격 시작";
                }
            } else {
                fireBtn.disabled = true;
                fireBtn.textContent = "🔥 포격";
            }
            
            const actionStatus = document.getElementById('action-status-text');
            if (combatMode === 'SCAN') {
                if (!j.scan_direction) {
                    actionStatus.textContent = '📡 방향(Q/E)을 선택하세요';
                    actionStatus.style.color = '#2196F3';
                } else {
                    actionStatus.textContent = '🔍 객체 식별 중...';
                    actionStatus.style.color = '#2196F3';
                }
            } else if (combatMode === 'STANDBY') {
                if (fireReady) {
                    actionStatus.textContent = '🎯 타겟 락온 완료 - FIRE 가능!';
                    actionStatus.style.color = '#f44336';
                } else if (hasTarget) {
                    actionStatus.textContent = '⏳ 타겟 조준 중...';
                    actionStatus.style.color = '#FF9800';
                } else {
                    actionStatus.textContent = '🔒 STANDBY 모드 - 타겟 대기 중...';
                    actionStatus.style.color = '#4CAF50';
                }
            }
            
            document.getElementById('target-count').textContent = `(${targets.length})`;
            
            if (lockedTarget) {
                document.getElementById('lock-distance').textContent = 
                    lockedTarget.distance_m ? `${lockedTarget.distance_m.toFixed(1)}m` : '-';
                document.getElementById('lock-yaw').textContent = 
                    lockedTarget.yaw_error_deg !== undefined ? `${lockedTarget.yaw_error_deg.toFixed(1)}°` : '-';
                document.getElementById('lock-conf').textContent = 
                    lockedTarget.confidence ? `${(lockedTarget.confidence * 100).toFixed(0)}%` : '-';
                document.getElementById('locked-target-card').style.borderColor = '#d16666';
            } else {
                document.getElementById('lock-distance').textContent = '-';
                document.getElementById('lock-yaw').textContent = '-';
                document.getElementById('lock-conf').textContent = '-';
                document.getElementById('locked-target-card').style.borderColor = '#333';
            }
            
            const targetList = document.getElementById('target-list');
            targetList.innerHTML = targets.slice(0, 10).map((t, i) => {
                const isLocked = t.is_locked || false;
                const dist = t.distance_m ? `${t.distance_m.toFixed(1)}m` : '';
                const className = t.className || t.category || 'Unknown';
                const itemClass = isLocked ? 'target-item target-locked' : 'target-item';
                const icon = isLocked ? '🔴' : '🔘';
                return `<div class="${itemClass}">${icon} ${className} ${dist}</div>`;
            }).join('');
        } 
        // SEQ 4: 자율주행
        else if (serverSeq === 4) {
            document.getElementById('autonomous-layout').classList.add('active');
            document.getElementById('autonomous-view').src = '/view_autonomous?t=' + t;
            document.getElementById('autonomous-costmap-global').src = '/view_autonomous?t=' + t;
        }

        // 공통 정보
        if (j.tank_pose) document.getElementById('current-pos').textContent = `(${j.tank_pose[0].toFixed(1)}, ${j.tank_pose[1].toFixed(1)})`;
        if (j.destination) document.getElementById('destination-pos').textContent = `(${j.destination[0].toFixed(1)}, ${j.destination[1].toFixed(1)})`;
        document.getElementById('path-nodes').textContent = j.path_nodes ? `${j.path_nodes}개` : '0';
    })
    .catch(err => console.error('디버그 상태 오류:', err));
}

// 정기 갱신 (150ms)
setInterval(() => {
    refresh();
}, 150);
