# script: test_vm.ps1
# Proposito: Validar la instalacion, ejecucion y desinstalacion en una VM Windows Virgen

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   Prueba en VM: AI Torrent Protocol v1.2    " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

$localAppData = [Environment]::GetFolderPath("LocalApplicationData")
$logPath = Join-Path $localAppData "GhostTerminal\worker.log"
$regPath = "HKCU:\Environment"

Write-Host "`n[*] Paso 1: Por favor, ejecuta 'Ghost Terminal Setup.exe' manualmente."
Write-Host "[*] Completa el asistente de instalacion y dale permisos de Administrador."
Pause

# 1. Validar Variables de Entorno (HKCU)
Write-Host "`n[1] Comprobando persistencia en el Registro (Variables de Entorno)..." -ForegroundColor Yellow
$token = (Get-ItemProperty -Path $regPath -Name "AI_TORRENT_AUTH_TOKEN" -ErrorAction SilentlyContinue).AI_TORRENT_AUTH_TOKEN
if ($token) {
    Write-Host "[PASS] AI_TORRENT_AUTH_TOKEN inyectado correctamente en HKCU." -ForegroundColor Green
} else {
    Write-Host "[FAIL] No se encontro AI_TORRENT_AUTH_TOKEN." -ForegroundColor Red
}

# 2. Validar Logging Oculto (--noconsole)
Write-Host "`n[2] Comprobando creacion de logs de worker.exe (--noconsole)..." -ForegroundColor Yellow
Start-Sleep -Seconds 3 # Dar tiempo a que el worker levante
if (Test-Path $logPath) {
    Write-Host "[PASS] El log del worker se creo correctamente en $logPath." -ForegroundColor Green
    Write-Host "-> Ultimas lineas del log:"
    Get-Content $logPath -Tail 5 | ForEach-Object { Write-Host "   $_" -ForegroundColor DarkGray }
} else {
    Write-Host "[FAIL] No se encontro el archivo de log en $logPath. worker.exe podria haber fallado." -ForegroundColor Red
}

# 3. Validar procesos levantados
Write-Host "`n[3] Comprobando procesos activos..." -ForegroundColor Yellow
$workerProc = Get-Process -Name "p2p_node" -ErrorAction SilentlyContinue
if ($workerProc) {
    Write-Host "[PASS] p2p_node.exe (Worker P2P) esta corriendo en segundo plano." -ForegroundColor Green
} else {
    Write-Host "[FAIL] No se encontro el proceso p2p_node.exe corriendo." -ForegroundColor Red
}

Write-Host "`n[*] Paso 2: Por favor, ejecuta el desinstalador desde 'Ghost Terminal/uninstaller.exe'."
Pause

# 4. Validar Desinstalacion
Write-Host "`n[4] Comprobando limpieza de desinstalador..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
$tokenAfter = (Get-ItemProperty -Path $regPath -Name "AI_TORRENT_AUTH_TOKEN" -ErrorAction SilentlyContinue).AI_TORRENT_AUTH_TOKEN
if ($tokenAfter) {
    Write-Host "[FAIL] AI_TORRENT_AUTH_TOKEN sigue en el registro despues de desinstalar." -ForegroundColor Red
} else {
    Write-Host "[PASS] AI_TORRENT_AUTH_TOKEN eliminado correctamente." -ForegroundColor Green
}

$workerProcAfter = Get-Process -Name "p2p_node" -ErrorAction SilentlyContinue
if ($workerProcAfter) {
    Write-Host "[FAIL] p2p_node.exe sigue corriendo despues de desinstalar." -ForegroundColor Red
} else {
    Write-Host "[PASS] p2p_node.exe cerrado correctamente." -ForegroundColor Green
}

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "             PRUEBAS FINALIZADAS             " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
