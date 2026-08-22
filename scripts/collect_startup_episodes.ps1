$ErrorActionPreference = "Stop"

$sshTarget = "jetson@192.168.137.100"
$repo = "D:\asv-vla-training"
$episodeRoot = Join-Path $repo "data\episodes\moving_target_valid"
$rawRoot = Join-Path $repo "data\raw\startup"
$remoteRoot = "/tmp/asv_moving_startup_20260820"
$ue = "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"
$project = "D:\asv-unreal-simulation\VLA.uproject"

$scenarios = @(
    @{ Slot = "RED_3M_TRAIN_START_01"; Color = "red"; Distance = 3; Layout = "L7"; Seed = 232101 },
    @{ Slot = "RED_3M_TRAIN_START_02"; Color = "red"; Distance = 3; Layout = "L7B"; Seed = 232102 },
    @{ Slot = "BLUE_3M_TRAIN_START_01"; Color = "blue"; Distance = 3; Layout = "L7"; Seed = 232201 },
    @{ Slot = "BLUE_3M_TRAIN_START_02"; Color = "blue"; Distance = 3; Layout = "L7B"; Seed = 232202 },
    @{ Slot = "RED_4M_TRAIN_START_01"; Color = "red"; Distance = 4; Layout = "L7"; Seed = 232301 },
    @{ Slot = "RED_4M_TRAIN_START_02"; Color = "red"; Distance = 4; Layout = "L7B"; Seed = 232302 }
)

function Invoke-Remote([string]$Command) {
    & ssh -o BatchMode=yes $sshTarget $Command
    if ($LASTEXITCODE -ne 0) { throw "remote command failed: $LASTEXITCODE" }
}

New-Item -ItemType Directory -Force -Path $rawRoot | Out-Null
foreach ($scenario in $scenarios) {
    if (Test-Path (Join-Path $episodeRoot $scenario.Slot)) {
        throw "refusing to overwrite $($scenario.Slot)"
    }
}

Invoke-Remote "set -e; test ! -e $remoteRoot; mkdir -p $remoteRoot; cd /home/jetson/jetson_asv_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; setsid ros2 run bridge bridge_node --ros-args --params-file src/bridge/config/ue_bridge.yaml -p outbound_command_mode:=disabled -p use_sim_time:=true > $remoteRoot/bridge.log 2>&1 < /dev/null & echo `$! > $remoteRoot/bridge.pid; sleep 3"

try {
    foreach ($scenario in $scenarios) {
        $slot = $scenario.Slot
        $remoteEpisode = "$remoteRoot/$slot"
        $task = "follow the $($scenario.Color) boat, keep $($scenario.Distance) meters distance"
        Write-Host "STARTUP_COLLECTION_START $slot"
        Invoke-Remote "set -e; mkdir -p $remoteEpisode; cd /home/jetson/jetson_asv_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; setsid ros2 bag record -o $remoteEpisode/bag /ue/asv_state /ue/camera_frame /ue/entities > $remoteEpisode/bag.log 2>&1 < /dev/null & echo `$! > $remoteEpisode/bag.pid; sleep 3"
        $args = @(
            $project, "Main_Map", "-game", "-SceneAuto", "-unattended", "-RenderOffscreen", "-nosplash",
            "-benchmark", "-fps=10", "-Slot=$slot", "-Layout=$($scenario.Layout)", "-Motion=S2", "-Seed=$($scenario.Seed)",
            "-MaxRuntimeSeconds=30", "-RenderWarmupSeconds=1", "-SineDelay=0",
            "-ExpertFollowColor=$($scenario.Color)", "-ExpertStandoffM=$($scenario.Distance)",
            "-ExpertMaxStepCm=30", "-ExpertMaxAccelerationCmPerSec2=20", "-log"
        )
        $process = Start-Process -FilePath $ue -ArgumentList $args -WindowStyle Hidden -PassThru -Wait
        if ($process.ExitCode -ne 0) { throw "UE5 failed for $slot" }
        Invoke-Remote "set -e; kill -INT `$(cat $remoteEpisode/bag.pid); sleep 4; cd /home/jetson/jetson_asv_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; python3 /tmp/export_moving_target_bag.py --bag $remoteEpisode/bag --output $remoteEpisode/episode --task '$task' --slot $slot --layout $($scenario.Layout) --motion S2; tar -czf $remoteRoot/$slot.tar.gz -C $remoteEpisode episode"
        $archive = Join-Path $rawRoot "$slot.tar.gz"
        & scp -q "$sshTarget`:$remoteRoot/$slot.tar.gz" $archive
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $slot" }
        $destination = Join-Path $episodeRoot $slot
        New-Item -ItemType Directory -Path $destination | Out-Null
        & tar -xzf $archive -C $destination --strip-components=1
        if ($LASTEXITCODE -ne 0) { throw "extract failed for $slot" }
        $manifest = Get-Content (Join-Path $destination "manifest.json") -Raw | ConvertFrom-Json
        Write-Host "STARTUP_COLLECTION_PASS $slot frames=$($manifest.frame_count)"
    }
}
finally {
    Invoke-Remote "set +e; kill -INT `$(cat $remoteRoot/bridge.pid) 2>/dev/null || true"
}

Write-Host "STARTUP_COLLECTION_COMPLETE"
