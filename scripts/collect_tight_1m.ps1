$ErrorActionPreference = "Stop"

$sshTarget = "jetson@192.168.137.100"
$repo = "D:\asv-vla-training"
$episodeRoot = Join-Path $repo "data\episodes\moving_target_valid"
$remoteRoot = "/tmp/asv_moving_collection_tight_1m_20260825"
$ue = "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"
$project = "D:\asv-unreal-simulation\VLA.uproject"
$runtimeSec = 180

function Invoke-Remote([string]$Command) {
    Write-Host "+ ssh $Command"
    & ssh -o BatchMode=yes $sshTarget $Command
    if ($LASTEXITCODE -ne 0) {
        throw "remote command failed with exit code $LASTEXITCODE"
    }
}

$scenarios = @(
    @{ Slot = "RED_3M_TRAIN_09"; Color = "red"; Distance = 3; Layout = "L7"; Seed = 241101; Accel = 1200 },
    @{ Slot = "RED_3M_TRAIN_10"; Color = "red"; Distance = 3; Layout = "L7B"; Seed = 241102; Accel = 1200 },
    @{ Slot = "RED_3M_TRAIN_11"; Color = "red"; Distance = 3; Layout = "L7"; Seed = 241103; Accel = 80 },
    @{ Slot = "RED_3M_TRAIN_12"; Color = "red"; Distance = 3; Layout = "L7B"; Seed = 241104; Accel = 1200 },
    @{ Slot = "BLUE_3M_TRAIN_09"; Color = "blue"; Distance = 3; Layout = "L7"; Seed = 241201; Accel = 1200 },
    @{ Slot = "BLUE_3M_TRAIN_10"; Color = "blue"; Distance = 3; Layout = "L7B"; Seed = 241202; Accel = 1200 },
    @{ Slot = "BLUE_3M_TRAIN_11"; Color = "blue"; Distance = 3; Layout = "L7"; Seed = 241203; Accel = 80 },
    @{ Slot = "BLUE_3M_TRAIN_12"; Color = "blue"; Distance = 3; Layout = "L7B"; Seed = 241204; Accel = 1200 },
    @{ Slot = "RED_4M_TRAIN_09"; Color = "red"; Distance = 4; Layout = "L7"; Seed = 241301; Accel = 1200 },
    @{ Slot = "RED_4M_TRAIN_10"; Color = "red"; Distance = 4; Layout = "L7B"; Seed = 241302; Accel = 1200 },
    @{ Slot = "RED_4M_TRAIN_11"; Color = "red"; Distance = 4; Layout = "L7"; Seed = 241303; Accel = 80 },
    @{ Slot = "RED_4M_TRAIN_12"; Color = "red"; Distance = 4; Layout = "L7B"; Seed = 241304; Accel = 1200 }
)

& scp -o BatchMode=yes (Join-Path $repo "src\export_moving_target_bag.py") "${sshTarget}:/tmp/export_moving_target_bag.py"
& scp -o BatchMode=yes (Join-Path $repo "scripts\jetson_collect_bridge_start.sh") "${sshTarget}:/tmp/jetson_collect_bridge_start.sh"
Invoke-Remote "sed -i 's/\r$//' /tmp/jetson_collect_bridge_start.sh /tmp/export_moving_target_bag.py; chmod +x /tmp/jetson_collect_bridge_start.sh; REMOTE_ROOT=$remoteRoot bash /tmp/jetson_collect_bridge_start.sh"

try {
    foreach ($scenario in $scenarios) {
        $slot = $scenario.Slot
        $localEpisode = Join-Path $episodeRoot $slot
        if (Test-Path $localEpisode) {
            Write-Host "SKIP_EXISTS $slot"
            continue
        }
        $color = $scenario.Color
        $distance = $scenario.Distance
        $layout = $scenario.Layout
        $seed = $scenario.Seed
        $accel = $scenario.Accel
        $followSlot = if ($color -eq "blue") { 1 } else { 0 }
        $remoteEpisode = "$remoteRoot/$slot"
        $task = "follow the $color boat, keep $distance meters distance"
        $archive = "$remoteRoot/$slot.tar.gz"
        $localArchive = Join-Path $repo "data\raw\$slot.tar.gz"

        Write-Host "COLLECT_START $slot seed=$seed accel=$accel"
        Invoke-Remote "rm -rf $remoteEpisode; mkdir -p $remoteEpisode; cd /home/jetson/jetson_asv_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; nohup ros2 bag record -o $remoteEpisode/bag /ue/asv_state /ue/camera_frame /ue/entities > $remoteEpisode/bag.log 2>&1 < /dev/null & echo `$! > $remoteEpisode/bag.pid; sleep 3"

        $ueArgs = @(
            $project, "Main_Map", "-game", "-SceneAuto", "-unattended", "-RenderOffscreen", "-nosplash",
            "-benchmark", "-fps=10", "-Slot=$slot", "-Layout=$layout", "-Motion=S2", "-Seed=$seed",
            "-MaxRuntimeSeconds=$runtimeSec", "-RenderWarmupSeconds=1", "-SineDelay=0",
            "-ExpertFollowColor=$color", "-ExpertStandoffM=$distance", "-ExpertMaxStepCm=30",
            "-ExpertMaxAccelerationCmPerSec2=$accel", "-log"
        )
        $process = Start-Process -FilePath $ue -ArgumentList $ueArgs -WindowStyle Hidden -PassThru -Wait
        if ($process.ExitCode -ne 0) {
            throw "UE5 failed for $slot with exit code $($process.ExitCode)"
        }

        Invoke-Remote "bag_pid=`$(cat $remoteEpisode/bag.pid); kill -INT `$bag_pid; sleep 4; cd /home/jetson/jetson_asv_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; python3 /tmp/export_moving_target_bag.py --bag $remoteEpisode/bag --output $remoteEpisode/episode --task '$task' --slot $slot --layout $layout --motion S2 --standoff-m $distance --follow-slot-index $followSlot --embedding-key '$task'; tar -czf $archive -C $remoteEpisode episode"
        New-Item -ItemType Directory -Force -Path (Split-Path $localArchive) | Out-Null
        & scp -q "$sshTarget`:$archive" $localArchive
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $slot" }
        New-Item -ItemType Directory -Force -Path $localEpisode | Out-Null
        & tar -xzf $localArchive -C $localEpisode --strip-components=1
        if ($LASTEXITCODE -ne 0) { throw "extract failed for $slot" }
        $manifest = Get-Content (Join-Path $localEpisode "manifest.json") -Raw | ConvertFrom-Json
        Write-Host "COLLECT_PASS $slot frames=$($manifest.frame_count)"
    }
}
finally {
    & ssh -o BatchMode=yes $sshTarget "set +e; if test -f $remoteRoot/bridge.pid; then kill -INT `$(cat $remoteRoot/bridge.pid) 2>/dev/null; fi; pkill -f 'ros2 bag record' || true; true"
}

Write-Host "TIGHT_1M_COLLECTION_COMPLETE"
